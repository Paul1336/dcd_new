# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# PPO implementation adapted from:
# Copyright (c) 2017 Ilya Kostrikov (MIT License)
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from interfaces import RnnHiddenState, ObsTensor, PPOUpdateInfo, ActOutput


class ACAgent:
    """
    Actor-Critic Agent.

    Owns the network (actor_critic), optimizer, and rollout storage.
    Exposes a clean interface to runners and evaluators — no internal
    structure (algo / actor_critic) needs to be accessed from outside.
    """

    def __init__(
        self,
        actor_critic,
        storage,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        kl_loss_coef=0.0,
        lr=None,
        eps=None,
        max_grad_norm=None,
        clip_value_loss=True,
        log_grad_norm=False,
    ):
        """
        Construct an ACAgent, wiring together the network, storage, and optimizer.

        Args:
            actor_critic: nn.Module — the actor-critic network; must expose
                act(), get_value(), evaluate_actions(), is_recurrent,
                recurrent_hidden_state_size, and optionally process_action().
            storage: RolloutStorage — rollout buffer used to accumulate
                transitions and produce mini-batches for PPO updates.
            clip_param: float — PPO clipping epsilon; ratio is clamped to
                [1 - clip_param, 1 + clip_param].
            ppo_epoch: int — number of full passes over the rollout buffer
                per call to update().
            num_mini_batch: int — number of mini-batches to split each epoch
                into; effective batch size = (num_processes * num_steps) /
                num_mini_batch.
            value_loss_coef: float — scalar weight applied to the value loss
                term in the total loss.
            entropy_coef: float — scalar weight applied to the entropy bonus
                (subtracted from total loss to encourage exploration).
            kl_loss_coef: float — scalar weight applied to the optional KL
                divergence regularisation loss; disabled when 0.0 (default).
            lr: float or None — initial Adam learning rate; passed directly
                to optim.Adam.
            eps: float or None — Adam epsilon for numerical stability; passed
                directly to optim.Adam.
            max_grad_norm: float or None — if positive, gradient L2 norm is
                clipped to this value before each optimizer step; set None or
                0 to disable.
            clip_value_loss: bool — if True, the value loss is computed as the
                max of clipped and unclipped squared errors (PPO-style); if
                False, plain MSE is used.
            log_grad_norm: bool — if True, the L2 gradient norm is recorded
                after every mini-batch and returned in info["grad_norms"].
        """
        self.actor_critic = actor_critic
        self.storage = storage

        # PPO hyperparams
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.clip_value_loss = clip_value_loss
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.kl_loss_coef = kl_loss_coef
        self.max_grad_norm = max_grad_norm
        self.log_grad_norm = log_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    def act(
        self,
        x: ObsTensor,
        rnn_hxs: RnnHiddenState,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> ActOutput:
        return self.actor_critic.act(x, rnn_hxs, masks, deterministic)

    def get_value(
        self,
        x: ObsTensor,
        rnn_hxs: RnnHiddenState,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.actor_critic.get_value(x, rnn_hxs, masks)

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        if hasattr(self.actor_critic, "process_action"):
            return self.actor_critic.process_action(action)
        return action

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #

    def insert(self, *args, **kwargs) -> None:
        return self.storage.insert(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Device / mode
    # ------------------------------------------------------------------ #

    def to(self, device: torch.device) -> "ACAgent":
        self.actor_critic.to(device)
        self.storage.to(device)
        return self

    def train(self) -> None:
        self.actor_critic.train()

    def eval(self) -> None:
        self.actor_critic.eval()

    def random(self) -> None:
        self.actor_critic.random = True

    # ------------------------------------------------------------------ #
    # Optimizer
    # ------------------------------------------------------------------ #

    def update_lr(self, lr: float) -> None:
        print("anneal lr to: ", lr)
        self.optimizer.param_groups[0]["lr"] = lr

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def is_recurrent(self):
        return self.actor_critic.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        return self.actor_critic.recurrent_hidden_state_size

    @property
    def is_lstm(self):
        return (
            self.is_recurrent
            and hasattr(self.actor_critic, "rnn")
            and self.actor_critic.rnn.arch == "lstm"
        )

    # ------------------------------------------------------------------ #
    # Training update (PPO)
    # ------------------------------------------------------------------ #

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor_critic": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.actor_critic.load_state_dict(state["actor_critic"])
        self.optimizer.load_state_dict(state["optimizer"])

    def _grad_norm(self) -> float:
        total_norm = 0
        for p in self.actor_critic.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def update(
        self,
        discard_grad: bool = False,
        kl_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float, float, PPOUpdateInfo]:
        """
        Args:
            discard_grad: bool — if True, loss.backward() is still called
                (for logging purposes) but optimizer.step() is skipped;
                also suppresses the KL loss even when kl_dict is provided.
            kl_dict: dict or None — when provided and kl_loss_coef > 0, must
                contain:
                    "antagonist_model": nn.Module — a frozen copy of the
                        policy whose distribution is the reference for KL
                        divergence regularisation (KL(antagonist || current)).

        Returns:
            Tuple of:
                value_loss:    float — mean value loss averaged over all
                               mini-batch updates (ppo_epoch * num_mini_batch).
                action_loss:   float — mean surrogate policy loss (negative
                               clipped ratio objective) averaged over all
                               mini-batch updates.
                dist_entropy:  float — mean action-distribution entropy
                               averaged over all mini-batch updates.
                info:          PPOUpdateInfo — diagnostic scalars:
                    "grad_norms":  list[float]  — per-mini-batch gradient L2
                                   norms; present only when log_grad_norm=True.
                    "approx_kl":   float        — approximate KL divergence
                                   between old and new policy from the last
                                   mini-batch of the last completed epoch
                                   (computed as E[(r-1) - log r]); omitted if
                                   no mini-batches were processed.
                    "clipfracs":   float        — mean fraction of samples
                                   whose probability ratio fell outside the
                                   clip range across all mini-batches.
                    "used_epoch":  int          — number of epochs actually
                                   completed before early-stopping (may be
                                   less than ppo_epoch).
                    "kl_loss":     float        — mean KL regularisation loss
                                   averaged over all updates; present only
                                   when KL loss was active.
                    "lr":          float        — current Adam learning rate
                                   at the time of return.
        """
        rollouts = self.storage
        use_kl_loss = (
            kl_dict is not None
            and self.kl_loss_coef > 0.0
            and not discard_grad
        )

        if rollouts.use_popart:
            value_preds = rollouts.denorm_value_preds
        else:
            value_preds = rollouts.value_preds

        advantages = rollouts.returns[:-1] - value_preds[:-1]

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        kl_loss_epoch = 0.0

        grad_norms = []
        clipfracs = []
        used_epoch = 0
        approx_kl = None

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch
                )
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch
                )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                if use_kl_loss:
                    values, action_log_probs, dist_entropy, _, dist_protagonist = (
                        self.actor_critic.evaluate_actions(
                            obs_batch,
                            recurrent_hidden_states_batch,
                            masks_batch,
                            actions_batch,
                            return_policy_logits=True,
                        )
                    )
                    with torch.no_grad():
                        _, _, _, _, dist_antagonist = (
                            kl_dict["antagonist_model"].evaluate_actions(
                                obs_batch,
                                recurrent_hidden_states_batch,
                                masks_batch,
                                actions_batch,
                                return_policy_logits=True,
                            )
                        )
                else:
                    values, action_log_probs, dist_entropy, _ = (
                        self.actor_critic.evaluate_actions(
                            obs_batch,
                            recurrent_hidden_states_batch,
                            masks_batch,
                            actions_batch,
                        )
                    )

                logratio = action_log_probs - old_action_log_probs_batch
                ratio = torch.exp(logratio)
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    frac = (
                        (ratio - 1.0).abs() > self.clip_param
                    ).float().mean().item()
                    clipfracs.append(frac)

                if rollouts.use_popart:
                    self.actor_critic.popart.update(return_batch)
                    return_batch = self.actor_critic.popart.normalize(return_batch)

                if self.clip_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_loss = 0.5 * torch.max(
                        (values - return_batch).pow(2),
                        (value_pred_clipped - return_batch).pow(2),
                    ).mean()
                else:
                    value_loss = 0.5 * ((values - return_batch) ** 2).mean()

                loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                )

                if use_kl_loss:
                    kl_div = torch.distributions.kl.kl_divergence(
                        dist_antagonist, dist_protagonist
                    )
                    kl_loss = kl_div.sum() / kl_div.shape[0]
                    loss += self.kl_loss_coef * kl_loss
                    kl_loss_epoch += kl_loss.item()

                self.optimizer.zero_grad()
                loss.backward()

                if self.log_grad_norm:
                    grad_norms.append(self._grad_norm())

                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.max_grad_norm
                    )

                if not discard_grad:
                    self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

            if approx_kl is not None and approx_kl > 0.03:
                print("break due to approx_kl: ", approx_kl, "e: ", e)
                break

            used_epoch += 1

        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        if use_kl_loss:
            kl_loss_epoch /= num_updates

        info = {}
        if self.log_grad_norm:
            info["grad_norms"] = grad_norms
        if approx_kl is not None:
            info["approx_kl"] = approx_kl.item()
        if clipfracs:
            info["clipfracs"] = np.mean(clipfracs)
        info["used_epoch"] = used_epoch
        if use_kl_loss:
            info["kl_loss"] = kl_loss_epoch
        info["lr"] = self.optimizer.param_groups[0]["lr"]

        rollouts.after_update()
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, info
