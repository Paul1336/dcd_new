from typing import Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from .common import DeviceAwareModule, Categorical, layer_init
from interfaces import RnnHiddenState, ObsTensor, ActOutput


class IphyreNetwork(DeviceAwareModule):
    """
    Args:
        observation_space: Gym space. Expected shapes:
            (122,) for obs_type="symbolic",
            (D,)   for obs_type="embedding" where D is the VLM embedding dim,
            (H, W, C) for obs_type="image" (not yet supported).
        action_space: Discrete Gym space; action_space.n gives the number of
            available actions.
        obs_type: str — one of "symbolic", "embedding", "image". Default "symbolic".
    """

    def __init__(
        self,
        observation_space,
        action_space,
        obs_type: str = "symbolic",
        should_freeze_embedding: bool = False,
        use_ball_relative: bool = False,
    ):
        super(IphyreNetwork, self).__init__()

        self.rnn = None
        self.obs_type = obs_type
        self.should_freeze_embedding = should_freeze_embedding
        self.use_ball_relative = use_ball_relative

        print('obs_type:', obs_type, '  should_freeze_embedding:', should_freeze_embedding)

        if obs_type == "image":
            # TODO: plug in CNN backbone (e.g. IMPALA/ResNet) to produce embedding
            raise NotImplementedError(
                "obs_type='image' requires a CNN backbone that is not yet implemented. "
                "Use obs_type='embedding' to encode images upstream with a frozen CLIP encoder."
            )
        elif obs_type == "symbolic":
            embedding_dim = 16
            embedding_hidden_dim = 32
            embedding_hidden_action_dim = 8
            self.should_use_deep_set = True
            self.block_embed_layer = nn.Sequential(
                layer_init(nn.Linear(9, embedding_hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(embedding_hidden_dim, embedding_dim)),
            )
            self.action_embed_layer = nn.Sequential(
                layer_init(nn.Linear(2, embedding_hidden_action_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(embedding_hidden_action_dim, embedding_dim)),
            )
            shape = np.array([embedding_dim])
        elif obs_type == "embedding":
            self.should_use_deep_set = False
            shape = np.array(observation_space.shape)
        else:
            raise ValueError(
                f"Unknown obs_type '{obs_type}'. Choose from 'symbolic', 'embedding', 'image'."
            )

        self.shape = shape
        self.action_num = action_space.n

        print('Action num:', self.action_num, 'Observation space:', observation_space.shape)
        print('Action space:', action_space)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(shape.prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(shape.prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            Categorical(256, action_space.n)
        )

    # def reset_critic(self):
    #     """Re-initialize critic head for fine-tuning on a new task."""
    #     self.critic = nn.Sequential(
    #         layer_init(nn.Linear(self.shape.prod(), 256)),
    #         nn.Tanh(),
    #         layer_init(nn.Linear(256, 256)),
    #         nn.Tanh(),
    #         layer_init(nn.Linear(256, 1), std=1.0),
    #     )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, obs_dim] — observation batch.
                obs_dim = 122 for obs_type="symbolic" (12*9 + 7*2).
                obs_dim = D   for obs_type="embedding" (e.g. 512 for ViT-B/32).

        Returns:
            Tensor [B, embed_dim] — embedding fed to actor and critic heads.
                embed_dim = 16  for obs_type="symbolic".
                embed_dim = D   for obs_type="embedding" (same as obs_dim).
        """
        if self.should_use_deep_set:
            blocks = x[:, :12 * 9].reshape(-1, 12, 9).clone()
            actions = x[:, 12 * 9:12 * 9 + 7 * 2].reshape(-1, 7, 2).clone()

            if self.use_ball_relative:
                # Ball: x1==x2, y1==y2 (center duplicated). Find it by |x1-x2| < 1.
                # blocks: [B, 12, 9], features: [x1, y1, x2, y2, r, eli, dynamic, joint, spring]
                is_ball = (blocks[:, :, 0] - blocks[:, :, 2]).abs() < 1.0  # [B, 12]
                ball_idx = is_ball.float().argmax(dim=1)                    # [B]
                b_idx = torch.arange(blocks.size(0), device=blocks.device)
                ball_x = blocks[b_idx, ball_idx, 0]  # [B]
                ball_y = blocks[b_idx, ball_idx, 1]  # [B]
                # Shift all (x1, y1, x2, y2) to be ball-relative
                blocks[:, :, 0] -= ball_x.unsqueeze(1)
                blocks[:, :, 1] -= ball_y.unsqueeze(1)
                blocks[:, :, 2] -= ball_x.unsqueeze(1)
                blocks[:, :, 3] -= ball_y.unsqueeze(1)
                # Shift action positions (x, y) to be ball-relative
                actions[:, :, 0] -= ball_x.unsqueeze(1)
                actions[:, :, 1] -= ball_y.unsqueeze(1)
            if self.should_freeze_embedding:
                with torch.no_grad():
                    block_embeddings = self.block_embed_layer(blocks)
                    action_embeddings = self.action_embed_layer(actions)
                    aggregated_embedding = torch.sum(block_embeddings, dim=1) + torch.sum(action_embeddings, dim=1)
            else:
                block_embeddings = self.block_embed_layer(blocks)
                action_embeddings = self.action_embed_layer(actions)
                aggregated_embedding = torch.sum(block_embeddings, dim=1) + torch.sum(action_embeddings, dim=1)
            return aggregated_embedding
        else:
            return x



    def get_value(
        self,
        x: ObsTensor,
        rnn_hxs: RnnHiddenState,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, D] — flat observation batch.

        Returns:
            Tensor [B, 1] — scalar value estimate V(s) for each observation in
                the batch, produced by the critic head
        """
        embedding = self.get_embedding(x)
        return self.critic(embedding)

    # def get_logits(self, x: torch.Tensor) -> torch.Tensor:
    #     """Return the pre-Categorical hidden representation (256-dim) — for distillation/analysis."""
    #     embedding = self.get_embedding(x)
    #     hidden = embedding
    #     for layer in list(self.actor)[:-1]:  # all layers except the final Categorical head
    #         hidden = layer(hidden)
    #     return hidden

    def act(
        self,
        x: ObsTensor,
        rnn_hxs: RnnHiddenState,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> ActOutput:
        """
        Args:
            x: Tensor [B, D] — flat observation batch.
            rnn_hxs: RnnHiddenState — recurrent hidden state.
            masks: Tensor [B, 1] — done masks used to reset hidden state.
            deterministic: bool — if True, return the greedy action (dist.mode())
                instead of sampling. Default: False.

        Returns:
            value: Tensor [B, 1] — critic value estimate V(s).
            action: Tensor [B, 1] — selected action index (sampled or greedy).
            actor_log_dist: Tensor [B, action_num] — raw logits (log-probabilities
                before normalisation) over all actions from the Categorical head.
            rnn_hxs: RnnHiddenState
        """
        embedding = self.get_embedding(x)

        dist = self.actor(embedding)
        value = self.critic(embedding)

        action = dist.sample()
        if deterministic:
            action = dist.mode()

        actor_log_dist = dist.logits

        return value, action, actor_log_dist, rnn_hxs

    def evaluate_actions(
        self,
        x: ObsTensor,
        rnn_hxs: RnnHiddenState,
        masks: torch.Tensor,
        action: torch.Tensor,
        return_policy_logits: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RnnHiddenState],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RnnHiddenState, torch.distributions.Categorical],
    ]:
        """
        Args:
            x: Tensor [B, D] — flat observation batch from the rollout buffer.
            rnn_hxs: RnnHiddenState — recurrent hidden state.
            masks: Tensor [B, 1] — done masks used to reset hidden state.
            action: Tensor [B, 1] — actions that were taken during rollout collection.
            return_policy_logits: bool — if True, the FixedCategorical distribution
                object is appended as the 5th return value. Default: False.

        Returns:
            value: Tensor [B, 1] — critic value estimate V(s) under current parameters.
            action_log_probs: Tensor [B, 1] — log-probability of action under the
                current policy, log pi(a|s).
            dist_entropy: Tensor [] — scalar mean entropy of the action distribution
                across the batch, H(pi(·|s)).mean(), used as the PPO entropy bonus.
            rnn_hxs: RnnHiddenState.
            dist (when return_policy_logits=True): FixedCategorical — the full
                action distribution object.
        """
        embedding = self.get_embedding(x)

        dist = self.actor(embedding)
        value = self.critic(embedding)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        if return_policy_logits:
            return value, action_log_probs, dist_entropy, rnn_hxs, dist

        return value, action_log_probs, dist_entropy, rnn_hxs
    
    @property
    def is_recurrent(self):
        return self.rnn is not None

    @property
    def recurrent_hidden_state_size(self):
        if self.rnn is not None:
            return self.rnn.recurrent_hidden_state_size
        else:
            return 0