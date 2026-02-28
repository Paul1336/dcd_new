from typing import Optional, Tuple, Dict, Iterator, Union
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from interfaces import RnnHiddenState, ObsTensor, RolloutBatch
try:
    from lempel_ziv_complexity import lempel_ziv_complexity
except ImportError:
    lempel_ziv_complexity = None


def to_tensor(a):
    if isinstance(a, dict):
        for k in a.keys():
            if isinstance(a[k], np.ndarray):
                a[k] = torch.from_numpy(a[k]).float()
    elif isinstance(a, np.ndarray):
        a = torch.from_numpy(a).float()
    elif isinstance(a, list):
        a = torch.tensor(a, dtype=torch.float)
    return a


def _flatten_helper(T, N, _tensor):
    if isinstance(_tensor, dict):
        return {k: _tensor[k].view(T * N, *_tensor[k].size()[2:]) for k in _tensor.keys()}
    else:
        return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self,
                model,
                num_steps, num_processes, observation_space, action_space,
                recurrent_hidden_state_size, recurrent_arch='rnn',
                use_proper_time_limits=False,
                use_popart=False,
                device='cpu'):
        """
        Allocate all rollout buffers for T steps across N parallel environments.

        Buffers (T = num_steps, N = num_processes):

          Size T+1 (include episode-start and bootstrap slot):
            obs                     [T+1, N, *obs_shape]
            recurrent_hidden_states [T+1, N, H]  
            masks                   [T+1, N, 1]           — 0.0 after a terminal state (resets RNN hidden state), else 1.0.
            bad_masks               [T+1, N, 1]           — 0.0 when episode ended by time limit (not true terminal), else 1.0.
            cliffhanger_masks       [T+1, N, 1]           — 0.0 at steps where the rollout cuts mid-episode, else 1.0.
            value_preds             [T+1, N, 1]           — V(s_t) from critic.
            returns                 [T+1, N, 1]           — GAE targets; groundtruth of V(s_t).

          Size T (only collected transitions):
            rewards                 [T, N, 1]            
            actions                 [T, N, action_shape]  
            action_log_probs        [T, N, 1]             
            action_log_dist         [T, N, n_actions]    
            level_seeds             [T, N, 1]             — level seed active at each step; used by UED/SFL tracking.

          Optional (allocated only when use_proper_time_limits=True):
            truncated_obs           same shape as obs     — obs at time-limit boundaries for correct bootstrap.
            truncated_value_preds   same shape as value_preds — value estimates at time-limit boundaries.

        Args:
            model: actor-critic model; must expose get_value() when
                use_proper_time_limits is True.
            num_steps: int — T, number of environment steps per rollout.
            num_processes: int — N, number of parallel environments.
            observation_space: gym.Space or dict of gym.Space.
            action_space: gym.Space — Discrete or Box.
            recurrent_hidden_state_size: int — H, the size of one hidden state vector.
            recurrent_arch: str — 'rnn' or 'lstm'.  Determines whether hidden
                states are stored as a flat tensor or packed as [h|c].
            use_proper_time_limits: bool — if True, allocate a truncated_obs
                buffer and a truncated_value_preds buffer used to correct
                value bootstrap at time-limit episode boundaries.
            use_popart: bool — if True, value predictions are normalised by a
                PopArt layer and must be de-normalised before computing TD
                targets.
            device.
        """

        self.device = device
        self.model = model
        self.num_processes = num_processes
        self.recurrent_arch = recurrent_arch
        self.recurrent_hidden_state_size = recurrent_hidden_state_size
        self.is_lstm = recurrent_arch == 'lstm'
        recurrent_hidden_state_buffer_size = 2*recurrent_hidden_state_size if self.is_lstm \
            else recurrent_hidden_state_size
        self.use_proper_time_limits = use_proper_time_limits
        self.use_popart = use_popart

        self.truncated_obs = None
        if isinstance(observation_space, dict):
            self.is_dict_obs = True
            self.obs = {k:torch.zeros(num_steps + 1, num_processes, *(observation_space[k]).shape) \
                for k,obs in observation_space.items()}

            if self.use_proper_time_limits:
                self.truncated_obs = {k:torch.zeros(num_steps + 1, num_processes, *(observation_space[k]).shape) \
                    for k,obs in observation_space.items()}
        else:
            self.is_dict_obs = False
            self.obs = torch.zeros(num_steps + 1, num_processes, *observation_space.shape)

            if self.use_proper_time_limits:
                self.truncated_obs = torch.zeros_like(self.obs)

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_buffer_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
            self.action_log_dist = torch.zeros(num_steps, num_processes, action_space.n)
        else:
            action_shape = action_space.shape[0]
            self.action_log_dist = torch.zeros(num_steps, num_processes, 1)

        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()

        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
        self.cliffhanger_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.truncated_value_preds = None
        if self.use_proper_time_limits:
            self.truncated_value_preds = torch.zeros_like(self.value_preds)

        self.denorm_value_preds = None

        self.level_seeds = torch.zeros(num_steps, num_processes, 1, dtype=torch.int)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device: torch.device) -> None:
        """
        Move all buffers in-place to the given device and update self.device.

        Handles both plain tensor and dict-obs layouts, and conditionally moves
        truncated_obs / truncated_value_preds when use_proper_time_limits is True.

        Args:
            device: torch.device or str — target device (e.g. 'cuda:0', 'cpu').

        Returns:
            None
        """
        self.device = device

        if self.is_dict_obs:
            for k, obs in self.obs.items():
                self.obs[k] = obs.to(device)
        else:
            self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.action_log_dist = self.action_log_dist.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.cliffhanger_masks = self.cliffhanger_masks.to(device)
        self.level_seeds = self.level_seeds.to(device)

        if self.use_proper_time_limits:
            if self.is_dict_obs:
                for k, obs in self.truncated_obs.items():
                    self.truncated_obs[k] = obs.to(device)
            else:
                self.truncated_obs = self.truncated_obs.to(device)

            self.truncated_value_preds = self.truncated_value_preds.to(device)

    def get_obs(self, idx: int) -> ObsTensor:
        if self.is_dict_obs:
            return {k: self.obs[k][idx] for k in self.obs.keys()}
        else:
            return self.obs[idx]

    def set_obs(self, obs: ObsTensor, step: int) -> None:
        if self.is_dict_obs:
            [self.obs[k][step].copy_(obs[k]) for k in self.obs.keys()]
        else:
            self.obs[step].copy_(obs)

    def insert(
        self,
        obs: ObsTensor,
        recurrent_hidden_states: RnnHiddenState,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        action_log_dist: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        bad_masks: Optional[torch.Tensor] = None,
        level_seeds: Optional[torch.Tensor] = None,
        cliffhanger_masks: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            obs: ObsTensor
            recurrent_hidden_states: RnnHiddenState 
            actions: torch.Tensor [N, act_shape]
            action_log_probs: torch.Tensor [N, 1] — log π(a|s).
            action_log_dist: torch.Tensor [N, n_act] (Discrete) or [N, 1] (continuous).
            value_preds: torch.Tensor [N, 1] — V(s) estimates.
            rewards: torch.Tensor [N, 1] or [N] or [N, 1, 1] — scalar rewards;
                squeezed to [N, 1] internally if 3-D.
            masks: torch.Tensor [N, 1] — 0 if the episode ended, 1 otherwise.
            bad_masks: Optional[torch.Tensor] [N, 1] — 0 if the episode ended
                due to a time-limit truncation, 1 for true terminals.  Defaults
                to ones when None.
            level_seeds: Optional[torch.Tensor] [N, 1] — integer seed of the
                level currently running in each environment.
            cliffhanger_masks: Optional[torch.Tensor] [N, 1] — auxiliary mask
                for cliffhanger bookkeeping; left unchanged when None.
        """
        rewards = to_tensor(rewards)
        if rewards.dim() == 3: rewards = rewards.squeeze(2)
        if rewards.dim() == 1: rewards = rewards.unsqueeze(-1)
        if bad_masks is None:
            bad_masks = torch.ones_like(masks)

        if self.is_dict_obs:
            [self.obs[k][self.step + 1].copy_(obs[k]) for k in self.obs.keys()]
        else:
            self.obs[self.step + 1].copy_(obs)

        if self.is_lstm:
            self.recurrent_hidden_states[self.step +1,:,
                :self.recurrent_hidden_state_size].copy_(recurrent_hidden_states[0])
            self.recurrent_hidden_states[self.step +1,:,
                self.recurrent_hidden_state_size:].copy_(recurrent_hidden_states[1])
        else:
            self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)

        self.actions[self.step].copy_(actions) 
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.action_log_dist[self.step].copy_(action_log_dist)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        if cliffhanger_masks is not None:
            self.cliffhanger_masks[self.step + 1].copy_(cliffhanger_masks)

        if level_seeds is not None:
            self.level_seeds[self.step].copy_(level_seeds)

        self.step = (self.step + 1) % self.num_steps

    def insert_truncated_obs(self, obs: ObsTensor, index: int) -> None:
        """
        Args:
            obs: ObsTensor — the observation at the truncation boundary for
                environment index.
            index: int — the environment index in [0, N-1] for which the
                truncated observation applies.
        """
        if self.is_dict_obs:
            [self.truncated_obs[k][self.step + 1][index].copy_(
                to_tensor(obs[k])) for k in self.truncated_obs.keys()]
        else:
            self.truncated_obs[self.step + 1][index].copy_(to_tensor(obs))

    def after_update(self) -> None:
        if self.is_dict_obs:
            [self.obs[k][0].copy_(self.obs[k][-1]) for k in self.obs.keys()]
        else:
            self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.cliffhanger_masks[0].copy_(self.cliffhanger_masks[-1])

    def replace_final_return(self, returns: torch.Tensor) -> None:
        self.rewards[-1] = returns

    def _compute_truncated_value_preds(self):
        """
        Returns:
            torch.Tensor [T+1, N, 1] — self.truncated_value_preds with
                corrected V(s) values at truncated steps and original V(s)
                values everywhere else.
        """
        self.truncated_value_preds.copy_(self.value_preds)
        with torch.no_grad():
            # For each process, forward truncated obs
            for i in range(self.num_processes):
                steps = (self.bad_masks[:,i,0] == 0).nonzero().squeeze()
                if len(steps.shape) == 0 or steps.shape[0] == 0:
                    continue

                if self.is_dict_obs:
                    obs = {k:self.truncated_obs[k][steps.squeeze(), i, :] 
                        for k in self.truncated_obs.keys()}
                else:
                    obs = self.truncated_obs[steps.squeeze(),i,:]

                rnn_hxs = self.recurrent_hidden_states[steps,i,:]
                if self.is_lstm:
                    rnn_hxs = self._split_batched_lstm_recurrent_hidden_states(rnn_hxs)
                masks = torch.ones((len(steps), 1), device=self.device)
                value_preds = self.model.get_value(obs, rnn_hxs, masks)

                self.truncated_value_preds[steps,i,:] = value_preds

        return self.truncated_value_preds

    def compute_gae_returns(
        self,
        returns_buffer: torch.Tensor,
        next_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """
        Args:
            returns_buffer: torch.Tensor [T+1, N, 1] — unused directly; the
                results are always written into self.returns.  Present for API
                symmetry with compute_discounted_returns.
            next_value: torch.Tensor [N, 1] — V(s_{T+1}), the bootstrap value
                estimated at the end of the rollout.
            gamma: float — discount factor.
            gae_lambda: float — GAE smoothing parameter (0 = TD(0),
                1 = Monte-Carlo).
        """
        self.value_preds[-1] = next_value
        gae = 0
        value_preds = self.value_preds

        if self.use_proper_time_limits:
            # Get truncated value preds
            self._compute_truncated_value_preds()
            value_preds = self.truncated_value_preds

        if self.use_popart:
            self.denorm_value_preds = self.model.popart.denormalize(value_preds) # denormalize all value predictions
            value_preds = self.denorm_value_preds

        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + \
                gamma*value_preds[step + 1]*self.masks[step + 1] - value_preds[step]

            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + value_preds[step]

    def compute_discounted_returns(
        self,
        returns_buffer: torch.Tensor,
        next_value: torch.Tensor,
        gamma: float,
    ) -> None:
        """
        Args:
            returns_buffer: torch.Tensor [T+1, N, 1] — buffer to write
                returns into (typically self.returns).
            next_value: torch.Tensor [N, 1] — V(s_{T+1}), the bootstrap value
                estimated at the end of the rollout; written to
                self.value_preds[-1].
            gamma: float — discount factor.
        """
        self.value_preds[-1] = next_value
        value_preds = self.value_preds

        if self.use_proper_time_limits:    
            self._compute_truncated_value_preds()
            value_preds = self.truncated_value_preds

        if self.use_popart:
            self.denorm_value_preds = self.model.popart.denormalize(value_preds) # denormalize all value predictions

        self.returns[-1] = value_preds[-1]

        for step in reversed(range(self.rewards.size(0))):
            returns_buffer[step] = returns_buffer[step + 1] * \
                gamma * self.masks[step + 1] + self.rewards[step]

    def compute_returns(
        self,
        next_value: torch.Tensor,
        use_gae: bool,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """
        Args:
            next_value: torch.Tensor [N, 1] — bootstrap value V(s_{T+1})
                estimated at the end of the rollout.
            use_gae: bool — if True use GAE; if False use plain discounted
                returns.
            gamma: float — discount factor.
            gae_lambda: float — GAE lambda (only used when use_gae=True).
        """
        if use_gae:
            self.compute_gae_returns(
                self.returns, next_value, gamma, gae_lambda)
        else:
            self.compute_discounted_returns(
                self.returns, next_value, gamma)

    def get_batched_value_loss(
        self,
        signed: bool = False,
        positive_only: bool = False,
        power: int = 1,
        clipped: bool = True,
        batched: bool = True,
    ) -> Union[torch.Tensor, float]:
        """
        Args:
            signed: bool — if True return signed TD error (returns - V);
                if False (default) return absolute TD error |returns - V|.
            positive_only: bool — if True clamp negative TD errors to 0 so
                only positive surprises are counted.  Overridden by signed
                when both are True.
            power: int — exponentiate the TD error to this power before
                averaging (e.g. power=2 gives MSE-like loss).
            clipped: bool — if True clamp the per-env mean TD error to [-1, 1]
                before returning.
            batched: bool — if True return per-env scores; if False return a
                single scalar mean over all environments.
        Returns:
            Union[torch.Tensor, float] —
                batched=True:  torch.Tensor [N, 1] — per-environment mean TD
                    error (optionally clipped).
                batched=False: float — scalar mean over all environments.
        """

        # If agent uses popart, then value_preds are normalized, while 
        # returns are not.
        if self.use_popart:
            value_preds = self.denorm_value_preds[:-1]
        else:
            value_preds = self.value_preds[:-1]

        returns = self.returns[:-1]

        if signed:
            td = returns - value_preds
        elif positive_only:
            td = (returns - value_preds).clamp(0)
        else:
            td = (returns - value_preds).abs()
        if power > 1:
            td = td**power

        batch_td = td.mean(0) # B x 1

        if clipped:
            batch_td = torch.clamp(batch_td, -1, 1) 
        
        if batched:
            return batch_td
        else:
            return batch_td.mean().item()

    def get_batched_action_complexity(self) -> torch.Tensor:
        """
        Requires the lempel_ziv_complexity package to be installed.
        Returns:
            torch.Tensor [N, 1] — mean LZ complexity score across all
                episodes seen in the buffer for each of the N environments.
        """
        num_processes = self.actions.shape[1]
        batched_complexity = torch.zeros(num_processes, 1, dtype=torch.float)
        for b in range(num_processes):
            num_traj = 0
            avg_complexity = 0
            done_steps = [0] + (self.masks[:,b,0] == 0).nonzero().flatten().tolist()
            for i, t in enumerate(done_steps[:-1]):
                if len(done_steps) > 1:
                    next_done = done_steps[i+1]
                else:
                    next_done = self.actions.shape[0]
                action_str = ' '.join([str(a.item()) for a in self.actions[t:next_done,b,0]])
                avg_complexity += lempel_ziv_complexity(action_str)
                num_traj += 1
            batched_complexity[b] = avg_complexity/num_traj

        return batched_complexity

    def get_action_complexity(self) -> float:
        """
        Requires the lempel_ziv_complexity package to be installed.
        Returns:
            float — mean LZ complexity averaged over every episode trajectory
                across all N environments in the buffer.
        """
        num_processes = self.actions.shape[1]
        avg_complexity = 0
        num_traj = 0
        for b in range(num_processes):
            done_steps = [0] + (self.masks[:,b,0] == 0).nonzero().flatten().tolist()
            for i, t in enumerate(done_steps[:-1]):
                if len(done_steps) > 1:
                    next_done = done_steps[i+1]
                else:
                    next_done = self.actions.shape[0]
                action_str = ' '.join([str(a.item()) for a in self.actions[t:next_done,b,0]])
                avg_complexity += lempel_ziv_complexity(action_str)
                num_traj += 1
    
        return avg_complexity/num_traj

    def get_action_traj(self, as_string: bool = False) -> Union[list, torch.Tensor]:
        """
        Args:
            as_string: bool — if False (default) return the raw action tensor
                with the trailing size-1 dimension squeezed out.  If True,
                return a list of space-separated action strings, one per
                environment.
        Returns:
            Union[torch.Tensor, list] —
                as_string=False: torch.Tensor [T, N] (long) — raw actions for
                    all T steps and N environments.
                as_string=True:  list[str] of length N — each string is the
                    sequence of integer action values for one environment,
                    separated by spaces.
        """
        if as_string:
            num_processes = self.actions.shape[1]
            traj = []
            for b in range(num_processes):
                action_str = ' '.join([str(a.item()) for a in self.actions[:,b,0]])
                traj.append(action_str)
            return traj
        else:
            return self.actions.squeeze(-1)

    def _split_batched_lstm_recurrent_hidden_states(
        self, hxs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hxs: torch.Tensor [N, 2*H] — packed hidden state where the first
                H columns are h (hidden state) and the last H columns are c
                (cell state).
        Returns:
            Tuple[torch.Tensor, torch.Tensor] —
                (h, c) where each is torch.Tensor [N, H].
        """
        return (hxs[:, :self.recurrent_hidden_state_size],
                hxs[:, self.recurrent_hidden_state_size:])

    def get_recurrent_hidden_state(self, step: int) -> RnnHiddenState:
        """
        Args:
            step: int — time index in [0, T].  Index 0 is the hidden state at
                the start of the rollout; index T is the hidden state after the
                final step.
        Returns:
            RnnHiddenState —
                RNN:  torch.Tensor [N, H] — the hidden state at step.
                LSTM: Tuple[torch.Tensor, torch.Tensor] — (h, c) each
                    torch.Tensor [N, H].
        """
        if self.is_lstm:
            return self._split_batched_lstm_recurrent_hidden_states(
                    self.recurrent_hidden_states[step,:].squeeze(0))
        return self.recurrent_hidden_states[step]

    def feed_forward_generator(
        self,
        advantages: Optional[torch.Tensor],
        num_mini_batch: Optional[int] = None,
        mini_batch_size: Optional[int] = None,
    ) -> Iterator[RolloutBatch]:
        """
        Args:
            advantages: Optional[torch.Tensor] [T, N, 1] — pre-computed
                advantage estimates.  If None, adv_targ in the yielded batch
                will also be None.
            num_mini_batch: Optional[int] — number of mini-batches to split
                the rollout into.  Mutually exclusive with mini_batch_size.
            mini_batch_size: Optional[int] — explicit size of each mini-batch
                in transitions.  Overrides num_mini_batch when provided.

        Yields:
            RolloutBatch — tuple of:
                obs_batch: ObsTensor [B, *obs_shape] or dict thereof.
                recurrent_hidden_states_batch: RnnHiddenState [B, H] (RNN) or
                    Tuple[[B, H], [B, H]] (LSTM).
                actions_batch: torch.Tensor [B, act_shape].
                value_preds_batch: torch.Tensor [B, 1].
                return_batch: torch.Tensor [B, 1].
                masks_batch: torch.Tensor [B, 1].
                old_action_log_probs_batch: torch.Tensor [B, 1].
                adv_targ: Optional[torch.Tensor] [B, 1].
            where B = mini_batch_size.
        """
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=False)
     
        for indices in sampler:
            if self.is_dict_obs:
                obs_batch = {k: self.obs[k][:-1].view(-1, *self.obs[k].size()[2:])[indices] for k in self.obs.keys()}
            else:
                obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]

            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]

            actions_batch = self.actions.view(-1,
                                            self.actions.size(-1))[indices]

            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]

            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            if self.is_lstm: 
                # Split into (hxs, cxs) for LSTM
                recurrent_hidden_states_batch = \
                    self._split_batched_lstm_recurrent_hidden_states(recurrent_hidden_states_batch)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(
        self,
        advantages: torch.Tensor,
        num_mini_batch: int,
    ) -> Iterator[RolloutBatch]:
        """
        Args:
            advantages: torch.Tensor [T, N, 1] — pre-computed advantage
                estimates.
            num_mini_batch: int — number of mini-batches.  Must satisfy
                N >= num_mini_batch so that each batch contains at least one
                environment.

        Yields:
            RolloutBatch — tuple of:
                obs_batch: ObsTensor [T*B, *obs_shape] or dict thereof,
                    where B = num_envs_per_batch.
                recurrent_hidden_states_batch: RnnHiddenState [B, H] (RNN) or
                    Tuple[[B, H], [B, H]] (LSTM) — initial hidden states only.
                actions_batch: torch.Tensor [T*B, act_shape].
                value_preds_batch: torch.Tensor [T*B, 1].
                return_batch: torch.Tensor [T*B, 1].
                masks_batch: torch.Tensor [T*B, 1].
                old_action_log_probs_batch: torch.Tensor [T*B, 1].
                adv_targ: torch.Tensor [T*B, 1].
        """
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)

        for start_ind in range(0, num_processes, num_envs_per_batch):
            if self.is_dict_obs:
                obs_batch = defaultdict(list)
            else:
                obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                if self.is_dict_obs:
                    [obs_batch[k].append(self.obs[k][:-1,ind]) for k in self.obs.keys()]
                else:
                    obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            if self.is_dict_obs:
                for k in obs_batch.keys():
                    obs_batch[k] = torch.stack(obs_batch[k],1)
            else:
                obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            if self.is_lstm: 
                # Split into (hxs, cxs) for LSTM
                recurrent_hidden_states_batch = \
                    self._split_batched_lstm_recurrent_hidden_states(recurrent_hidden_states_batch)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
