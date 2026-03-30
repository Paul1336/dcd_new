import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DeviceAwareModule, RNN, Categorical, layer_init


class MultigridNetwork(DeviceAwareModule):
    """Actor-Critic: flat obs (image + direction one-hot) → MLP → LSTM → heads.

    Replaces the original CNN-based encoder with the lightweight MLP architecture
    validated in the ppo_crossing_baseline script.
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 mlp_hidden=64,
                 recurrent_arch='lstm',
                 recurrent_hidden_size=128):
        super().__init__()

        self.action_space = action_space
        num_actions = action_space.n

        # Flat obs: C*H*W (channels-first image) + 4 (direction one-hot)
        img_shape = observation_space['image'].shape   # (C, H, W)
        obs_dim = int(np.prod(img_shape)) + 4

        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, mlp_hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(mlp_hidden, mlp_hidden)),
            nn.Tanh(),
        )

        self.rnn = None
        rnn_out = mlp_hidden
        if recurrent_arch:
            self.rnn = RNN(input_size=mlp_hidden,
                           hidden_size=recurrent_hidden_size,
                           arch=recurrent_arch)
            rnn_out = recurrent_hidden_size

        self.actor  = Categorical(rnn_out, num_actions)
        self.critic = layer_init(nn.Linear(rnn_out, 1), std=1.0)

        self.train()

    @property
    def is_recurrent(self):
        return self.rnn is not None

    @property
    def recurrent_hidden_state_size(self):
        return self.rnn.recurrent_hidden_state_size if self.rnn is not None else 1

    def _flat_obs(self, inputs):
        image     = inputs['image']                         # (B, C, H, W)
        img_flat  = image.flatten(start_dim=1)              # (B, C*H*W)
        direction = inputs['direction'].reshape(-1)         # (B,)
        dir_oh    = F.one_hot(direction.long(), 4).float()  # (B, 4)
        return torch.cat([img_flat, dir_oh], dim=-1)        # (B, obs_dim)

    def _forward_base(self, inputs, rnn_hxs, masks):
        x = self._flat_obs(inputs)
        x = self.encoder(x)
        if self.rnn is not None:
            x, rnn_hxs = self.rnn(x, rnn_hxs, masks)
        return x, rnn_hxs

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        core, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)
        dist  = self.actor(core)
        value = self.critic(core)
        action = dist.mode() if deterministic else dist.sample()
        return value, action, dist.logits, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        core, _ = self._forward_base(inputs, rnn_hxs, masks)
        return self.critic(core)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, return_policy_logits=False):
        core, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)
        dist  = self.actor(core)
        value = self.critic(core)
        action_log_probs = dist.log_probs(action)
        dist_entropy     = dist.entropy().mean()
        if return_policy_logits:
            return value, action_log_probs, dist_entropy, rnn_hxs, dist
        return value, action_log_probs, dist_entropy, rnn_hxs
