# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DeviceAwareModule, RNN, Categorical


# ── Utilities (ported from archive/models/common.py) ──────────────────────────

def _init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

_init_ = lambda m: _init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
_init_tanh_ = lambda m: _init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))


def apply_init_(modules, gain=None):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=gain) if gain else nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Conv2d_tf(nn.Conv2d):
    """Conv2d with TF-style padding ('valid' or 'same')."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding_mode_tf = kwargs.get('padding', 'same')

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total = max(0, (out_size - 1) * self.stride[dim] + effective - input_size)
        return int(total % 2 != 0), total

    def forward(self, input):
        if self._padding_mode_tf == 'valid':
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            padding=0, dilation=self.dilation, groups=self.groups)
        rows_odd, pad_rows = self._compute_padding(input, dim=0)
        cols_odd, pad_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        padding=(pad_rows // 2, pad_cols // 2),
                        dilation=self.dilation, groups=self.groups)


def one_hot(dim, inputs, device='cpu'):
    return F.one_hot(inputs.long(), dim).squeeze(1).float()


def make_fc_layers_with_hidden_sizes(sizes, input_size):
    layers = []
    for i, layer_size in enumerate(sizes[:-1]):
        in_sz = input_size if i == 0 else sizes[i]
        out_sz = sizes[i + 1]
        layers.append(_init_tanh_(nn.Linear(in_sz, out_sz)))
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


# ── Main network ──────────────────────────────────────────────────────────────

class MultigridNetwork(DeviceAwareModule):
    """Actor-Critic network for MultiGrid environments."""

    def __init__(self,
                 observation_space,
                 action_space,
                 actor_fc_layers=(32, 32),
                 value_fc_layers=(32, 32),
                 conv_filters=16,
                 conv_kernel_size=3,
                 scalar_fc=5,
                 scalar_dim=4,
                 random_z_dim=0,
                 xy_dim=0,
                 recurrent_arch='lstm',
                 recurrent_hidden_size=256,
                 random=False):
        super().__init__()

        self.random = random
        self.action_space = action_space
        num_actions = action_space.n

        # Image encoder
        obs_shape = observation_space['image'].shape
        m = obs_shape[-2]   # H
        n = obs_shape[-1]   # W (= 3 channels for grid encoding)
        c = obs_shape[-3]   # channels

        self.image_conv = nn.Sequential(
            Conv2d_tf(3, conv_filters, kernel_size=conv_kernel_size, stride=1, padding='valid'),
            nn.Flatten(),
            nn.ReLU(),
        )
        self.image_embedding_size = (n - conv_kernel_size + 1) * (m - conv_kernel_size + 1) * conv_filters
        self.preprocessed_input_size = self.image_embedding_size

        # x, y positional embeddings
        self.xy_embed = None
        self.xy_dim = xy_dim
        if xy_dim:
            self.xy_embed = nn.Linear(xy_dim, xy_dim)
            self.preprocessed_input_size += 2 * xy_dim

        # Scalar (direction / timestep) embedding
        self.scalar_embed = None
        self.scalar_dim = scalar_dim
        if scalar_dim:
            self.scalar_embed = nn.Linear(scalar_dim, scalar_fc)
            self.preprocessed_input_size += scalar_fc

        self.preprocessed_input_size += random_z_dim
        self.base_output_size = self.preprocessed_input_size

        # RNN
        self.rnn = None
        if recurrent_arch:
            self.rnn = RNN(input_size=self.preprocessed_input_size,
                           hidden_size=recurrent_hidden_size,
                           arch=recurrent_arch)
            self.base_output_size = recurrent_hidden_size

        # Policy and value heads
        self.actor = nn.Sequential(
            make_fc_layers_with_hidden_sizes(actor_fc_layers, input_size=self.base_output_size),
            Categorical(actor_fc_layers[-1], num_actions),
        )
        self.critic = nn.Sequential(
            make_fc_layers_with_hidden_sizes(value_fc_layers, input_size=self.base_output_size),
            _init_(nn.Linear(value_fc_layers[-1], 1)),
        )

        apply_init_(self.modules())
        self.train()

    @property
    def is_recurrent(self):
        return self.rnn is not None

    @property
    def recurrent_hidden_state_size(self):
        return self.rnn.recurrent_hidden_state_size if self.rnn is not None else 0

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def _forward_base(self, inputs, rnn_hxs, masks):
        image = inputs.get('image')
        scalar = inputs.get('direction')
        if scalar is None:
            scalar = inputs.get('time_step')
        x = inputs.get('x')
        y = inputs.get('y')
        in_z = inputs.get('random_z', torch.tensor([], device=self.device))

        in_image = self.image_conv(image)

        if self.xy_embed:
            x_enc = one_hot(self.xy_dim, x, device=self.device)
            y_enc = one_hot(self.xy_dim, y, device=self.device)
            in_x = self.xy_embed(x_enc)
            in_y = self.xy_embed(y_enc)
        else:
            in_x = torch.tensor([], device=self.device)
            in_y = torch.tensor([], device=self.device)

        if self.scalar_embed:
            in_scalar = one_hot(self.scalar_dim, scalar).to(self.device)
            in_scalar = self.scalar_embed(in_scalar)
        else:
            in_scalar = torch.tensor([], device=self.device)

        in_embedded = torch.cat((in_image, in_x, in_y, in_scalar, in_z), dim=-1)

        if self.rnn is not None:
            core_features, rnn_hxs = self.rnn(in_embedded, rnn_hxs, masks)
        else:
            core_features = in_embedded

        return core_features, rnn_hxs

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        if self.random:
            B = inputs['image'].shape[0]
            action = torch.zeros((B, 1), dtype=torch.int64, device=self.device)
            values = torch.zeros((B, 1), device=self.device)
            action_log_dist = torch.ones(B, self.action_space.n, device=self.device)
            for b in range(B):
                action[b] = self.action_space.sample()
            return values, action, action_log_dist, rnn_hxs

        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)
        dist = self.actor(core_features)
        value = self.critic(core_features)

        action = dist.mode() if deterministic else dist.sample()
        action_log_dist = dist.logits
        return value, action, action_log_dist, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        core_features, _ = self._forward_base(inputs, rnn_hxs, masks)
        return self.critic(core_features)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, return_policy_logits=False):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)
        dist = self.actor(core_features)
        value = self.critic(core_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        if return_policy_logits:
            return value, action_log_probs, dist_entropy, rnn_hxs, dist
        return value, action_log_probs, dist_entropy, rnn_hxs
