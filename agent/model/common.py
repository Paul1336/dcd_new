# Copyright (c) 2017 Ilya Kostrikov
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py

import numpy as np
import torch
import torch.nn as nn


class DeviceAwareModule(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device


class RNN(nn.Module):
    """GRU/LSTM backbone for recurrent actor-critic models."""

    def __init__(self, input_size, hidden_size=128, arch='lstm'):
        super().__init__()

        self.arch = arch
        self.is_lstm = arch == 'lstm'
        self._hidden_size = hidden_size

        if arch == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size)
        elif arch == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size)
        else:
            raise ValueError(f'Unsupported RNN architecture {arch}.')

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, x, hxs, masks):
        if self.is_lstm:
            hidden_batch_size = x.size(0) if hxs is None else hxs[0].size(0)
        else:
            hidden_batch_size = hxs.size(0)

        if x.size(0) == hidden_batch_size:
            masked_hxs = tuple((h * masks).unsqueeze(0) for h in hxs) if self.is_lstm \
                else (hxs * masks).unsqueeze(0)
            x, hxs = self.rnn(x.unsqueeze(0), masked_hxs)
            x = x.squeeze(0)
            hxs = tuple(h.squeeze(0) for h in hxs) if self.is_lstm else hxs.squeeze(0)
        else:
            # x is (T*N, -1); unflatten to (T, N, -1) for sequential RNN processing
            N = hxs[0].size(0) if self.is_lstm else hxs.size(0)
            T = int(x.size(0) / N)
            x = x.view(T, N, x.size(1))
            masks = masks.view(T, N)

            # Find timesteps where any env resets (mask == 0) so we can split the sequence
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()
            has_zeros = [0] + has_zeros + [T]

            hxs = (h.unsqueeze(0) for h in hxs) if self.is_lstm else hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                masked_hxs = tuple(h * masks[start_idx].view(1, -1, 1) for h in hxs) if self.is_lstm \
                    else hxs * masks[start_idx].view(1, -1, 1)
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], masked_hxs)
                outputs.append(rnn_scores)

            x = torch.cat(outputs, dim=0).view(T * N, -1)
            hxs = tuple(h.squeeze(0) for h in hxs) if self.is_lstm else hxs.squeeze(0)

        return x, hxs


class FixedCategorical(torch.distributions.Categorical):
    """Categorical distribution with shaped sample/log_probs for actor-critic use."""

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Categorical(nn.Module):
    """Linear head that produces a FixedCategorical distribution."""

    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = layer_init(nn.Linear(num_inputs, num_outputs), std=0.01)

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)
