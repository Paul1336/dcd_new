# Copyright (c) 2019 Antonin Raffin
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is an extended version of
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_normalize.py

from .vec_env import VecEnvWrapper
import numpy as np


class _RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        b_mean, b_var, b_count = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean  = self.mean + delta * b_count / total
        self.var   = (self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / total) / total
        self.count = total


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, use_tf=False):
        VecEnvWrapper.__init__(self, venv)
        if use_tf:
            raise RuntimeError("use_tf=True is not supported (TensorFlow dependency removed).")
        self.ob_rms  = _RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = _RunningMeanStd(shape=()) if ret else None
        self.clipob  = clipob
        self.cliprew = cliprew
        self.ret     = np.zeros(self.num_envs)
        self.gamma   = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)

    def reset_agent(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset_agent()
        return self._obfilt(obs)

    def reset_random(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset_random()
        return self._obfilt(obs)

    def reset_alp_gmm(self, level):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset_alp_gmm(level)
        return self._obfilt(obs)
