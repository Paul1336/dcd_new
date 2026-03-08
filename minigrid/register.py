# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Register MultiGrid environments with OpenAI gym."""

import gym
from env.registration import register as gym_register

env_list = []


def register(env_id, entry_point, reward_threshold=0.95, max_episode_steps=None):
    """Register a new environment with OpenAI gym based on id."""
    assert env_id.startswith("MultiGrid-")
    if env_id in env_list:
        del gym.envs.registry.env_specs[env_id]
    else:
        env_list.append(env_id)

    kwargs = dict(
        id=env_id,
        entry_point=entry_point,
        reward_threshold=reward_threshold,
    )

    if max_episode_steps:
        kwargs['max_episode_steps'] = max_episode_steps

    gym_register(**kwargs)
