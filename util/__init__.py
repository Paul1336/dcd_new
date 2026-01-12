# util/__init__.py
# This file marks the util directory as a Python package and re-exports key symbols for 'from util import ...' usage.

import numpy as np
import torch
from torchvision import utils as vutils

from .FileWriter import FileWriter
from .create_agent import create_agent
from .create_runner import create_runner
from .create_evaluator import create_evaluator
from .Parser import Parser

def save_images(images, path, normalize=True, channels_first=False):
    if path is None:
        return
    if isinstance(images, (list, tuple)):
        images = torch.tensor(np.stack(images), dtype=torch.float)
    elif isinstance(images, np.ndarray):
        images = torch.tensor(images, dtype=torch.float)

    if normalize:
        images = images/255

    if not channels_first:
        if len(images.shape) == 4:
            images = images.permute(0,3,1,2)
        else:
            images = images.permute(2,0,1)

    grid = vutils.make_grid(images, 4)
    vutils.save_image(grid, path)


def make_plr_args(args):
    return dict( 
        seeds=[], 
        obs_space=args.obs_space, 
        action_space=args.action_space, 
        num_actors=args.num_processes,
        strategy=args.level_replay_strategy,
        replay_schedule=args.level_replay_schedule,
        score_transform=args.level_replay_score_transform,
        temperature=args.level_replay_temperature,
        eps=args.level_replay_eps,
        rho=args.level_replay_rho,
        replay_prob=args.level_replay_prob, 
        alpha=args.level_replay_alpha,
        staleness_coef=args.staleness_coef,
        staleness_transform=args.staleness_transform,
        staleness_temperature=args.staleness_temperature,
        sample_full_distribution=args.train_full_distribution,
        seed_buffer_size=args.level_replay_seed_buffer_size,
        seed_buffer_priority=args.level_replay_seed_buffer_priority,
        use_dense_rewards=True if args.env_name.startswith('CarRacing') else False,
        gamma=args.gamma
    )

__all__ = [
	"save_images",
	"make_plr_args",
]
