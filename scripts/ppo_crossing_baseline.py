"""
CleanRL-style recurrent PPO (LSTM) for MiniGrid-SimpleCrossingS11N5-v0
- image+direction obs (5×5×3 + 4 = 79-dim) compatible with eval suites
- MLP encoder → LSTM → actor/critic heads
- Minibatch = group of envs × full sequences (BPTT)
- Periodic eval on the same test suites as the main codebase

Usage:
    python scripts/ppo_crossing_baseline.py
    python scripts/ppo_crossing_baseline.py --env-id MiniGrid-SimpleCrossingS11N5-v0
    python scripts/ppo_crossing_baseline.py --reward-shaping --shaping-coef 0.5
    python scripts/ppo_crossing_baseline.py --wandb_group my_group --eval-interval 100
"""

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List

# Ensure project root is on path so `env.*` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.distributions import Categorical

from env.benchmark.minigrid.official_wrapper import _ensure_official_minigrid_registered


# ── Hyperparameters ─────────────────────────────────────────────────────────

@dataclass
class Args:
    exp_name: str = "ppo_crossing_baseline"
    seed: int = 1
    cuda: bool = True
    wandb_group: str = None
    capture_video: bool = False

    env_id: str = "MiniGrid-SimpleCrossingS11N5-v0"

    total_timesteps: int = 2_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    mlp_hidden: int = 64
    lstm_hidden: int = 128

    reward_shaping: bool = False
    shaping_coef: float = 0.5

    # Eval
    eval_interval: int = 20        # iterations between eval runs (0 = no eval)
    test_num_tasks: int = 10        # levels per suite
    test_env_names: str = (
        'MultiGrid-VLMSampled-v0,'
        'MultiGrid-RandomGenerated-v0,'
        'MultiGrid-FourRooms-v0,'
        'MultiGrid-SimpleCrossing-v0'
    )

    # Derived
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# ── Obs helpers ──────────────────────────────────────────────────────────────

def flat_obs(obs_dict: dict) -> np.ndarray:
    """Convert {image:(H,W,3), direction:scalar} → float32 vector (H*W*3 + 4,)."""
    img = obs_dict['image'].flatten().astype(np.float32)
    d = obs_dict['direction']
    if hasattr(d, 'flat'):
        d = int(d.flat[0])
    elif isinstance(d, (list, tuple)):
        d = int(d[0])
    else:
        d = int(d)
    dir_oh = np.zeros(4, dtype=np.float32)
    dir_oh[d] = 1.0
    return np.concatenate([img, dir_oh])


class MinigridFlatWrapper(gym.Wrapper):
    """Flatten {image, direction} dict obs → (H*W*3 + 4,) float32 vector.

    Replaces FlatObsWrapper: no mission-string encoding, so obs_dim stays
    small (75 + 4 = 79 for agent_view_size=5) and matches suite env obs.
    """

    def __init__(self, env):
        super().__init__(env)
        img_shape = env.observation_space['image'].shape   # (H, W, 3)
        obs_dim = int(np.prod(img_shape)) + 4
        self.observation_space = gym.spaces.Box(
            low=0.0, high=255.0, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return flat_obs(obs_dict), info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        return flat_obs(obs_dict), reward, terminated, truncated, info


# ── Reward shaping wrapper ───────────────────────────────────────────────────

class RewardShapingWrapper(gym.Wrapper):
    """Potential-based shaping: F(s) = -manhattan_dist(agent, goal) / (W+H).
    shaped_reward = r + coef * (gamma * F(s') - F(s))
    """

    def __init__(self, env, gamma: float = 0.99, coef: float = 0.5):
        super().__init__(env)
        self._gamma = gamma
        self._coef  = coef
        self._goal_pos = None
        self._prev_pot = 0.0

    def _find_goal(self):
        u = self.unwrapped
        for i in range(u.width):
            for j in range(u.height):
                cell = u.grid.get(i, j)
                if cell is not None and cell.type == 'goal':
                    return (i, j)
        return None

    def _potential(self):
        if self._goal_pos is None:
            return 0.0
        u = self.unwrapped
        ax, ay = u.agent_pos
        gx, gy = self._goal_pos
        dist = abs(ax - gx) + abs(ay - gy)
        return -dist / (u.width + u.height)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._goal_pos = self._find_goal()
        self._prev_pot = self._potential()
        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_pot        = self._potential()
        shaping        = self._gamma * new_pot - self._prev_pot
        self._prev_pot = new_pot
        info['true_reward'] = float(reward)
        return obs, reward + self._coef * shaping, terminated, truncated, info


# ── Environment factory ──────────────────────────────────────────────────────

def make_env(env_id, seed, idx, capture_video, run_name,
             use_shaping=False, shaping_coef=0.5, gamma=0.99):
    def thunk():
        _ensure_official_minigrid_registered()
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", agent_view_size=5)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, agent_view_size=5)
        if use_shaping:
            env = RewardShapingWrapper(env, gamma=gamma, coef=shaping_coef)
        env = MinigridFlatWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env
    return thunk


# ── Network ──────────────────────────────────────────────────────────────────

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """MLP encoder → LSTM → actor/critic heads."""

    def __init__(self, obs_dim, n_actions, mlp_hidden=64, lstm_hidden=128):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, mlp_hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(mlp_hidden, mlp_hidden)),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(mlp_hidden, lstm_hidden)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.actor  = layer_init(nn.Linear(lstm_hidden, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(lstm_hidden, 1),          std=1.0)

    def _lstm_step(self, x, lstm_state, done):
        h, c = lstm_state
        h = h * (1.0 - done).view(1, -1, 1)
        c = c * (1.0 - done).view(1, -1, 1)
        features = self.encoder(x)
        out, (h, c) = self.lstm(features.unsqueeze(0), (h, c))
        return out.squeeze(0), (h, c)

    def get_value(self, x, lstm_state, done):
        features, _ = self._lstm_step(x, lstm_state, done)
        return self.critic(features)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        features, new_lstm_state = self._lstm_step(x, lstm_state, done)
        logits = self.actor(features)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(features), new_lstm_state


# ── Eval on test suites ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_suites(agent, suite_names: List[str], num_tasks: int,
                    device: torch.device, lstm_hidden: int) -> dict:
    from eval.suites import load_minigrid_test_suite

    agent.eval()
    all_results = {}

    for suite_name in suite_names:
        try:
            env_names, env_fns = load_minigrid_test_suite(suite_name, num_tasks=num_tasks)
        except Exception as e:
            print(f"[Eval] Failed to load {suite_name}: {e}")
            continue

        successes, ep_returns = [], []
        for env_fn in env_fns:
            env = env_fn()
            obs_raw = env.reset()
            # suite envs return plain obs (not (obs, info) tuple)
            if isinstance(obs_raw, tuple):
                obs_raw = obs_raw[0]
            obs = torch.tensor(flat_obs(obs_raw), dtype=torch.float32, device=device).unsqueeze(0)
            lstm_state = (
                torch.zeros(1, 1, lstm_hidden, device=device),
                torch.zeros(1, 1, lstm_hidden, device=device),
            )
            done = torch.zeros(1, device=device)
            ep_ret = 0.0
            success = False

            for _ in range(512):
                action, _, _, _, lstm_state = agent.get_action_and_value(obs, lstm_state, done)
                step_result = env.step(action.item())
                if len(step_result) == 4:
                    obs_raw, reward, d, info = step_result
                    ep_ret += reward
                    if d:
                        success = reward > 0
                        break
                else:
                    obs_raw, reward, term, trunc, info = step_result
                    ep_ret += reward
                    if term or trunc:
                        success = bool(term)
                        break
                obs = torch.tensor(flat_obs(obs_raw), dtype=torch.float32, device=device).unsqueeze(0)
                done = torch.zeros(1, device=device)

            successes.append(float(success))
            ep_returns.append(ep_ret)
            env.close()

        sr   = float(np.mean(successes))
        mean_r = float(np.mean(ep_returns))
        all_results[suite_name] = {'success_rate': sr, 'mean_return': mean_r}
        print(f"[Eval] {suite_name:40s}  success={sr:.3f}  return={mean_r:.3f}")

    agent.train()
    return all_results


# ── Training loop ────────────────────────────────────────────────────────────

def train(args: Args):
    assert args.num_envs % args.num_minibatches == 0
    envs_per_batch = args.num_envs // args.num_minibatches

    args.batch_size     = args.num_envs * args.num_steps
    args.minibatch_size = envs_per_batch * args.num_steps
    args.num_iterations = args.total_timesteps // args.batch_size

    suite_names = [s.strip() for s in args.test_env_names.split(',') if s.strip()]

    run_name = f"{args.env_id}__{args.exp_name}__seed{args.seed}__{int(time.time())}"

    wandb.init(
        project="dcd_new",
        config=vars(args),
        name=run_name,
        group=args.wandb_group,
        save_code=True,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Env: {args.env_id} | Seed: {args.seed}")

    envs = gym.vector.SyncVectorEnv([
        make_env(args.env_id, args.seed, i, args.capture_video, run_name,
                 use_shaping=args.reward_shaping, shaping_coef=args.shaping_coef,
                 gamma=args.gamma)
        for i in range(args.num_envs)
    ])
    obs_dim   = int(np.prod(envs.single_observation_space.shape))
    n_actions = envs.single_action_space.n

    agent = Agent(obs_dim, n_actions, args.mlp_hidden, args.lstm_hidden).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Params: {total_params:,}  obs_dim={obs_dim}  actions={n_actions}  lstm={args.lstm_hidden}")

    obs_buf  = torch.zeros((args.num_steps, args.num_envs, obs_dim),  device=device)
    actions  = torch.zeros((args.num_steps, args.num_envs),           device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs),           device=device)
    rewards  = torch.zeros((args.num_steps, args.num_envs),           device=device)
    dones    = torch.zeros((args.num_steps, args.num_envs),           device=device)
    values   = torch.zeros((args.num_steps, args.num_envs),           device=device)

    global_step = 0
    start_time  = time.time()

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs    = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done   = torch.zeros(args.num_envs, device=device)
    next_lstm   = (
        torch.zeros(1, args.num_envs, args.lstm_hidden, device=device),
        torch.zeros(1, args.num_envs, args.lstm_hidden, device=device),
    )

    ep_ret_buf      = np.zeros(args.num_envs, dtype=np.float32)
    ep_true_ret_buf = np.zeros(args.num_envs, dtype=np.float32)
    ep_len_buf      = np.zeros(args.num_envs, dtype=np.int32)

    # Zero-shot eval before training
    if args.eval_interval > 0 and suite_names:
        print("\n[Eval] Zero-shot evaluation...")
        eval_results = evaluate_suites(agent, suite_names, args.test_num_tasks,
                                       device, args.lstm_hidden)
        eval_log = {}
        for suite, metrics in eval_results.items():
            short = suite.replace('MultiGrid-', '').replace('-v0', '')
            eval_log[f"eval/{short}/success_rate"] = metrics['success_rate']
            eval_log[f"eval/{short}/mean_return"]  = metrics['mean_return']
        wandb.log(eval_log, step=0)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        initial_lstm = (next_lstm[0].clone(), next_lstm[1].clone())

        # ── Collect rollout ──────────────────────────────────────────────────
        ep_returns, ep_true_returns, ep_lengths, ep_successes = [], [], [], []
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            dones[step]   = next_done

            with torch.no_grad():
                action, logprob, _, value, next_lstm = agent.get_action_and_value(
                    next_obs, next_lstm, next_done
                )
            actions[step]  = action
            logprobs[step] = logprob
            values[step]   = value.flatten()

            next_obs_np, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = torch.tensor(
                np.logical_or(terminations, truncations), dtype=torch.float32, device=device
            )
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            next_obs      = torch.tensor(next_obs_np, dtype=torch.float32, device=device)

            true_reward = infos.get('true_reward', reward) if args.reward_shaping else reward
            ep_ret_buf      += reward
            ep_true_ret_buf += true_reward
            ep_len_buf      += 1
            for i, done in enumerate(np.logical_or(terminations, truncations)):
                if done:
                    success = bool(terminations[i])
                    ep_returns.append(float(ep_ret_buf[i]))
                    ep_true_returns.append(float(ep_true_ret_buf[i]))
                    ep_lengths.append(int(ep_len_buf[i]))
                    ep_successes.append(float(success))
                    print(f"step={global_step:>8}  true_return={ep_true_ret_buf[i]:.3f}"
                          f"  shaped_return={ep_ret_buf[i]:.3f}  ep_len={ep_len_buf[i]}"
                          f"  {'SUCCESS' if success else 'timeout'}")
                    ep_ret_buf[i]      = 0.0
                    ep_true_ret_buf[i] = 0.0
                    ep_len_buf[i]      = 0

        # ── GAE ─────────────────────────────────────────────────────────────
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_lstm, next_done).reshape(1, -1)
            advantages  = torch.zeros_like(rewards, device=device)
            lastgaelam  = 0
            for t in reversed(range(args.num_steps)):
                nextnonterminal = 1.0 - (next_done if t == args.num_steps - 1 else dones[t + 1])
                nextvalues      = next_value if t == args.num_steps - 1 else values[t + 1]
                delta           = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t]   = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # ── PPO update ───────────────────────────────────────────────────────
        clipfracs = []
        for epoch in range(args.update_epochs):
            env_order = torch.randperm(args.num_envs, device=device)
            for start in range(0, args.num_envs, envs_per_batch):
                mb_env_inds = env_order[start:start + envs_per_batch]
                mb_lstm = (
                    initial_lstm[0][:, mb_env_inds, :],
                    initial_lstm[1][:, mb_env_inds, :],
                )
                mb_new_logprobs, mb_entropies, mb_new_values = [], [], []
                for t in range(args.num_steps):
                    _, new_logprob, entropy, new_value, mb_lstm = agent.get_action_and_value(
                        obs_buf[t, mb_env_inds], mb_lstm, dones[t, mb_env_inds],
                        actions[t, mb_env_inds].long()
                    )
                    mb_new_logprobs.append(new_logprob)
                    mb_entropies.append(entropy)
                    mb_new_values.append(new_value.view(-1))

                new_logprob  = torch.stack(mb_new_logprobs).reshape(-1)
                entropy_loss = torch.stack(mb_entropies).reshape(-1).mean()
                new_value    = torch.stack(mb_new_values).reshape(-1)
                old_logprob  = logprobs[:, mb_env_inds].reshape(-1)
                mb_adv       = advantages[:, mb_env_inds].reshape(-1)
                mb_returns   = returns[:, mb_env_inds].reshape(-1)
                old_value    = values[:, mb_env_inds].reshape(-1)

                logratio = new_logprob - old_logprob
                ratio    = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                ).mean()

                if args.clip_vloss:
                    v_clipped = old_value + torch.clamp(new_value - old_value, -args.clip_coef, args.clip_coef)
                    vf_loss = torch.max(
                        (new_value - mb_returns).pow(2),
                        (v_clipped  - mb_returns).pow(2)
                    ).mean() / 2
                else:
                    vf_loss = (new_value - mb_returns).pow(2).mean() / 2

                loss = pg_loss - args.ent_coef * entropy_loss + vf_loss * args.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ── Logging ─────────────────────────────────────────────────────────
        sps = int(global_step / (time.time() - start_time))
        log_dict = {
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "charts/SPS":           sps,
            "losses/value_loss":    vf_loss.item(),
            "losses/policy_loss":   pg_loss.item(),
            "losses/entropy":       entropy_loss.item(),
            "losses/approx_kl":     approx_kl.item(),
            "losses/clipfrac":      np.mean(clipfracs),
            "global_step":          global_step,
        }
        if ep_returns:
            log_dict["charts/mean_shaped_return"] = np.mean(ep_returns)
            log_dict["charts/mean_true_return"]   = np.mean(ep_true_returns)
            log_dict["charts/success_rate"]       = np.mean(ep_successes)
            log_dict["charts/mean_ep_length"]     = np.mean(ep_lengths)

        # ── Periodic eval ────────────────────────────────────────────────────
        if args.eval_interval > 0 and suite_names and iteration % args.eval_interval == 0:
            print(f"\n[Eval] iter={iteration}  step={global_step}")
            eval_results = evaluate_suites(agent, suite_names, args.test_num_tasks,
                                           device, args.lstm_hidden)
            for suite, metrics in eval_results.items():
                short = suite.replace('MultiGrid-', '').replace('-v0', '')
                log_dict[f"eval/{short}/success_rate"] = metrics['success_rate']
                log_dict[f"eval/{short}/mean_return"]  = metrics['mean_return']

        wandb.log(log_dict, step=global_step)

        if iteration % 10 == 0:
            print(f"iter={iteration}/{args.num_iterations}  steps={global_step}  SPS={sps}  "
                  f"vf={vf_loss.item():.4f}  pg={pg_loss.item():.4f}  ent={entropy_loss.item():.4f}")

    # Final eval
    if args.eval_interval > 0 and suite_names:
        print("\n[Eval] Final evaluation...")
        eval_results = evaluate_suites(agent, suite_names, args.test_num_tasks,
                                       device, args.lstm_hidden)
        final_log = {"global_step": global_step}
        for suite, metrics in eval_results.items():
            short = suite.replace('MultiGrid-', '').replace('-v0', '')
            final_log[f"eval/{short}/success_rate"] = metrics['success_rate']
            final_log[f"eval/{short}/mean_return"]  = metrics['mean_return']
        wandb.log(final_log, step=global_step)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{run_name}.pt"
    torch.save(agent.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    envs.close()
    wandb.finish()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id",          type=str,   default="MiniGrid-SimpleCrossingS11N5-v0")
    parser.add_argument("--seed",            type=int,   default=1)
    parser.add_argument("--total-timesteps", type=int,   default=2_000_000)
    parser.add_argument("--num-envs",        type=int,   default=8)
    parser.add_argument("--num-steps",       type=int,   default=128)
    parser.add_argument("--learning-rate",   type=float, default=2.5e-4)
    parser.add_argument("--ent-coef",        type=float, default=0.01)
    parser.add_argument("--mlp-hidden",      type=int,   default=64)
    parser.add_argument("--lstm-hidden",     type=int,   default=128)
    parser.add_argument("--reward-shaping",  action="store_true")
    parser.add_argument("--shaping-coef",    type=float, default=0.5)
    parser.add_argument("--eval-interval",   type=int,   default=100)
    parser.add_argument("--test-num-tasks",  type=int,   default=10)
    parser.add_argument("--test-env-names",  type=str,
                        default='MultiGrid-VLMSampled-v0,MultiGrid-RandomGenerated-v0,'
                                'MultiGrid-SimpleCrossing-v0')
    parser.add_argument("--no-cuda",         action="store_true")
    parser.add_argument("--capture-video",   action="store_true")
    parser.add_argument("--exp_name",        type=str,   default="ppo_crossing_baseline")
    parser.add_argument("--wandb_group",     type=str,   default=None)
    parser.add_argument("--num-trials",      type=int,   default=1,
                        help="Number of parallel trials with consecutive seeds.")
    parser.add_argument("--base-seed",       type=int,   default=None,
                        help="Override base seed for multi-trial runs (default: --seed value).")
    cli = parser.parse_args()

    num_trials = cli.num_trials
    base_seed  = cli.base_seed if cli.base_seed is not None else cli.seed

    if num_trials == 1:
        args = Args(
            env_id=cli.env_id,
            seed=base_seed,
            total_timesteps=cli.total_timesteps,
            num_envs=cli.num_envs,
            num_steps=cli.num_steps,
            learning_rate=cli.learning_rate,
            ent_coef=cli.ent_coef,
            mlp_hidden=cli.mlp_hidden,
            lstm_hidden=cli.lstm_hidden,
            reward_shaping=cli.reward_shaping,
            shaping_coef=cli.shaping_coef,
            eval_interval=cli.eval_interval,
            test_num_tasks=cli.test_num_tasks,
            test_env_names=cli.test_env_names,
            cuda=not cli.no_cuda,
            capture_video=cli.capture_video,
            exp_name=cli.exp_name,
            wandb_group=cli.wandb_group,
        )
        train(args)
    else:
        # Build one subprocess command per trial, forwarding all flags except
        # --num-trials / --base-seed, and injecting --seed per trial.
        import subprocess
        base_cmd = [sys.executable, __file__]
        skip = {'--num-trials', '--base-seed', '--seed'}
        argv = sys.argv[1:]
        filtered = []
        i = 0
        while i < len(argv):
            tok = argv[i]
            key = tok.split('=')[0]
            if key in skip:
                # skip value token too if flag and value are separate
                if '=' not in tok and i + 1 < len(argv) and not argv[i + 1].startswith('--'):
                    i += 2
                else:
                    i += 1
            else:
                filtered.append(tok)
                i += 1

        procs = []
        for trial in range(num_trials):
            seed = base_seed + trial
            cmd  = base_cmd + filtered + [f'--seed={seed}']
            print(f'[Trial {trial}] seed={seed}  ' + ' '.join(cmd))
            procs.append(subprocess.Popen(cmd, text=True))

        try:
            for trial, p in enumerate(procs):
                ret = p.wait()
                if ret != 0:
                    print(f'[Trial {trial}] failed with exit code {ret}')
                else:
                    print(f'[Trial {trial}] completed successfully')
        except KeyboardInterrupt:
            print('\nStopping all trials...')
            for p in procs:
                p.terminate()
            sys.exit(1)
