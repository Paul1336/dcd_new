"""
CleanRL-style recurrent PPO (LSTM) for MiniGrid-SimpleCrossingS9N1-v0
- Flat obs + MLP encoder + LSTM + separate actor/critic heads
- Minibatch = group of envs processed as full sequences (required for BPTT)
- Should converge within 1-2M steps

Usage:
    python scripts/ppo_crossing_baseline.py
    python scripts/ppo_crossing_baseline.py --env-id MiniGrid-SimpleCrossingS11N5-v0
    python scripts/ppo_crossing_baseline.py --reward-shaping --shaping-coef 0.5
    python scripts/ppo_crossing_baseline.py --wandb_group my_group
"""

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass

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

    env_id: str = "MiniGrid-SimpleCrossingS9N1-v0"

    total_timesteps: int = 2_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128          # rollout length per env
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4      # envs are split into this many groups
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
    shaping_coef: float = 0.5    # weight on the potential-based bonus

    # Derived
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# ── Reward shaping wrapper ───────────────────────────────────────────────────

class RewardShapingWrapper(gym.Wrapper):
    """Potential-based reward shaping: F(s) = -manhattan_dist(agent, goal) / max_dist.

    Shaped reward = r + coef * (gamma * F(s') - F(s)).
    Does not change the optimal policy (potential-based shaping theorem).
    """

    def __init__(self, env, gamma: float = 0.99, coef: float = 0.5):
        super().__init__(env)
        self._gamma = gamma
        self._coef  = coef
        self._goal_pos   = None
        self._prev_pot   = 0.0

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
        max_dist = u.width + u.height
        return -dist / max_dist

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
        info['true_reward'] = float(reward)   # unshaped reward, for logging
        return obs, reward + self._coef * shaping, terminated, truncated, info


# ── Environment factory ──────────────────────────────────────────────────────

def make_env(env_id, seed, idx, capture_video, run_name, use_shaping=False, shaping_coef=0.5, gamma=0.99):
    def thunk():
        _ensure_official_minigrid_registered()
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        if use_shaping:
            env = RewardShapingWrapper(env, gamma=gamma, coef=shaping_coef)
        from minigrid.wrappers import FlatObsWrapper
        env = FlatObsWrapper(env)
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
    """MLP encoder → LSTM → separate actor/critic heads."""

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
        """Single-step LSTM forward. Resets hidden state where done=1."""
        h, c = lstm_state
        # done: (N,) → reset hidden state for finished episodes
        h = h * (1.0 - done).view(1, -1, 1)
        c = c * (1.0 - done).view(1, -1, 1)
        features = self.encoder(x)                          # (N, mlp_hidden)
        out, (h, c) = self.lstm(features.unsqueeze(0), (h, c))
        return out.squeeze(0), (h, c)                       # (N, lstm_hidden)

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


# ── Training loop ────────────────────────────────────────────────────────────

def train(args: Args):
    assert args.num_envs % args.num_minibatches == 0, \
        "num_envs must be divisible by num_minibatches"
    envs_per_batch = args.num_envs // args.num_minibatches

    args.batch_size     = args.num_envs * args.num_steps
    args.minibatch_size = envs_per_batch * args.num_steps
    args.num_iterations = args.total_timesteps // args.batch_size

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
                 use_shaping=args.reward_shaping, shaping_coef=args.shaping_coef, gamma=args.gamma)
        for i in range(args.num_envs)
    ])
    obs_dim   = int(np.prod(envs.single_observation_space.shape))
    n_actions = envs.single_action_space.n

    agent = Agent(obs_dim, n_actions, args.mlp_hidden, args.lstm_hidden).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Params: {total_params:,}  obs_dim={obs_dim}  actions={n_actions}  "
          f"lstm_hidden={args.lstm_hidden}")

    # Rollout buffers — shape (num_steps, num_envs, ...)
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

    # Manual episode tracking (more reliable than RecordEpisodeStatistics in vecenv)
    ep_ret_buf      = np.zeros(args.num_envs, dtype=np.float32)  # shaped return
    ep_true_ret_buf = np.zeros(args.num_envs, dtype=np.float32)  # unshaped return
    ep_len_buf      = np.zeros(args.num_envs, dtype=np.int32)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # Save LSTM state at start of rollout (needed for BPTT during update)
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

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_returns.append(info["episode"]["r"])
                        ep_lengths.append(info["episode"]["l"])
                        print(f"step={global_step:>8}  ep_return={info['episode']['r']:.3f}"
                              f"  ep_len={info['episode']['l']}")

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

        # ── PPO update (env-sequential minibatches for BPTT) ─────────────────
        # Each minibatch = envs_per_batch envs × num_steps steps (full sequences)
        clipfracs = []
        for epoch in range(args.update_epochs):
            env_order = torch.randperm(args.num_envs, device=device)
            for start in range(0, args.num_envs, envs_per_batch):
                mb_env_inds = env_order[start:start + envs_per_batch]

                # Initial LSTM state for this group of envs
                mb_lstm = (
                    initial_lstm[0][:, mb_env_inds, :],
                    initial_lstm[1][:, mb_env_inds, :],
                )

                # Run full sequence through LSTM to get updated logprobs / values
                mb_new_logprobs, mb_entropies, mb_new_values = [], [], []
                for t in range(args.num_steps):
                    obs_t    = obs_buf[t, mb_env_inds]
                    done_t   = dones[t, mb_env_inds]
                    action_t = actions[t, mb_env_inds].long()

                    _, new_logprob, entropy, new_value, mb_lstm = agent.get_action_and_value(
                        obs_t, mb_lstm, done_t, action_t
                    )
                    mb_new_logprobs.append(new_logprob)
                    mb_entropies.append(entropy)
                    mb_new_values.append(new_value.view(-1))

                # Flatten: (num_steps × envs_per_batch,)
                new_logprob  = torch.stack(mb_new_logprobs).reshape(-1)
                entropy_loss = torch.stack(mb_entropies).reshape(-1).mean()
                new_value    = torch.stack(mb_new_values).reshape(-1)

                old_logprob = logprobs[:, mb_env_inds].reshape(-1)
                mb_adv      = advantages[:, mb_env_inds].reshape(-1)
                mb_returns  = returns[:, mb_env_inds].reshape(-1)
                old_value   = values[:, mb_env_inds].reshape(-1)

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
            "charts/learning_rate":       optimizer.param_groups[0]["lr"],
            "charts/SPS":                 sps,
            "losses/value_loss":          vf_loss.item(),
            "losses/policy_loss":         pg_loss.item(),
            "losses/entropy":             entropy_loss.item(),
            "losses/approx_kl":           approx_kl.item(),
            "losses/clipfrac":            np.mean(clipfracs),
            "global_step":                global_step,
        }
        if ep_returns:
            log_dict["charts/mean_shaped_return"] = np.mean(ep_returns)
            log_dict["charts/mean_true_return"]   = np.mean(ep_true_returns)
            log_dict["charts/success_rate"]       = np.mean(ep_successes)
            log_dict["charts/mean_ep_length"]     = np.mean(ep_lengths)
        wandb.log(log_dict, step=global_step)

        if iteration % 10 == 0:
            print(f"iter={iteration}/{args.num_iterations}  steps={global_step}  SPS={sps}  "
                  f"vf={vf_loss.item():.4f}  pg={pg_loss.item():.4f}  ent={entropy_loss.item():.4f}")

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{run_name}.pt"
    torch.save(agent.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    envs.close()
    wandb.finish()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id",          type=str,   default="MiniGrid-SimpleCrossingS9N1-v0")
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
    parser.add_argument("--no-cuda",         action="store_true")
    parser.add_argument("--capture-video",   action="store_true")
    parser.add_argument("--exp_name",        type=str,   default="ppo_crossing_baseline")
    parser.add_argument("--wandb_group",     type=str,   default=None)
    cli = parser.parse_args()

    args = Args(
        env_id=cli.env_id,
        seed=cli.seed,
        total_timesteps=cli.total_timesteps,
        num_envs=cli.num_envs,
        num_steps=cli.num_steps,
        learning_rate=cli.learning_rate,
        ent_coef=cli.ent_coef,
        mlp_hidden=cli.mlp_hidden,
        lstm_hidden=cli.lstm_hidden,
        reward_shaping=cli.reward_shaping,
        shaping_coef=cli.shaping_coef,
        cuda=not cli.no_cuda,
        capture_video=cli.capture_video,
        exp_name=cli.exp_name,
        wandb_group=cli.wandb_group,
    )
    train(args)
