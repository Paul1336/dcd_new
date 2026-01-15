from collections import defaultdict
import os
import json
import random
from tqdm import tqdm
import numpy as np
import torch

from ..runner import Runner, AgentRole
from .learnability import LearnabilitySampler 
from ...interfaces import SampledLevelInfo , RunnerStats, RolloutResult, RunnerStateDict

class SFLRunner(Runner):

    def __init__(self, args, venv, agents, ued_venv=None, train=True):
        super().__init__(
            args=args,
            venv=venv,
            agents=agents,
            agent_type={AgentRole.AGENT},
            ued_venv=ued_venv,
        )

        self.agent = self.agents[AgentRole.AGENT]

        # mode
        if train:
            self.train()
            self.is_training = True
        else:
            self.eval()
            self.is_training = False

        self.reset()

        # --- learnability sampler ---
        self.learnability_sampler = LearnabilitySampler(
            venv=venv,
            learnability_alpha=args.learnability_alpha,
            learnability_c=args.learnability_c,
            top_k_to_sample_uniformly=args.top_k_to_sample_uniformly,
            staleness=args.learnability_staleness,
        )

        # --- bookkeeping ---
        self.env_sampling_total_count = defaultdict(int)
        self.env_sampling_current_count = defaultdict(int)

    # -------------------------
    # checkpointing
    # -------------------------
    def state_dict(self) -> RunnerStateDict:
        return {
            "runner": {
                "num_updates": self.num_updates,
                "total_episodes_collected": self.total_episodes_collected,
                "total_seeds_collected": self.total_seeds_collected,
            },
            "agents": {
                role.value: agent.state_dict()
                for role, agent in self.agents.items()
            },
        }

    def load_state_dict(self, state: dict):
        super().load_state_dict(state)

        sfl_state = state.get("sfl")
        if not sfl_state:
            return

        ls = sfl_state.get("learnability_sampler")
        if ls is not None:
            self.learnability_sampler.load_state_dict(ls)

        self.env_sampling_total_count = defaultdict(
            int, sfl_state.get("env_sampling_total_count", {})
        )
        self.env_sampling_current_count = defaultdict(
            int, sfl_state.get("env_sampling_current_count", {})
        )

    # -------------------------
    # learnability update
    # -------------------------
    def _update_learnability_metrics(self, global_step: int):
        """
        評估一批 env_id 的 success_rate，更新 learnability_sampler.task_info_dict
        並把 sampling_count / total_count dump 到 log_dir。
        """
        args = self.args
        os.makedirs(f"{args.log_dir}/learnability", exist_ok=True)

        # dump current learnability info + sampling_count
        task_info = dict(self.learnability_sampler.task_info_dict)
        for env_id, cnt in self.env_sampling_current_count.items():
            if env_id in task_info:
                task_info[env_id]["sampling_count"] = cnt
        self.env_sampling_current_count.clear()

        with open(f"{args.log_dir}/learnability/learnability_{global_step}.json", "w") as f:
            json.dump(task_info, f)

        with open(f"{args.log_dir}/learnability/env_sampling_total_count.json", "w") as f:
            json.dump(dict(self.env_sampling_total_count), f)

        # subsample envs to evaluate
        env_names = list(self.learnability_sampler.env_names)
        if len(env_names) == 0:
            return

        k = min(args.learnability_buffer_size, len(env_names))
        sampled_env_ids = random.sample(env_names, k)

        # 你原本的 code 用 evaluate_parallel_envs，我沿用
        from eval import evaluate_parallel_envs

        chunk_size = 40
        chunks = [sampled_env_ids[i:i + chunk_size] for i in range(0, len(sampled_env_ids), chunk_size)]

        for chunk in tqdm(chunks, desc="Updating learnability"):
            results = evaluate_parallel_envs(
                env_names=chunk,
                env_task_configs=[None] * len(chunk),
                agent=self.agent,
                num_episodes=10,
                device=self.device,
            )
            for env_id in chunk:
                self.learnability_sampler.update_learnability(
                    env_id=env_id,
                    global_step=global_step,
                    success_rate=results[env_id]["success_rate"],
                )

    # -------------------------
    # rollout
    # -------------------------
    def _agent_rollout(self, num_steps: int, update: bool = True)-> RolloutResult:
        args = self.args
        # 
        # self.agent.storage.copy_obs_to_index(obs,0)
        # rollout_info = {}
        # rollout_returns = [[] for _ in range(args.num_processes)]
        # sample levels via learnability
        sampled_levels = [self.learnability_sampler.sample() for _ in range(args.num_processes)]
        self.venv.reset_to_level_batch(sampled_levels)

        obs = self.venv.reset_agent()
        self.agent.storage.copy_obs_to_index(obs, 0)
        rollout_returns = [[] for _ in range(args.num_processes)]

        for step in range(num_steps):
            if args.render:
                self.venv.render_to_screen()
            with torch.no_grad():
                obs_id = self.agent.storage.get_obs(step)
                value, action, action_log_dist, rnn_state = self.agent.act(
                    obs_id,
                    self.agent.storage.get_recurrent_hidden_state(step),
                    self.agent.storage.masks[step],
                )
                if self.is_discrete_actions:
                    action_log_prob = action_log_dist.gather(-1, action)
                else:
                    action_log_prob = action_log_dist
                
            obs, reward, done, infos = self.venv.step_env(self.agent.process_action(action.cpu()))
            if args.clip_reward:
                reward = torch.clamp(reward, -args.clip_reward, args.clip_reward)

            for i, info in enumerate(infos):
                if "episode" in info:
                    r = info["episode"]["r"]
                    rollout_returns[i].append(r)
                    env_name = self.venv.remote_attr("level_seed", index=[i])[0][0]
                    self.env_sampling_total_count[env_name] += 1
                    self.env_sampling_current_count[env_name] += 1

                    self.total_episodes_collected += 1

                    print(
                        f" env_index={i:02d}, episodic_return={info['episode']['r']}, env={env_name}, current_count={self.env_sampling_current_count[env_name]}, total_count={self.env_sampling_total_count[env_name]}"
                    )

            masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])
            bad_masks = torch.ones_like(masks)
            cliffhanger_masks = torch.ones_like(masks)

            self.agent.insert(
                obs,
                rnn_state,
                action,
                action_log_prob,
                action_log_dist,
                value,
                reward,
                masks,
                bad_masks,
                level_seeds=None,
                cliffhanger_masks=cliffhanger_masks,
            )
        value_loss = action_loss = dist_entropy = None
        info = {}

        if update and self.is_training:
            with torch.no_grad():
             #   obs_id = self.agent.storage.get_obs(-1)
                last_obs_id = self.agent.storage.get_obs(-1)
                next_value = self.agent.get_value(
                    last_obs_id,
                    self.agent.storage.get_recurrent_hidden_state(-1),
                    self.agent.storage.masks[-1],
                ).detach()

            self.agent.storage.compute_returns(
                next_value, args.use_gae, args.gamma, args.gae_lambda)
            value_loss, action_loss, dist_entropy, info = self.agent.update()
            self.num_updates += 1
        result: RolloutResult = {
            "returns": rollout_returns,
            "value_loss": value_loss,
            "action_loss": action_loss,
            "dist_entropy": dist_entropy,
            "update_info": info,
        }
        return result

    # -------------------------
    # main loop
    # -------------------------
    def run(self, global_step: int, iteration: int, total_iterations: int) -> dict:
        args = self.args

        # update learnability periodically
        if self.is_training and (
            iteration == 1 or iteration % args.update_learnability_every_iterations == 0
        ):
            self._update_learnability_metrics(global_step)

        # LR annealing
        if self.is_training:
            frac = 1.0 - (iteration - 1.0) / total_iterations
            self.agent.update_lr(frac * args.lr)

        agent_info = self._agent_rollout(
            num_steps=args.num_steps,
            update=self.is_training,
        )
        for b in agent_info["returns"]:
            for r in reversed(b):
                self.agent_returns.append(r)

        mean_agent_return = (
            float(np.mean(self.agent_returns)) if len(self.agent_returns) > 0 else 0.0
        )
        sampled_level_info: SampledLevelInfo = {
            "source": "learnability",
            "env_ids": sampled_levels,   # 或你能拿到的 env_id list
            "level_replay": False,
            "num_edits": [0 for _ in range(args.num_processes)],
        }
        self._sampled_level_info = sampled_level_info

        stats: RunnerStats = {
            "steps": self.num_updates * args.num_processes * args.num_steps,
            "total_episodes": self.total_episodes_collected,
            "total_seeds": self.total_seeds_collected,
            "mean_agent_return": mean_agent_return,
            "agent_value_loss": agent_info["value_loss"],
            "agent_pg_loss": agent_info["action_loss"],
            "agent_dist_entropy": agent_info["dist_entropy"],
            "agent_lr": agent_info["update_info"].get("lr", None),
        }
        return stats