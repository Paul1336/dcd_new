from collections import defaultdict
import os
import json
import random
from tqdm import tqdm
import numpy as np
import torch

from ..runner import Runner, AgentRole
from .learnability import LearnabilitySampler 
from interfaces import SampledLevelInfo, RunnerStats, RolloutResult, RunnerStateDict

class SFLRunner(Runner):

    def __init__(self, args, venv, agents, ued_venv=None, train=True, obs_encoder=None):
        super().__init__(
            args=args,
            venv=venv,
            agents=agents,
            required_roles={AgentRole.AGENT},
            ued_venv=ued_venv,
        )

        self.agent = self.agents[AgentRole.AGENT]
        self.obs_encoder = obs_encoder

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
                "agent_returns": list(self.agent_returns),
            },
            "agents": {
                role.value: agent.state_dict()
                for role, agent in self.agents.items()
            },
            "sfl": {
                "learnability_sampler": self.learnability_sampler.state_dict(),
                "env_sampling_total_count": dict(self.env_sampling_total_count),
                "env_sampling_current_count": dict(self.env_sampling_current_count),
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

        use_image_obs = getattr(self.args, "obs_type", "symbolic") == "embedding"
        env_config = {"state_type": "image"} if use_image_obs else {}

        # PARAS is process-local in spawned subprocesses. Always fetch configs from
        # the main-process PARAS so each subprocess can register the task in its own
        # PARAS via IphyreGameEnv.__init__(env_task_config=...).
        from iphyre.simulator import PARAS as _PARAS

        for chunk in tqdm(chunks, desc="Updating learnability"):
            task_configs = [_PARAS.get(env_id) for env_id in chunk]

            results = evaluate_parallel_envs(
                env_names=chunk,
                env_task_configs=task_configs,
                agent=self.agent,
                num_episodes=10,
                device=self.device,
                env_config=env_config,
                obs_encoder=self.obs_encoder,
            )
            for env_id in chunk:
                self.learnability_sampler.update_learnability(
                    env_id=env_id,
                    global_step=global_step,
                    success_rate=results[env_id]["success_rate"],
                )

    def _get_obs_at_index(self, obs, i):
        if isinstance(obs, dict):
            return {k: v[i] for k, v in obs.items()}
        return obs[i]

    def _set_obs_at_index(self, obs, obs_, i):
        if isinstance(obs, dict):
            for k in obs.keys():
                x = obs_[k]
                if hasattr(x, "ndim") and x.ndim >= 1 and x.shape[0] == 1:
                    x = x[0]
                obs[k][i] = x
        else:
            x = obs_
            if hasattr(x, "ndim") and x.ndim >= 1 and x.shape[0] == 1:
                x = x[0]
            obs[i] = x

    # -------------------------
    # rollout
    # -------------------------
    def _agent_rollout(self, num_steps: int, update: bool = True)-> RolloutResult:
        args = self.args
        # 
        initial_levels = [self.learnability_sampler.sample() for _ in range(args.num_processes)]
        self.venv.reset_to_level_batch(initial_levels)
        self.total_seeds_collected += args.num_processes

        levels_history = [[lvl] for lvl in initial_levels]   # list[list[level_id]]

        obs = self.venv.reset_agent()
        self.agent.storage.set_obs(obs, 0)
        rollout_returns = [[] for _ in range(args.num_processes)]

        for step in range(num_steps):
            if getattr(args, 'render', False):
                self.venv.render_to_screen()
            with torch.no_grad():
                obs_id = self.agent.storage.get_obs(step)
                value, action, action_log_dist, rnn_state = self.agent.act(
                    obs_id,
                    self.agent.storage.get_recurrent_hidden_state(step),
                    self.agent.storage.masks[step],
                )
                if self.venv.action_space_is_discrete:
                    action_log_prob = action_log_dist.gather(-1, action)
                else:
                    action_log_prob = action_log_dist
                
            obs, reward, done, infos = self.venv.step_env(self.agent.process_action(action.cpu()))
            if args.clip_reward:
                reward = torch.clamp(reward, -args.clip_reward, args.clip_reward)

            # Cliffhanger: force done=True at the last rollout step for all envs.
            # Non-done envs are mid-episode cuts; mark them truncated+cliffhanger
            # so the learnability sampler and GAE handle them correctly.
            if step == num_steps - 1:
                if self.agent.storage.use_proper_time_limits:
                    for i, done_ in enumerate(done):
                        if not done_:
                            infos[i]['cliffhanger'] = True
                            infos[i]['truncated'] = True
                            infos[i]['truncated_obs'] = self._get_obs_at_index(obs, i)
                done = [True] * len(done)

            for i, info in enumerate(infos):
                if "episode" in info:
                    rollout_returns[i].append(info["episode"]["r"])
                    env_name = self.venv.remote_attr("level_seed", index=[i])[0][0]

                    self.env_sampling_total_count[env_name] += 1
                    self.env_sampling_current_count[env_name] += 1
                    self.total_episodes_collected += 1

                    print(
                        f" env_index={i:02d}, episodic_return={info['episode']['r']}, env={env_name}, current_count={self.env_sampling_current_count[env_name]}, total_count={self.env_sampling_total_count[env_name]}"
                    )

                    if self.agent.storage.use_proper_time_limits:
                        if 'truncated_obs' in info:
                            self.agent.storage.insert_truncated_obs(info['truncated_obs'], index=i)

                    new_level = self.learnability_sampler.sample()
                    obs_i = self.venv.reset_to_level(new_level, i)
                    self._set_obs_at_index(obs, obs_i, i)
                    self.total_seeds_collected += 1

                    levels_history[i].append(new_level)

            masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])
            bad_masks = torch.FloatTensor([[0.0] if 'truncated' in info else [1.0] for info in infos])
            cliffhanger_masks = torch.FloatTensor([[0.0] if 'cliffhanger' in info else [1.0] for info in infos])
            self.agent.insert(
                obs,
                rnn_state,
                action,
                action_log_prob,
                action_log_dist,
                value,
                reward,
                masks,
                bad_masks=bad_masks,
                cliffhanger_masks=cliffhanger_masks,
                level_seeds=None,
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
            "sampled_levels": levels_history,
        }
        return result

    # -------------------------
    # main loop
    # -------------------------
    def run(self, global_step: int, iteration: int, total_iterations: int) -> RunnerStats:

        # update learnability periodically
        if self.is_training and (
            iteration == 1 or iteration % self.args.update_learnability_every_iterations == 0
        ):
            self._update_learnability_metrics(global_step)

        # LR annealing
        if self.is_training:
            frac = 1.0 - (iteration - 1.0) / total_iterations
            self.agent.update_lr(frac * self.args.lr)

        agent_info = self._agent_rollout(
            num_steps=self.args.num_steps,
            update=self.is_training,
        )
        for b in agent_info["returns"]:
            for r in reversed(b):
                self.agent_returns.append(r)
        mean_agent_return = (
            float(np.mean(self.agent_returns)) if len(self.agent_returns) > 0 else 0.0
        )
        # if self.is_training:
        #     self.num_updates += 1
        self._sampled_level_info: SampledLevelInfo = {
            "source": "learnability",
            "env_ids": [h[-1] for h in agent_info["sampled_levels"]] if agent_info["sampled_levels"] else [],
            "level_replay": False,
            "num_edits": [0] * self.args.num_processes,
        }

        stats = RunnerStats(
            steps=global_step,
            global_step=global_step,
            total_episodes=self.total_episodes_collected,
            total_seeds=self.total_seeds_collected,
            mean_agent_return=mean_agent_return,
            agent_value_loss=agent_info["value_loss"],
            agent_pg_loss=agent_info["action_loss"],
            agent_dist_entropy=agent_info["dist_entropy"],
            agent_lr=agent_info["update_info"].get("lr", None),
        )
        return stats