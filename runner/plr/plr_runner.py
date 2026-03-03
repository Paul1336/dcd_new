import random
from collections import defaultdict

import numpy as np
import torch

from ..runner import Runner, AgentRole
from ..accel.level_sampler import LevelSampler
from interfaces import SampledLevelInfo, RunnerStats, RunnerStateDict
from util import make_plr_args


_VLM_ENV_NAMES = frozenset({
    'Iphyre-AdversarialVLM4k-v0',
    'Iphyre-AdversarialVLM10k-v0',
    'Iphyre-AdversarialClaudeVLM10k-v0',
    'Iphyre-AdversarialGeminiVLM10k-v0',
})


class PLRRunner(Runner):
    """
    PLR (Prioritized Level Replay) runner.

    Curriculum is maintained by a LevelSampler that scores levels by
    value-loss (or another strategy) and replays high-scoring levels.
    Unlike ACCEL, no level editing or mutation is performed — PLR only
    decides which levels to replay based on learnability scores.

    Level IDs in the iphyre env are strings (VLM task names).
    LevelSampler requires integer seeds, so a bijective mapping is
    maintained:  int_seed  <->  str_level_id.
    """

    def __init__(self, args, venv, agents, ued_venv=None, train=True):
        super().__init__(
            args=args,
            venv=venv,
            agents=agents,
            required_roles={AgentRole.AGENT},
            ued_venv=ued_venv,
        )

        self.agent = self.agents[AgentRole.AGENT]

        self._vlm_mode = args.env_name in _VLM_ENV_NAMES

        # ---- int seed <-> str level_id mapping ----
        if self._vlm_mode:
            env_names = venv.remote_attr('subsampled_env_ids', index=[0])[0][0]
            self._vlm_env_names = list(env_names)
            self.seed2level_id: dict = {i: env_id for i, env_id in enumerate(env_names)}
            self.level_id2seed: dict = {env_id: i for i, env_id in enumerate(env_names)}
        else:
            # Procedural: dynamic mapping grows during training
            self._vlm_env_names = []
            self.seed2level_id: dict = {}
            self.level_id2seed: dict = {}
        self._next_proc_seed: int = len(self.seed2level_id)

        # ---- PLR level sampler ----
        plr_kwargs = make_plr_args(args, venv.observation_space, venv.action_space)
        self.level_sampler = LevelSampler(**plr_kwargs)

        # current active int seed per env (updated during rollout)
        self.current_int_seeds = [-1] * args.num_processes

        # env sampling counters
        self.env_sampling_total_count = defaultdict(int)

        if train:
            self.train()
            self.is_training = True
        else:
            self.eval()
            self.is_training = False

        self.reset()

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------
    def state_dict(self) -> RunnerStateDict:
        return {
            "runner": {
                "num_updates":              self.num_updates,
                "total_episodes_collected": self.total_episodes_collected,
                "total_seeds_collected":    self.total_seeds_collected,
                "agent_returns":            list(self.agent_returns),
            },
            "agents": {
                role.value: agent.state_dict()
                for role, agent in self.agents.items()
            },
            "plr": {
                "seed2level_id":            {str(k): v for k, v in self.seed2level_id.items()},
                "level_id2seed":            dict(self.level_id2seed),
                "_next_proc_seed":          self._next_proc_seed,
                "env_sampling_total_count": dict(self.env_sampling_total_count),
            },
        }

    def load_state_dict(self, state: dict):
        super().load_state_dict(state)

        plr_state = state.get("plr")
        if not plr_state:
            return

        self.seed2level_id = {int(k): v for k, v in plr_state.get("seed2level_id", {}).items()}
        self.level_id2seed = dict(plr_state.get("level_id2seed", {}))
        self._next_proc_seed = plr_state.get("_next_proc_seed", len(self.seed2level_id))
        self.env_sampling_total_count = defaultdict(int, plr_state.get("env_sampling_total_count", {}))

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
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

    def _register_proc_level(self, str_level: str) -> int:
        """Register a procedural level encoding and return its int seed."""
        if str_level in self.level_id2seed:
            return self.level_id2seed[str_level]
        int_seed = self._next_proc_seed
        self._next_proc_seed += 1
        self.seed2level_id[int_seed] = str_level
        self.level_id2seed[str_level] = int_seed
        return int_seed

    def _sample_new_level_int_seed(self) -> int:
        """Randomly pick a level int seed from the known pool."""
        if self._vlm_mode:
            env_id = random.choice(self._vlm_env_names)
            return self.level_id2seed[env_id]
        else:
            return random.choice(list(self.seed2level_id.keys()))

    # -----------------------------------------------------------------------
    # Main rollout loop
    # -----------------------------------------------------------------------
    def _agent_rollout(self, num_steps: int, update: bool = True) -> dict:
        """
        Collect num_steps of experience and optionally update the agent.

        Returns a dict with keys:
            returns, value_loss, action_loss, dist_entropy, update_info,
            level_replay, initial_int_seeds
        """
        args = self.args
        N = args.num_processes

        # --- decide: replay or new level ---
        level_replay = self.level_sampler.sample_replay_decision()

        if level_replay:
            int_seeds = [self.level_sampler.sample_replay_level() for _ in range(N)]
            level_ids = [self.seed2level_id[s] for s in int_seeds]
            self.venv.reset_to_level_batch(level_ids)
            obs = self.venv.reset_agent()
        elif self._vlm_mode:
            int_seeds = [self._sample_new_level_int_seed() for _ in range(N)]
            self.level_sampler.observe_external_unseen_sample(int_seeds)
            level_ids = [self.seed2level_id[s] for s in int_seeds]
            self.venv.reset_to_level_batch(level_ids)
            obs = self.venv.reset_agent()
        else:
            # Procedural: discover N new levels via random reset
            obs = self.venv.reset_random()
            encodings = [self.venv.remote_attr('encoding', index=[i])[0][0] for i in range(N)]
            int_seeds = [self._register_proc_level(enc) for enc in encodings]
            self.level_sampler.observe_external_unseen_sample(int_seeds)

        self.current_int_seeds = list(int_seeds)
        self.total_seeds_collected += N
        self.agent.storage.set_obs(obs, 0)
        rollout_returns = [[] for _ in range(N)]

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

            # cliffhanger at last step
            if step == num_steps - 1:
                if self.agent.storage.use_proper_time_limits:
                    for i, done_ in enumerate(done):
                        if not done_:
                            infos[i]['cliffhanger'] = True
                            infos[i]['truncated']   = True
                            infos[i]['truncated_obs'] = self._get_obs_at_index(obs, i)
                done = [True] * N

            for i, info in enumerate(infos):
                if 'episode' in info:
                    rollout_returns[i].append(info['episode']['r'])
                    self.env_sampling_total_count[self.current_int_seeds[i]] += 1
                    self.total_episodes_collected += 1

                    if self.agent.storage.use_proper_time_limits:
                        if 'truncated_obs' in info:
                            self.agent.storage.insert_truncated_obs(info['truncated_obs'], index=i)

                    # sample next level
                    if level_replay:
                        new_int_seed = self.level_sampler.sample_replay_level()
                    elif self._vlm_mode:
                        new_int_seed = self._sample_new_level_int_seed()
                    else:
                        # Procedural: replay from known pool
                        new_int_seed = random.choice(list(self.seed2level_id.keys()))

                    new_level_id = self.seed2level_id[new_int_seed]
                    obs_i = self.venv.reset_to_level(new_level_id, i)
                    self._set_obs_at_index(obs, obs_i, i)
                    self.current_int_seeds[i] = new_int_seed
                    self.total_seeds_collected += 1

            masks             = torch.FloatTensor([[0.0] if d else [1.0] for d in done])
            bad_masks         = torch.FloatTensor([[0.0] if 'truncated'   in info else [1.0] for info in infos])
            cliffhanger_masks = torch.FloatTensor([[0.0] if 'cliffhanger' in info else [1.0] for info in infos])
            level_seeds_t     = torch.tensor(self.current_int_seeds, dtype=torch.int).view(N, 1)

            self.agent.insert(
                obs, rnn_state, action, action_log_prob, action_log_dist, value,
                reward, masks, bad_masks=bad_masks, cliffhanger_masks=cliffhanger_masks,
                level_seeds=level_seeds_t,
            )

        # --- compute returns + update ---
        value_loss = action_loss = dist_entropy = None
        update_info = {}

        if update and self.is_training:
            with torch.no_grad():
                last_obs = self.agent.storage.get_obs(-1)
                next_value = self.agent.get_value(
                    last_obs,
                    self.agent.storage.get_recurrent_hidden_state(-1),
                    self.agent.storage.masks[-1],
                ).detach()

            self.agent.storage.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)
            value_loss, action_loss, dist_entropy, update_info = self.agent.update()
            self.num_updates += 1

        # --- update PLR scores ---
        if self.level_sampler.strategy not in ('random', 'off'):
            self.level_sampler.update_with_rollouts(self.agent.storage)
            self.level_sampler.after_update()

        return {
            'returns':           rollout_returns,
            'value_loss':        value_loss,
            'action_loss':       action_loss,
            'dist_entropy':      dist_entropy,
            'update_info':       update_info,
            'level_replay':      level_replay,
            'initial_int_seeds': int_seeds,
        }

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------
    def run(self, global_step: int, iteration: int, total_iterations: int) -> RunnerStats:
        # LR annealing
        if self.is_training:
            frac = 1.0 - (iteration - 1.0) / total_iterations
            self.agent.update_lr(frac * self.args.lr)

        # training rollout
        agent_info = self._agent_rollout(num_steps=self.args.num_steps, update=self.is_training)

        # collect returns for logging
        for b in agent_info['returns']:
            for r in reversed(b):
                self.agent_returns.append(r)
        mean_agent_return = (
            float(np.mean(self.agent_returns)) if len(self.agent_returns) > 0 else 0.0
        )

        self._sampled_level_info: SampledLevelInfo = {
            'source':       'plr',
            'env_ids':      [self.seed2level_id.get(s, str(s)) for s in agent_info['initial_int_seeds']],
            'level_replay': agent_info['level_replay'],
            'num_edits':    [0] * self.args.num_processes,
        }

        stats = RunnerStats(
            steps=global_step,
            global_step=global_step,
            total_episodes=self.total_episodes_collected,
            total_seeds=self.total_seeds_collected,
            mean_agent_return=mean_agent_return,
            agent_value_loss=agent_info['value_loss'],
            agent_pg_loss=agent_info['action_loss'],
            agent_dist_entropy=agent_info['dist_entropy'],
            agent_lr=agent_info['update_info'].get('lr', None),
        )
        stats.extra['level_replay']    = agent_info['level_replay']
        stats.extra['plr_buffer_fill'] = float(self.level_sampler._proportion_filled)
        return stats
