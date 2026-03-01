import random
from collections import defaultdict
import numpy as np
import torch

from ..runner import Runner, AgentRole
from .level_sampler import LevelSampler
from interfaces import SampledLevelInfo, RunnerStats, RolloutResult, RunnerStateDict
from util import make_plr_args


class ACCELRunner(Runner):
    """
    ACCEL runner: PLR curriculum + level editing.

    Curriculum is maintained by a LevelSampler (PLR) that scores levels by
    value-loss (or another strategy) and replays high-scoring levels.
    When a replayed level is selected and the edit roll succeeds, that level
    is mutated, the agent is evaluated on the mutated level, and the mutated
    level is added to the PLR buffer.

    Level IDs in the iphyre env are strings (VLM task names or rand_int_seed
    strings).  LevelSampler requires integer seeds, so we maintain a
    bijective mapping:  int_seed  <->  str_level_id.
    - Initial VLM tasks  : int_seed = sequential index 0, 1, 2, ...
    - Mutated levels     : int_seed = self._next_mut_seed (auto-incremented)
    """

    def __init__(self, args, venv, agents, ued_venv=None, train=True):
        required_roles = {AgentRole.AGENT}
        if getattr(args, 'use_accel_paired', False):
            required_roles.add(AgentRole.ADVERSARY_AGENT)

        super().__init__(
            args=args,
            venv=venv,
            agents=agents,
            required_roles=required_roles,
            ued_venv=ued_venv,
        )

        self.agent = self.agents[AgentRole.AGENT]
        self.adversary_agent = self.agents.get(AgentRole.ADVERSARY_AGENT)

        # ---- int seed <-> str level_id mapping ----
        env_names = venv.remote_attr('subsampled_env_ids', index=[0])[0][0]
        self._vlm_env_names = list(env_names)
        self.seed2level_id: dict = {i: env_id for i, env_id in enumerate(env_names)}
        self.level_id2seed: dict = {env_id: i for i, env_id in enumerate(env_names)}
        self._next_mut_seed: int = len(env_names)   # counter for mutated level ids

        # ---- PLR level sampler ----
        plr_kwargs = make_plr_args(args, venv.observation_space, venv.action_space)
        self.level_sampler = LevelSampler(**plr_kwargs)

        # ---- ACCEL config ----
        self.use_editor   = getattr(args, 'use_editor', False)
        self.edit_prob    = getattr(args, 'level_editor_prob', 0.0)
        self.base_levels  = getattr(args, 'base_levels', 'batch')
        self.weighted_num_edits = 0.0
        self.total_num_edits    = 0

        # current active int seed per env (updated during rollout)
        self.current_int_seeds = [-1] * args.num_processes

        # env sampling counters
        self.env_sampling_total_count   = defaultdict(int)
        self.env_sampling_current_count = defaultdict(int)

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
                "num_updates":             self.num_updates,
                "total_episodes_collected": self.total_episodes_collected,
                "total_seeds_collected":   self.total_seeds_collected,
                "agent_returns":           list(self.agent_returns),
            },
            "agents": {
                role.value: agent.state_dict()
                for role, agent in self.agents.items()
            },
            "plr": {
                "seed2level_id":            {str(k): v for k, v in self.seed2level_id.items()},
                "level_id2seed":            dict(self.level_id2seed),
                "_next_mut_seed":           self._next_mut_seed,
                "env_sampling_total_count": dict(self.env_sampling_total_count),
                "total_num_edits":          self.total_num_edits,
            },
        }

    def load_state_dict(self, state: dict):
        super().load_state_dict(state)

        plr_state = state.get("plr")
        if not plr_state:
            return

        self.seed2level_id = {int(k): v for k, v in plr_state.get("seed2level_id", {}).items()}
        self.level_id2seed = dict(plr_state.get("level_id2seed", {}))
        self._next_mut_seed = plr_state.get("_next_mut_seed", len(self._vlm_env_names))
        self.env_sampling_total_count = defaultdict(int, plr_state.get("env_sampling_total_count", {}))
        self.total_num_edits = plr_state.get("total_num_edits", 0)

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

    def _sample_new_level_int_seed(self) -> int:
        """Randomly pick from the initial VLM task pool."""
        env_id = random.choice(self._vlm_env_names)
        return self.level_id2seed[env_id]

    def _should_edit_level(self) -> bool:
        return self.use_editor and (np.random.rand() < self.edit_prob)

    def _register_mutated_level(self, str_seed: str) -> int:
        """Register a newly mutated level string ID and return its int seed."""
        if str_seed not in self.level_id2seed:
            int_seed = self._next_mut_seed
            self._next_mut_seed += 1
            self.seed2level_id[int_seed] = str_seed
            self.level_id2seed[str_seed] = int_seed
        return self.level_id2seed[str_seed]

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
        else:
            int_seeds = [self._sample_new_level_int_seed() for _ in range(N)]
            self.level_sampler.observe_external_unseen_sample(int_seeds)

        self.current_int_seeds = list(int_seeds)
        level_ids = [self.seed2level_id[s] for s in int_seeds]

        self.venv.reset_to_level_batch(level_ids)
        self.total_seeds_collected += N

        obs = self.venv.reset_agent()
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
                    else:
                        new_int_seed = self._sample_new_level_int_seed()

                    new_level_id = self.seed2level_id[new_int_seed]
                    obs_i = self.venv.reset_to_level(new_level_id, i)
                    self._set_obs_at_index(obs, obs_i, i)
                    self.current_int_seeds[i] = new_int_seed
                    self.total_seeds_collected += 1

            masks            = torch.FloatTensor([[0.0] if d else [1.0] for d in done])
            bad_masks        = torch.FloatTensor([[0.0] if 'truncated'   in info else [1.0] for info in infos])
            cliffhanger_masks = torch.FloatTensor([[0.0] if 'cliffhanger' in info else [1.0] for info in infos])
            level_seeds_t    = torch.tensor(self.current_int_seeds, dtype=torch.int).view(N, 1)

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
            'returns':          rollout_returns,
            'value_loss':       value_loss,
            'action_loss':      action_loss,
            'dist_entropy':     dist_entropy,
            'update_info':      update_info,
            'level_replay':     level_replay,
            'initial_int_seeds': int_seeds,
        }

    # -----------------------------------------------------------------------
    # ACCEL eval rollout (scores mutated levels for PLR; no weight update)
    # -----------------------------------------------------------------------
    def _accel_eval_loop(self, obs, int_seeds: list, num_steps: int) -> None:
        """
        Run a rollout on already-reset envs (post-mutation) to score them
        in the PLR buffer.  Uses agent.storage in-place; does NOT update
        agent weights.
        """
        args = self.args
        N = args.num_processes

        self.agent.storage.set_obs(obs, 0)
        self.agent.storage.masks[0].fill_(1.0)
        self.agent.storage.recurrent_hidden_states[0].fill_(0.0)

        current_int_seeds = list(int_seeds)

        for step in range(num_steps):
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
                    if self.agent.storage.use_proper_time_limits:
                        if 'truncated_obs' in info:
                            self.agent.storage.insert_truncated_obs(info['truncated_obs'], index=i)
                    # Reset to same mutated level to stay on it for eval
                    obs_i = self.venv.reset_to_level(self.seed2level_id[current_int_seeds[i]], i)
                    self._set_obs_at_index(obs, obs_i, i)

            masks             = torch.FloatTensor([[0.0] if d else [1.0] for d in done])
            bad_masks         = torch.FloatTensor([[0.0] if 'truncated'   in info else [1.0] for info in infos])
            cliffhanger_masks = torch.FloatTensor([[0.0] if 'cliffhanger' in info else [1.0] for info in infos])
            level_seeds_t     = torch.tensor(current_int_seeds, dtype=torch.int).view(N, 1)

            self.agent.insert(
                obs, rnn_state, action, action_log_prob, action_log_dist, value,
                reward, masks, bad_masks=bad_masks, cliffhanger_masks=cliffhanger_masks,
                level_seeds=level_seeds_t,
            )

        # compute returns so PLR scoring functions (value_l1 etc.) work
        with torch.no_grad():
            last_obs = self.agent.storage.get_obs(-1)
            next_value = self.agent.get_value(
                last_obs,
                self.agent.storage.get_recurrent_hidden_state(-1),
                self.agent.storage.masks[-1],
            ).detach()
        self.agent.storage.compute_returns(
            next_value, self.args.use_gae, self.args.gamma, self.args.gae_lambda
        )

        # score mutated levels in PLR (no weight update)
        if self.level_sampler.strategy not in ('random', 'off'):
            self.level_sampler.update_with_rollouts(self.agent.storage)
            self.level_sampler.after_update()

    # -----------------------------------------------------------------------
    # ACCEL editing phase
    # -----------------------------------------------------------------------
    def _accel_edit_phase(self, level_replay: bool, initial_int_seeds: list, rollout_result: dict):
        """
        If conditions are met:
          1. Select base levels (batch / easy / hard).
          2. Reset envs to base levels, then mutate.
          3. Register new level IDs with PLR.
          4. Run eval rollout to score new levels.
        """
        if not level_replay or not self._should_edit_level():
            return

        args = self.args
        N = args.num_processes

        # --- select base levels ---
        if self.base_levels == 'easy':
            # "easy" heuristic from archive: argmin(mean_return - value_loss)
            # = levels with lowest (mean_return - value_loss) ~ hardest-but-learnable
            per_env_returns = rollout_result['returns']
            mean_returns = np.array([
                float(np.mean(r)) if r else 0.0
                for r in per_env_returns
            ])
            batched_vl = (
                self.agent.storage.get_batched_value_loss(positive_only=True, batched=True)
                .detach().cpu().numpy().flatten()
            )
            scores = mean_returns - batched_vl
            if N >= 4:
                easy_indices = list(np.argsort(scores)[:4])
                fixed_int_seeds = [initial_int_seeds[x] for x in easy_indices] * (N // 4)
            else:
                easy_idx = int(np.argmin(scores))
                fixed_int_seeds = [initial_int_seeds[easy_idx]] * N
        else:
            # 'batch' or 'hard' (hard not implemented separately; use batch)
            fixed_int_seeds = list(initial_int_seeds)

        # --- reset envs to base levels and mutate ---
        base_level_ids = [self.seed2level_id[s] for s in fixed_int_seeds]
        self.venv.reset_to_level_batch(base_level_ids)
        obs = self.venv.mutate_level(num_edits=args.num_edits)

        # --- get new level seeds from envs ---
        new_str_seeds = [
            self.venv.remote_attr('level_seed', index=[i])[0][0]
            for i in range(N)
        ]
        new_int_seeds = [self._register_mutated_level(s) for s in new_str_seeds]

        # --- register with PLR as unseen ---
        self.level_sampler.observe_external_unseen_sample(new_int_seeds)

        # --- run eval rollout to score mutated levels ---
        self.current_int_seeds = list(new_int_seeds)
        self._accel_eval_loop(obs, new_int_seeds, num_steps=args.num_steps)

        self.total_num_edits += 1

    # -----------------------------------------------------------------------
    # main entry point
    # -----------------------------------------------------------------------
    def run(self, global_step: int, iteration: int, total_iterations: int) -> RunnerStats:
        # LR annealing
        if self.is_training:
            frac = 1.0 - (iteration - 1.0) / total_iterations
            self.agent.update_lr(frac * self.args.lr)

        # training rollout
        agent_info = self._agent_rollout(num_steps=self.args.num_steps, update=self.is_training)

        # ACCEL editing phase
        if self.is_training and self.use_editor:
            self._accel_edit_phase(
                level_replay=agent_info['level_replay'],
                initial_int_seeds=agent_info['initial_int_seeds'],
                rollout_result=agent_info,
            )

        # collect returns for logging
        for b in agent_info['returns']:
            for r in reversed(b):
                self.agent_returns.append(r)
        mean_agent_return = (
            float(np.mean(self.agent_returns)) if len(self.agent_returns) > 0 else 0.0
        )

        # weighted_num_edits for logging (proportion of buffer that's been edited)
        if self.use_editor and self.level_sampler.is_warm:
            try:
                w = self.level_sampler.sample_weights()
                # fraction of buffer slots that are mutated levels
                n_vlm = len(self._vlm_env_names)
                edited_mask = np.array([
                    (int(self.level_sampler.seeds[i]) >= n_vlm)
                    for i in range(len(self.level_sampler.seeds))
                ], dtype=np.float64)
                self.weighted_num_edits = float(np.dot(w, edited_mask))
            except Exception:
                pass

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
        stats.extra['level_replay']       = agent_info['level_replay']
        stats.extra['total_num_edits']    = self.total_num_edits
        stats.extra['weighted_num_edits'] = self.weighted_num_edits
        stats.extra['plr_buffer_fill']    = float(self.level_sampler._proportion_filled)
        return stats
