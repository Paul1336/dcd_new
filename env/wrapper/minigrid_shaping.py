"""Potential-based reward shaping for MiniGrid/MultiGrid envs.

F(s) = -manhattan_dist(agent_pos, goal_pos) / (width + height)
shaped_reward = reward + coef * (gamma * F(s') - F(s))

Works for both env families:
  - GoalLastAdversarialEnv  (MultiGrid-*):  exposes env.goal_pos directly
  - OfficialMiniGridTrainingEnv (MiniGrid-*): scans grid for 'goal' cell

Also intercepts all PLR/SFL reset variants so shaping state stays consistent:
  reset(), reset_to_level(), reset_random(), reset_agent()
"""


def _get_goal_pos(env):
    """Return (gx, gy) or None. Tries env.goal_pos first, then grid scan."""
    # GoalLastAdversarialEnv and AdversarialEnv expose goal_pos directly
    gp = getattr(env, 'goal_pos', None)
    if gp is not None:
        return int(gp[0]), int(gp[1])

    # OfficialMiniGridTrainingEnv: unwrap to reach the gymnasium env's grid
    inner = env
    while inner is not None:
        grid = getattr(inner, 'grid', None)
        if grid is not None:
            w = getattr(inner, 'width', None)
            h = getattr(inner, 'height', None)
            if w and h:
                for i in range(w):
                    for j in range(h):
                        cell = grid.get(i, j)
                        if cell is not None and getattr(cell, 'type', None) == 'goal':
                            return i, j
        inner = getattr(inner, '_env', None) or getattr(inner, 'env', None)

    return None


def _get_agent_pos(env):
    """Return (ax, ay) or None."""
    # GoalLastAdversarialEnv: agent_pos is a list of positions per agent
    ap = getattr(env, 'agent_pos', None)
    if ap is not None:
        if isinstance(ap, (list, tuple)) and len(ap) > 0:
            first = ap[0]
            if first is not None:
                return int(first[0]), int(first[1])
        else:
            # numpy array or direct (x, y)
            try:
                return int(ap[0]), int(ap[1])
            except (TypeError, IndexError):
                pass

    # OfficialMiniGridTrainingEnv: dig into _env
    inner = env
    while inner is not None:
        ap = getattr(inner, 'agent_pos', None)
        if ap is not None:
            try:
                return int(ap[0]), int(ap[1])
            except (TypeError, IndexError):
                pass
        inner = getattr(inner, '_env', None) or getattr(inner, 'env', None)

    return None


class MiniGridShapingWrapper:
    """Thin wrapper that adds potential-based shaping without breaking
    the PLR/SFL env interface (reset_to_level, reset_random, reset_agent,
    encoding, mutate_level, etc. are all forwarded transparently).
    """

    def __init__(self, env, gamma: float = 0.99, coef: float = 0.5):
        self._env   = env
        self._gamma = gamma
        self._coef  = coef
        self._prev_pot = 0.0

        # Forward spaces and other attributes
        self.observation_space = env.observation_space
        self.action_space      = env.action_space
        self.spec              = getattr(env, 'spec', None)

    # ── Potential helpers ────────────────────────────────────────────────────

    def _dims(self):
        w = getattr(self._env, 'width', None)
        h = getattr(self._env, 'height', None)
        # OfficialMiniGridTrainingEnv: dig into _env
        inner = self._env
        while (w is None or h is None) and inner is not None:
            w = getattr(inner, 'width', w)
            h = getattr(inner, 'height', h)
            inner = getattr(inner, '_env', None)
        return (w or 11), (h or 11)

    def _potential(self):
        goal = _get_goal_pos(self._env)
        agent = _get_agent_pos(self._env)
        if goal is None or agent is None:
            return 0.0
        w, h = self._dims()
        dist = abs(agent[0] - goal[0]) + abs(agent[1] - goal[1])
        return -dist / (w + h)

    def _post_reset(self, obs):
        self._prev_pot = self._potential()
        return obs

    # ── gym interface ────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        return self._post_reset(obs)

    def step(self, action):
        result = self._env.step(action)
        obs, reward, done, info = result
        new_pot        = self._potential()
        shaping        = self._gamma * new_pot - self._prev_pot
        self._prev_pot = new_pot
        info = dict(info) if info else {}
        info['true_reward'] = float(reward)
        return obs, reward + self._coef * shaping, done, info

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        return self._env.close()

    # ── PLR / SFL interface ──────────────────────────────────────────────────

    def _fix_timelimit(self):
        # gym.make() wraps the env in gym's TimeLimit which lacks reset_to_level;
        # those calls proxy via __getattr__ and never set _elapsed_steps, so we
        # must do it manually here after every non-standard reset.
        if hasattr(self._env, '_elapsed_steps'):
            self._env._elapsed_steps = 0

    def reset_to_level(self, level):
        obs = self._env.reset_to_level(level)
        self._fix_timelimit()
        return self._post_reset(obs)

    def reset_random(self):
        obs = self._env.reset_random()
        self._fix_timelimit()
        return self._post_reset(obs)

    def reset_agent(self):
        obs = self._env.reset_agent()
        self._fix_timelimit()
        return self._post_reset(obs)

    def mutate_level(self, num_edits=1):
        obs = self._env.mutate_level(num_edits=num_edits)
        self._fix_timelimit()
        return self._post_reset(obs)

    # ── Transparent attribute forwarding ────────────────────────────────────

    def __getattr__(self, name):
        return getattr(self._env, name)
