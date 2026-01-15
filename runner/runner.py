from enum import Enum
from baselines.common.running_mean_std import RunningMeanStd
from collections import deque
from typing import Optional
from ..interfaces import SampledLevelInfo, RunnerStats, RunnerStateDict
class AgentRole(str, Enum):
    AGENT = "agent"
    ADVERSARY_AGENT = "adversary_agent"
    ADVERSARY_ENV = "adversary_env"

class Runner:

    VALID_AGENT_ROLES = {
        AgentRole.AGENT,
        AgentRole.ADVERSARY_AGENT,
        AgentRole.ADVERSARY_ENV,}
    
    def __init__(self, args, venv, agents, agent_type, ued_venv=None):
        self.args = args
        self.venv = venv
        self.agents = agents
        self._validate_agents(agent_type)
        self.device = args.device
        self.agent_rollout_steps = args.num_steps
        # ---- shared bookkeeping ----
        self.num_updates = 0
        self.total_episodes_collected = 0
        self.total_seeds_collected = 0
        # last iteration info (for logging / screenshot)
        self._sampled_level_info: Optional[SampledLevelInfo] = None
        if args.adv_normalize_returns:
            self.env_return_rms = RunningMeanStd(shape=())

    def get_agent(self, role: AgentRole | str):
        if isinstance(role, str):
            role = AgentRole(role)
        agent = self.agents.get(role)
        if agent is None:
            raise RuntimeError(
                f"Agent role '{role.value}' is not available in {self.__class__.__name__}"
            )
        return agent
    
    def train(self, roles: list[str] | None = None):
        """Switch all agents to training mode."""
        roles = self._resolve_roles(roles)
        for role in roles:
            agent = self.get_agent(role)
            if agent is not None:
                agent.train()

    def eval(self, roles: list[str] | None = None):
        """Switch all agents to eval mode."""
        roles = self._resolve_roles(roles)
        for role in roles:
            agent = self.agents.get(role)
            if agent is not None:
                agent.eval()

    def reset(self):
        """Reset runner internal counters (called at experiment start)."""
        self.num_updates = 0
        self.total_episodes_collected = 0
        self.total_seeds_collected = 0
        self._sampled_level_info = None
        max_return_queue_size = 10
        self.agent_returns = deque(maxlen=max_return_queue_size)
        self.adversary_agent_returns = deque(maxlen=max_return_queue_size)

    def run(self, global_step, iteration, total_iterations)-> RunnerStats:
        raise NotImplementedError(
            f"{self.__class__.__name__}.run() is not implemented. "
            "Use a concrete Runner subclass (e.g. SFLRunner)."
        )

    @property
    def sampled_level_info(self):
        """
        Information about the levels sampled in the *last* run().
        Used for logging, screenshots, debugging.
        """
        return self._sampled_level_info


    def state_dict(self) -> RunnerStateDict:
        state: RunnerStateDict = super().state_dict()

        state["sfl"] = {
            "learnability_sampler": self.learnability_sampler.state_dict(),
            "env_sampling_total_count": dict(self.env_sampling_total_count),
            "env_sampling_current_count": dict(self.env_sampling_current_count),
        }

        return state

    def load_state_dict(self, state: dict):
        runner_state = state.get("runner", {})
        self.num_updates = runner_state.get("num_updates", 0)
        self.total_episodes_collected = runner_state.get(
            "total_episodes_collected", 0
        )
        self.total_seeds_collected = runner_state.get(
            "total_seeds_collected", 0
        )

        agents_state = state.get("agents", {})
        for role_str, agent_state in agents_state.items():
            role = AgentRole(role_str)
            if role in self.agents:
                self.agents[role].load_state_dict(agent_state)

    def _validate_agents(self, agent_type):
        if not isinstance(self.agents, dict):
            raise TypeError("agents must be a dict[AgentRole, Agent]")

        for k in self.agents.keys():
            if not isinstance(k, AgentRole):
                raise TypeError(f"Invalid agent key: {k} (must be AgentRole)")

        missing = agent_type - self.agents.keys()
        if missing:
            raise ValueError(f"Missing required agents: {missing}")
        unknown = set(self.agents.keys()) - agent_type
        if unknown:
            raise ValueError(f"Unknown agent roles: {unknown}")
        
    def _resolve_roles(self, roles: list[str] | None) -> list[AgentRole]:
        if roles is None:
            return list(self.agents.keys())

        resolved: list[AgentRole] = []

        for role_str in roles:
            try:
                role = AgentRole(role_str)
            except ValueError:
                raise ValueError(
                    f"Invalid agent role '{role_str}'. "
                    f"Valid roles: {[r.value for r in self.VALID_AGENT_ROLES]}"
                )

            if role not in self.agents:
                raise ValueError(
                    f"Agent role '{role.value}' not present in this runner. "
                    f"Available roles: {[r.value for r in self.agents.keys()]}"
                )

            resolved.append(role)

        return resolved