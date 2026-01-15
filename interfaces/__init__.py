# interfaces/returns.py
from typing import TypedDict, List, Optional, Any, Dict
import torch


# -------------------------
# Debug / observation
# -------------------------

class FullObservation(TypedDict, total=False):
    state: Any
    objects: Any
    metadata: Any


# -------------------------
# Rollout
# -------------------------

class EpisodeResult(TypedDict):
    env_id: str                 # level name / seed / task id
    return_: float
    success: Optional[bool]     # SFL / curriculum 可用，沒有就 None


class RolloutResult(TypedDict):
    # --- episode-level outcomes (curriculum cares about this) ---
    episodes: List[EpisodeResult]

    # --- optimization-related (trainer cares) ---
    value_loss: Optional[torch.Tensor]
    action_loss: Optional[torch.Tensor]
    dist_entropy: Optional[torch.Tensor]

    update_info: Dict[str, Any]


# -------------------------
# Runner statistics (per run())
# -------------------------

class RunnerCoreState(TypedDict):
    num_updates: int
    total_episodes_collected: int
    total_seeds_collected: int


class RunnerStateDict(TypedDict, total=False):
    runner: RunnerCoreState
    agents: Dict[str, Dict[str, Any]]

    # extensions (optional)
    sfl: Dict[str, Any]
    plr: Dict[str, Any]
    paired: Dict[str, Any]

class RunnerStats(TypedDict):
    # --- progress ---
    steps: int
    global_step: int
    total_episodes: int
    total_seeds: int

    # --- returns ---
    mean_agent_return: float

    # --- losses ---
    agent_value_loss: Optional[float]
    agent_pg_loss: Optional[float]
    agent_dist_entropy: Optional[float]

    # --- optimization ---
    agent_lr: Optional[float]
    sps: Optional[float]

    # --- debug / curriculum ---
    sampled_level_info: Optional["SampledLevelInfo"]


# -------------------------
# Curriculum / level sampling
# -------------------------

class SampledLevelInfo(TypedDict):
    source: str                 # "learnability" | "plr" | "random" | ...
    env_ids: List[str]          # one per env process
    level_replay: bool
    num_edits: List[int]
