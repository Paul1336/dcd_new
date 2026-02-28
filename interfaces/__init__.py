# interfaces/returns.py
from typing import TypedDict, List, Optional, Any, Dict, Union, Tuple, Iterator
import torch
from torch import Tensor

# -------------------------
# Primitive tensor types (shared across model / storage / runner)
# -------------------------

# RNN hidden state: plain Tensor for GRU/non-recurrent, (h, c) tuple for LSTM
RnnHiddenState = Union[Tensor, Tuple[Tensor, Tensor]]

# Observation: flat Tensor (symbolic) or dict of Tensors (multi-modal)
ObsTensor = Union[Tensor, Dict[str, Tensor]]

# Return type of act(): (value, action, actor_log_dist, rnn_hxs)
ActOutput = Tuple[Tensor, Tensor, Tensor, RnnHiddenState]

# Minibatch tuple yielded by feed_forward_generator / recurrent_generator
#   (obs, rnn_hxs, actions, value_preds, returns, masks, old_log_probs, adv_targ)
RolloutBatch = Tuple[
    ObsTensor,        # obs_batch
    RnnHiddenState,   # recurrent_hidden_states_batch
    Tensor,           # actions_batch
    Tensor,           # value_preds_batch
    Tensor,           # return_batch
    Tensor,           # masks_batch
    Tensor,           # old_action_log_probs_batch
    Optional[Tensor], # adv_targ (None when advantages=None)
]


# -------------------------
# Debug / observation
# -------------------------

class FullObservation(TypedDict, total=False):
    state: Any
    objects: Any
    metadata: Any


# -------------------------
# PPO update diagnostics
# -------------------------

class PPOUpdateInfo(TypedDict, total=False):
    grad_norms:  List[float]  # per-mini-batch L2 grad norm; only when log_grad_norm=True
    approx_kl:   float        # E[(r-1) - log r] from last mini-batch of last epoch
    clipfracs:   float        # mean fraction of samples outside clip range
    used_epoch:  int          # epochs actually completed (may be < ppo_epoch due to early-stop)
    kl_loss:     float        # mean KL regularisation loss; only when kl_loss_coef > 0
    lr:          float        # Adam lr at time of return


# -------------------------
# Rollout
# -------------------------

class EpisodeResult(TypedDict):
    env_id: str                 # level name / seed / task id
    return_: float
    success: Optional[bool]     # SFL / curriculum 可用，沒有就 None


class RolloutResult(TypedDict):
    # episodic returns per process: List[List[float]]
    returns: List[List[float]]

    # --- optimization-related (trainer cares) ---
    value_loss: Optional[float]
    action_loss: Optional[float]
    dist_entropy: Optional[float]

    update_info: PPOUpdateInfo
    sampled_levels: Optional[List[List[str]]]  # level history per process


# -------------------------
# Runner statistics (per run())
# -------------------------

class RunnerCoreState(TypedDict, total=False):
    num_updates: int
    total_episodes_collected: int
    total_seeds_collected: int
    agent_returns: List[float]


class RunnerStateDict(TypedDict, total=False):
    runner: RunnerCoreState
    agents: Dict[str, Dict[str, Any]]

    # extensions (optional)
    sfl: Dict[str, Any]
    plr: Dict[str, Any]
    paired: Dict[str, Any]

class RunnerStats:
    """Container for per-iteration training statistics returned by Runner.run()."""

    def __init__(
        self,
        steps: int = 0,
        global_step: int = 0,
        total_episodes: int = 0,
        total_seeds: int = 0,
        mean_agent_return: float = 0.0,
        agent_value_loss: Optional[float] = None,
        agent_pg_loss: Optional[float] = None,
        agent_dist_entropy: Optional[float] = None,
        agent_lr: Optional[float] = None,
        sps: Optional[float] = None,
    ):
        self.steps = steps
        self.global_step = global_step
        self.total_episodes = total_episodes
        self.total_seeds = total_seeds
        self.mean_agent_return = mean_agent_return
        self.agent_value_loss = agent_value_loss
        self.agent_pg_loss = agent_pg_loss
        self.agent_dist_entropy = agent_dist_entropy
        self.agent_lr = agent_lr
        self.sps = sps
        self.extra: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "steps": self.steps,
            "global_step": self.global_step,
            "total_episodes": self.total_episodes,
            "total_seeds": self.total_seeds,
            "mean_agent_return": self.mean_agent_return,
        }
        if self.agent_value_loss is not None:
            d["agent_value_loss"] = self.agent_value_loss
        if self.agent_pg_loss is not None:
            d["agent_pg_loss"] = self.agent_pg_loss
        if self.agent_dist_entropy is not None:
            d["agent_dist_entropy"] = self.agent_dist_entropy
        if self.agent_lr is not None:
            d["agent_lr"] = self.agent_lr
        if self.sps is not None:
            d["sps"] = self.sps
        d.update(self.extra)
        return d


# -------------------------
# Curriculum / level sampling
# -------------------------

class SampledLevelInfo(TypedDict):
    source: str                 # "learnability" | "plr" | "random" | ...
    env_ids: List[str]          # one per env process
    level_replay: bool
    num_edits: List[int]


# -------------------------
# Evaluation statistics
# -------------------------

class EvaluationStats:
    """Flat container for evaluation results returned by Evaluator.evaluate()."""
    def __init__(self):
        self.extra: Dict[str, Any] = {}
        self.total_student_grad_updates: Optional[int] = None
        self.global_step: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = dict(self.extra)
        if self.total_student_grad_updates is not None:
            d["total_student_grad_updates"] = self.total_student_grad_updates
        if self.global_step is not None:
            d["global_step"] = self.global_step
        return d
