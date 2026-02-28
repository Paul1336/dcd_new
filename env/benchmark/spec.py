from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class BenchmarkSpec:
    """
    Bundles all default hyper-parameters for one benchmark.

    Attributes
    ----------
    env_name : str
        Canonical env name (e.g. "Iphyre-Adversarial-v0").
    env_cfg : dict
        Rollout / environment parameters (num_steps, gamma, …).
    model_cfg : dict
        Actor-critic architecture parameters (recurrent_hidden_size, …).
    ppo_cfg : dict
        PPO optimisation parameters (lr, ppo_epoch, …).
    eval_cfg : dict
        Evaluation parameters (test_env_names, test_num_episodes, …).
    """
    env_name: str
    env_cfg: Dict[str, Any] = field(default_factory=dict)
    model_cfg: Dict[str, Any] = field(default_factory=dict)
    ppo_cfg: Dict[str, Any] = field(default_factory=dict)
    eval_cfg: Dict[str, Any] = field(default_factory=dict)

    def apply_to(self, args) -> None:
        """
        Write spec defaults into *args* without overwriting keys that are
        already explicitly set (i.e. non-None and non-empty).

        Call this right after argparse, before anything else reads args.
        """
        for cfg in (self.env_cfg, self.model_cfg, self.ppo_cfg, self.eval_cfg):
            for k, v in cfg.items():
                if getattr(args, k, None) is None:
                    setattr(args, k, v)
