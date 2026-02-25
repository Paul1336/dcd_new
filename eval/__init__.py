# eval/__init__.py
import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

from env.registration import make as gym_make  # <- use your env/registration, not star-import envs.*
from interfaces import EvaluationStats  # keep your typed object


def create_evaluator(args):
    return Evaluator(
        test_env_names=args.test_env_names,
        num_episodes=args.num_episodes_test if hasattr(args, "num_episodes_test") else args.num_episodes,
        device=args.device,
        deterministic=getattr(args, "deterministic_eval", False),
        record_video=getattr(args, "record_video", False),
    )


class Evaluator:
    """
    Decoupled evaluator:
    - builds envs via env.registration.make
    - only calls agent.act(...)
    - no access to runner/algo/storage internals
    """

    def __init__(
        self,
        test_env_names: str,
        num_episodes: int,
        device: str = "cpu",
        deterministic: bool = False,
        record_video: bool = False,
        **kwargs,
    ):
        # test_env_names can be CSV: "EnvA,EnvB" or suite name(s) for Iphyre.
        self.test_env_names_raw = test_env_names
        self.device = device
        self.num_episodes = num_episodes
        self.deterministic = deterministic
        self.record_video = record_video
        self.kwargs = kwargs

        self.env_specs = self._build_env_specs(test_env_names)
        # env objects created lazily per evaluation call (avoid fork issues)
        self._opened_envs = []

    def close(self):
        for e in self._opened_envs:
            try:
                e.close()
            except Exception:
                pass
        self._opened_envs = []

    def get_stats_keys(self) -> List[str]:
        keys = []
        for suite_name, env_id in self.env_specs:
            keys.append(f"{suite_name}/{env_id}/success_rate")
        # plus aggregate keys
        suites = sorted(set(s for s, _ in self.env_specs))
        for s in suites:
            keys.append(f"{s}/mean_success_rate")
        return keys

    # ---------- public ----------
    def evaluate(self, agent) -> EvaluationStats:
        agent.eval()
        agent.to(self.device)

        stats: Dict[str, float] = {}
        suite_to_scores: Dict[str, List[float]] = {}

        for suite_name, env_id in self.env_specs:
            env, solved_threshold = self._make_env_for_spec(suite_name, env_id)
            self._opened_envs.append(env)

            out = self._eval_one_env(env, agent, solved_threshold=solved_threshold)
            sr = out["success_rate"]

            stats[f"{suite_name}/{env_id}/success_rate"] = sr
            suite_to_scores.setdefault(suite_name, []).append(sr)

        for suite_name, scores in suite_to_scores.items():
            stats[f"{suite_name}/mean_success_rate"] = float(np.mean(scores)) if scores else 0.0

        # Convert to your EvaluationStats
        # assuming EvaluationStats has .extra or .to_dict, adapt here
        ev = EvaluationStats()
        ev.extra = stats  # if your dataclass uses `extra`
        return ev

    # ---------- internals ----------
    def _build_env_specs(self, test_env_names: str) -> List[Tuple[str, str]]:
        """
        Returns list of (suite_name, env_id).
        For normal envs: suite_name == env_name, env_id == env_name.
        For Iphyre suites: suite_name == suite, env_id == task_name.
        """
        names = [x.strip() for x in test_env_names.split(",") if x.strip()]
        env_specs: List[Tuple[str, str]] = []
        for name in names:
            if name.startswith("Iphyre-"):
                env_ids, _ = self._load_test_suite_envs_iphyre(name)
                for env_id in env_ids:
                    env_specs.append((name, env_id))
            else:
                env_specs.append((name, name))
        return env_specs

    def _infer_solved_threshold(self, suite_or_env_name: str) -> float:
        if "Bipedal" in suite_or_env_name:
            return 230.0
        if "Iphyre" in suite_or_env_name:
            return 0.9
        return 0.0

    def _make_env_for_spec(self, suite_name: str, env_id: str):
        # For Iphyre suite, env_id is task_name and we need task_config
        if suite_name.startswith("Iphyre-"):
            env_names, env_task_configs = self._load_test_suite_envs_iphyre(suite_name)
            idx = env_names.index(env_id)
            task_cfg = env_task_configs[idx]
            env = gym_make("Iphyre-Game-v0", env_name=env_id, env_task_config=task_cfg)
        else:
            env = gym_make(env_id)

        env = RecordEpisodeStatistics(env)
        solved_threshold = self._infer_solved_threshold(suite_name)
        return env, solved_threshold

    def _load_test_suite_envs_iphyre(self, test_suite_name: str):
        task_suite_path = {
            "Iphyre-HandDesign-v0": "../iphyre/test_toy20250110/20250525/output_hand_test",
            "Iphyre-ProceduralShift-v0": "../iphyre/test_toy20250110/20250602/output_eval_shift",
            "Iphyre-ProceduralRotate-v0": "../iphyre/test_toy20250110/20250602/output_eval_rotate",
            "Iphyre-VLMGeneratedShift-v0": "../iphyre/test_toy20250110/20250427/output_shift",
            "Iphyre-VLMGeneratedRotate-v0": "../iphyre/test_toy20250110/20250427/output_rotate",
        }
        tasks_path = task_suite_path[test_suite_name]
        task_dirs = os.listdir(tasks_path)

        env_names = []
        env_task_configs = []
        test_hard_limit = 100

        for task_dir in task_dirs:
            config_path = os.path.join(tasks_path, task_dir, "config.json")
            config = json.load(open(config_path))
            task_config = config["config"]
            task_name = task_dir

            if "VLM" in test_suite_name:
                if config.get("success_rate", 0.0) > 0.0:
                    env_names.append(task_name)
                    env_task_configs.append(task_config)
            else:
                env_names.append(task_name)
                env_task_configs.append(task_config)

            if len(env_names) >= test_hard_limit:
                break

        return env_names, env_task_configs

    @torch.no_grad()
    def _eval_one_env(self, env, agent, solved_threshold: float) -> Dict[str, Any]:
        returns: List[float] = []
        solved = 0

        obs = env.reset()
        rnn_state = None
        mask = None

        while len(returns) < self.num_episodes:
            out = agent.act(obs, rnn_state=rnn_state, mask=mask, deterministic=self.deterministic)

            # expect either dict or tuple; keep evaluator generic
            if isinstance(out, dict):
                action = out["action"]
                rnn_state = out.get("rnn_state", None)
            else:
                # tolerate common patterns
                action = out[0] if len(out) == 2 else out[1]
                rnn_state = out[-1] if len(out) >= 2 else None

            obs, reward, done, info = env.step(action)

            if done:
                ep = info.get("episode", None)
                r = float(ep["r"]) if ep is not None else float(reward)
                returns.append(r)
                if r > solved_threshold:
                    solved += 1

                obs = env.reset()
                rnn_state = None
                mask = None

        return {
            "success_rate": float(solved / self.num_episodes),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
        }
