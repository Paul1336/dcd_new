import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from env.registration import make as gym_make
from env.wrapper.parallel_wrappers import SubprocVecEnv
from interfaces import EvaluationStats
from env.benchmark.iphyre.suites import load_test_suite


def create_evaluator(args, obs_encoder: Optional[Callable] = None):
    test_suite_names = [x.strip() for x in args.test_env_names.split(",") if x.strip()]
    env_config = {"state_type": "image"} if getattr(args, "obs_type", "symbolic") == "embedding" else {}
    return Evaluator(
        test_suite_names=test_suite_names,
        num_episodes=args.test_num_episodes,
        device=args.device,
        deterministic=getattr(args, "deterministic_eval", False),
        record_video=getattr(args, "record_video", False),
        env_config=env_config,
        obs_encoder=obs_encoder,
    )


class Evaluator:
    """
    Decoupled evaluator.
    - Loads test suites via eval/suites/iphyre.py
    - Runs all envs in parallel via gym.vector.SyncVectorEnv (matching archive behaviour)
    - Calls agent.act(obs, rnn_hxs, masks, deterministic) — positional, returns 4-tuple
    - Returns EvaluationStats (compatible with train.py)

    Args:
        env_config: dict forwarded to gym_make as the `config` kwarg for each eval env.
            When obs_type="embedding", pass {"state_type": "image"} so the env produces
            (224,224,3) images that obs_encoder can process.
        obs_encoder: optional callable np.ndarray [N,H,W,C] → np.ndarray [N,D].
            Applied to raw observations before feeding the agent.  Required when
            the agent was trained with CLIP embeddings.
    """

    def __init__(
        self,
        test_suite_names: List[str],
        num_episodes: int = 10,
        device: str = "cpu",
        deterministic: bool = False,
        record_video: bool = False,
        env_config: Optional[Dict] = None,
        obs_encoder: Optional[Callable] = None,
        **kwargs,
    ):
        self.test_suite_names = test_suite_names
        self.device = device
        self.num_episodes = num_episodes
        self.deterministic = deterministic
        self.record_video = record_video
        self.env_config = env_config or {}
        self.obs_encoder = obs_encoder
        self.kwargs = kwargs

        first = test_suite_names[0] if test_suite_names else ""
        if "Bipedal" in first:
            self.solved_threshold = 230.0
        elif "Iphyre" in first:
            self.solved_threshold = 0.9
        else:
            self.solved_threshold = 0.0
        print("Solved threshold:", self.solved_threshold)

        self._opened_envs = []

    def close(self):
        for e in self._opened_envs:
            try:
                e.close()
            except Exception:
                pass
        self._opened_envs = []

    def get_stats_keys(self) -> List[str]:
        """Keys emitted by evaluate(); used by train.py when eval is skipped."""
        return [f"{suite}/mean_success_rate" for suite in self.test_suite_names]

    # ---------- public ----------

    def evaluate(self, agent) -> EvaluationStats:
        agent.eval()
        agent.to(self.device)

        stats: Dict[str, float] = {}

        for test_suite_name in self.test_suite_names:
            print(f"Running test suite: {test_suite_name}")
            start_time = time.time()

            env_names, env_task_configs = load_test_suite(test_suite_name)
            suite_results = self._evaluate_parallel(env_names, env_task_configs, agent)

            total_success_rates = []
            for env_id in env_names:
                sr = suite_results[env_id]["success_rate"]
                total_success_rates.append(sr)
                print(f"[{env_id}] Success rate: [{sr}]")
                stats[f"{test_suite_name}/{env_id}/success_rate"] = sr

            mean_sr = float(np.mean(total_success_rates)) if total_success_rates else 0.0
            stats[f"{test_suite_name}/mean_success_rate"] = mean_sr
            print(f"Test suite {test_suite_name} average success rate: {mean_sr}")
            print(f"Time taken: {time.time() - start_time:.1f}s")

        ev = EvaluationStats()
        ev.extra = stats
        return ev

    # ---------- internals ----------

    def _encode_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Apply obs_encoder if present, then convert to float32 tensor."""
        if self.obs_encoder is not None:
            obs = self.obs_encoder(obs)
        return torch.from_numpy(obs).to(dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _evaluate_parallel(
        self,
        env_names: List[str],
        env_task_configs: List[Any],
        agent,
    ) -> Dict[str, Dict[str, float]]:
        """Run all envs in parallel; collect num_episodes per env.

        Uses SubprocVecEnv (the same worker used by training) for all obs types.
        gym.vector.AsyncVectorEnv triggers a mandatory reset in __init__ to
        populate its shared-memory observation buffer; this deadlocks with
        iphyre envs because the forced reset races with pygame initialisation.
        SubprocVecEnv defers all resets until the caller invokes envs.reset(),
        avoiding the deadlock.
        """
        n = len(env_names)
        env_config = self.env_config

        def make_env_fn(env_name, env_task_config):
            def thunk():
                return gym_make(
                    "Iphyre-Game-v0",
                    env_name=env_name,
                    env_task_config=env_task_config,
                    config=env_config,
                )
            return thunk

        fns = [make_env_fn(name, cfg) for name, cfg in zip(env_names, env_task_configs)]
        # Use the project's own SubprocVecEnv (same worker that training uses).
        # gym.vector.AsyncVectorEnv sends a mandatory reset during __init__ to
        # populate shared memory, which deadlocks with the iphyre/pygame envs.
        # SubprocVecEnv only queries one worker for spaces during __init__ and
        # defers resets until the caller explicitly calls envs.reset().
        envs = SubprocVecEnv(fns, is_eval=True)
        self._opened_envs.append(envs)

        episodic_returns: Dict[str, List[float]] = {name: [] for name in env_names}
        episodic_return = torch.zeros(n)

        obs = self._encode_obs(envs.reset())

        hidden_size = agent.recurrent_hidden_state_size
        rnn_hxs = torch.zeros(n, hidden_size, device=self.device)
        if agent.is_lstm:
            rnn_hxs = (rnn_hxs, torch.zeros_like(rnn_hxs))
        masks = torch.ones(n, 1, device=self.device)

        while True:
            _, action, _, rnn_hxs = agent.act(obs, rnn_hxs, masks, deterministic=self.deterministic)

            action_np = action.cpu().numpy()
            obs, reward, done, _ = envs.step(action_np)
            obs = self._encode_obs(obs)

            episodic_return += torch.tensor(reward, dtype=torch.float32)
            masks = torch.tensor(
                [[0.0] if d else [1.0] for d in done],
                dtype=torch.float32, device=self.device,
            )

            for i, env_name in enumerate(env_names):
                if done[i] and len(episodic_returns[env_name]) < self.num_episodes:
                    episodic_returns[env_name].append(episodic_return[i].item())
                    episodic_return[i] = 0.0

            if all(len(episodic_returns[name]) >= self.num_episodes for name in env_names):
                break

        results = {}
        for env_name in env_names:
            returns = np.array(episodic_returns[env_name])
            results[env_name] = {
                "success_rate": float(np.mean(returns > self.solved_threshold)),
                "mean_return": float(np.mean(returns)),
            }
        return results


def evaluate_parallel_envs(
    env_names: List[str],
    env_task_configs: List[Any],
    agent,
    num_episodes: int = 10,
    device: str = "cpu",
    solved_threshold: float = 0.9,
    env_config: Optional[Dict] = None,
    obs_encoder: Optional[Callable] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Standalone wrapper used by SFLRunner._update_learnability_metrics.
    Delegates to a temporary Evaluator instance.
    """
    ev = Evaluator(
        test_suite_names=[],
        num_episodes=num_episodes,
        device=device,
        env_config=env_config or {},
        obs_encoder=obs_encoder,
    )
    ev.solved_threshold = solved_threshold
    results = ev._evaluate_parallel(env_names, env_task_configs, agent)
    ev.close()
    return results
