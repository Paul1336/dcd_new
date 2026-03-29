import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from env.registration import make as gym_make
from env.wrapper.parallel_wrappers import SubprocVecEnv
from interfaces import EvaluationStats
from eval.suites import load_iphyre_test_suite, load_minigrid_test_suite, MINIGRID_SUITE_NAMES


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
        method=getattr(args, 'method', ''),
        log_dir=getattr(args, 'log_dir', '.'),
        suite_num_tasks=getattr(args, 'test_suite_num_tasks', 20),
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
        method: str = '',
        log_dir: str = '.',
        suite_num_tasks: int = 20,
        **kwargs,
    ):
        self.test_suite_names = test_suite_names
        self.device = device
        self.num_episodes = num_episodes
        self.deterministic = deterministic
        self.record_video = record_video
        self.env_config = env_config or {}
        self.obs_encoder = obs_encoder
        self.method = method
        self.log_dir = log_dir
        self.suite_num_tasks = suite_num_tasks
        self._eval_count = 0
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

        # Pre-load and cache all minigrid suites so every eval uses the same
        # fixed set of tasks (suites are deterministic via base_seed=0).
        self._minigrid_suite_cache: Dict[str, tuple] = {}
        for name in test_suite_names:
            if name in MINIGRID_SUITE_NAMES:
                self._minigrid_suite_cache[name] = load_minigrid_test_suite(
                    name, num_tasks=self.suite_num_tasks
                )
                print(f"[Evaluator] Loaded suite {name}: "
                      f"{len(self._minigrid_suite_cache[name][0])} levels")

    def close(self):
        for e in self._opened_envs:
            try:
                e.close()
            except Exception:
                pass
        self._opened_envs = []

    def get_stats_keys(self) -> List[str]:
        """Keys emitted by evaluate(); used by train.py when eval is skipped."""
        keys = []
        for suite in self.test_suite_names:
            if suite in self._minigrid_suite_cache:
                env_names, _ = self._minigrid_suite_cache[suite]
                for env_id in env_names:
                    keys.append(f"{suite}/{env_id}/success_rate")
            keys.append(f"{suite}/mean_success_rate")
        return keys

    # ---------- public ----------

    def evaluate(self, agent) -> EvaluationStats:
        agent.eval()
        agent.to(self.device)
        self._eval_count += 1

        stats: Dict[str, float] = {}

        for test_suite_name in self.test_suite_names:
            print(f"Running test suite: {test_suite_name}")
            start_time = time.time()

            if test_suite_name in MINIGRID_SUITE_NAMES:
                env_names, env_fns = self._minigrid_suite_cache[test_suite_name]
                suite_results = self._evaluate_parallel_fns(env_names, env_fns, agent)
                # Record one trajectory per suite for visual inspection
                try:
                    self._record_trajectory(env_fns[0], agent, test_suite_name, env_names[0])
                except Exception as e:
                    print(f"[Trajectory] Warning: failed for {test_suite_name}: {e}")
            else:
                env_names, env_task_configs = load_iphyre_test_suite(test_suite_name)
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

    def _evaluate_parallel_fns(
        self,
        env_names: List[str],
        env_fns: List,
        agent,
        chunk_size: int = 50,
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        for start in range(0, len(env_names), chunk_size):
            chunk_names = env_names[start : start + chunk_size]
            chunk_fns   = env_fns[start : start + chunk_size]
            results.update(self._evaluate_chunk_fns(chunk_names, chunk_fns, agent))
        return results

    @staticmethod
    def _preprocess_minigrid_obs(obs: np.ndarray) -> Dict[str, torch.Tensor]:
        """(N,H,W,C) uint8 → {'image': tensor(N,C,H,W) float32 0-1}."""
        t = torch.from_numpy(obs).float() / 255.0   # (N,H,W,C)
        return {'image': t.permute(0, 3, 1, 2)}     # (N,C,H,W)

    @torch.no_grad()
    def _evaluate_chunk_fns(
        self,
        env_names: List[str],
        env_fns: List,
        agent,
    ) -> Dict[str, Dict[str, float]]:
        n = len(env_names)
        envs = SubprocVecEnv(env_fns, context='forkserver', is_eval=True)
        self._opened_envs.append(envs)

        episodic_returns: Dict[str, List[float]] = {name: [] for name in env_names}
        episodic_return = torch.zeros(n)

        obs = self._preprocess_minigrid_obs(envs.reset())

        hidden_size = agent.recurrent_hidden_state_size
        rnn_hxs = torch.zeros(n, hidden_size, device=self.device)
        if agent.is_lstm:
            rnn_hxs = (rnn_hxs, torch.zeros_like(rnn_hxs))
        masks = torch.ones(n, 1, device=self.device)

        while True:
            _, action, _, rnn_hxs = agent.act(obs, rnn_hxs, masks, deterministic=self.deterministic)

            action_np = action.cpu().numpy()
            obs, reward, done, _ = envs.step(action_np)
            obs = self._preprocess_minigrid_obs(obs)

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

        envs.close()
        self._opened_envs.remove(envs)

        results = {}
        for env_name in env_names:
            returns = np.array(episodic_returns[env_name])
            results[env_name] = {
                "success_rate": float(np.mean(returns > self.solved_threshold)),
                "mean_return": float(np.mean(returns)),
            }
        return results

    def _encode_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Apply obs_encoder if present, then convert to float32 tensor."""
        if self.obs_encoder is not None:
            obs = self.obs_encoder(obs)
        return torch.from_numpy(obs).to(dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _record_trajectory(self, env_fn: Callable, agent, suite_name: str, level_name: str):
        """Run one deterministic episode, save GIF to disk and log to wandb.

        File: {log_dir}/trajectories/{method}_eval{n:04d}_{suite}_{level}.gif
        Wandb key: trajectory/{suite_name}/{level_name}
        """
        try:
            import PIL.Image as PILImage
        except ImportError:
            return  # skip silently if Pillow not installed

        env = env_fn()
        raw_obs = env.reset()
        frames: List[np.ndarray] = []

        hidden_size = agent.recurrent_hidden_state_size
        rnn_hxs = torch.zeros(1, hidden_size, device=self.device)
        if agent.is_lstm:
            rnn_hxs = (rnn_hxs, torch.zeros_like(rnn_hxs))
        masks = torch.ones(1, 1, device=self.device)

        def _to_frame(raw) -> np.ndarray:
            """Extract (H, W, C) uint8 image from raw env obs."""
            img = raw['image'] if isinstance(raw, dict) else raw
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            return img.astype(np.uint8)

        def _to_agent_obs(raw) -> Dict:
            """(H,W,C) uint8 → {'image': tensor(1,C,H,W) float32 0-1}."""
            img = raw['image'] if isinstance(raw, dict) else raw
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            t = torch.from_numpy(img).float() / 255.0
            return {'image': t.unsqueeze(0).permute(0, 3, 1, 2).to(self.device)}

        frames.append(_to_frame(raw_obs))
        obs = _to_agent_obs(raw_obs)

        for _ in range(300):
            _, action, _, rnn_hxs = agent.act(obs, rnn_hxs, masks, deterministic=True)
            raw_obs, _, done, _ = env.step(int(action.cpu().item()))
            frames.append(_to_frame(raw_obs))
            obs = _to_agent_obs(raw_obs)
            masks = torch.zeros(1, 1, device=self.device) if done else masks
            if done:
                break
        env.close()

        # upscale each frame (MinGrid obs is 7×7, hard to see)
        scale = 16
        pil_frames = [
            PILImage.fromarray(f).resize(
                (f.shape[1] * scale, f.shape[0] * scale), PILImage.NEAREST
            )
            for f in frames
        ]

        # safe filename
        safe_suite = suite_name.replace('/', '-')
        safe_level = level_name.replace('/', '-')
        tag = f"{self.method}_eval{self._eval_count:04d}_{safe_suite}_{safe_level}"
        traj_dir = os.path.join(self.log_dir, 'trajectories')
        os.makedirs(traj_dir, exist_ok=True)
        gif_path = os.path.join(traj_dir, f"{tag}.gif")
        pil_frames[0].save(
            gif_path, save_all=True, append_images=pil_frames[1:],
            duration=150, loop=0,
        )
        print(f"[Trajectory] {tag} — {len(frames)} steps → {gif_path}")

        # log to wandb if active
        try:
            import wandb
            if wandb.run is not None:
                frames_np = np.stack([np.array(f) for f in pil_frames])  # (T,H,W,C)
                wandb.log(
                    {f"trajectory/{suite_name}/{level_name}": wandb.Video(
                        frames_np.transpose(0, 3, 1, 2), fps=6, format="gif"
                    )},
                    step=self._eval_count,
                )
        except Exception:
            pass

    @torch.no_grad()
    def _evaluate_parallel(
        self,
        env_names: List[str],
        env_task_configs: List[Any],
        agent,
        chunk_size: int = 50,
    ) -> Dict[str, Dict[str, float]]:
        """Run envs in parallel chunks of chunk_size via SubprocVecEnv.

        Workers use context='forkserver' so they are forked from the pre-CUDA
        forkserver process rather than the CUDA-initialised main process.
        chunk_size=50 is a practical limit on concurrent workers; increase if
        the host has enough file descriptors / memory.
        """
        results = {}
        for start in range(0, len(env_names), chunk_size):
            chunk_names   = env_names[start : start + chunk_size]
            chunk_configs = env_task_configs[start : start + chunk_size]
            results.update(self._evaluate_chunk(chunk_names, chunk_configs, agent))
        return results

    @torch.no_grad()
    def _evaluate_chunk(
        self,
        env_names: List[str],
        env_task_configs: List[Any],
        agent,
    ) -> Dict[str, Dict[str, float]]:
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
        envs = SubprocVecEnv(fns, context='forkserver', is_eval=True)
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

        envs.close()
        self._opened_envs.remove(envs)

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
