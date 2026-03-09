import random

import numpy as np

_VLM_ENV_NAMES = frozenset({
    'Iphyre-AdversarialVLM4k-v0',
    'Iphyre-AdversarialVLM10k-v0',
    'Iphyre-AdversarialClaudeVLM10k-v0',
    'Iphyre-AdversarialGeminiVLM10k-v0',
    'MultiGrid-AdversarialVLM-v0',
})


def _enc_key(enc):
    """Return a hashable dict key for any encoding type.

    Iphyre encodings are strings (already hashable).
    MultiGrid encodings are numpy arrays → convert to bytes.
    """
    if isinstance(enc, np.ndarray):
        return enc.tobytes()
    return enc


class LearnabilitySampler(object):

    def __init__(self,
                 venv,
                 learnability_alpha=0.5,
                 learnability_c=0.0,
                 top_k_to_sample_uniformly=-1,
                 staleness=0.1,
                 ued_algo=None,
                 env_name='',
                 max_pool_size=1000,
                 ):
        self.task_info_dict = {}
        self.learnability_alpha = learnability_alpha
        self.learnability_c = learnability_c
        self.top_k_to_sample_uniformly = top_k_to_sample_uniformly
        self.staleness = staleness
        self.ued_algo = ued_algo
        self.max_pool_size = max_pool_size
        self._vlm_mode = env_name in _VLM_ENV_NAMES

        if self._vlm_mode:
            self.env_names = venv.remote_attr('subsampled_env_ids', index=[0])[0][0]
            print('Learnability Sampler: First 10 env_names: ', self.env_names[:10])
            print('Learnability Sampler: len(env_names): ', len(self.env_names))
            self.task_info_dict = {
                env_id: {
                    'zero_shot_success_rate': 0.0,
                    'last_updated_global_step_for_learnability': 0
                }
                for env_id in self.env_names
            }
        else:
            # Procedural: start with empty pool; levels added via register_level()
            self.env_names = []
            print('Learnability Sampler: procedural mode, pool starts empty (max_pool_size=%d).' % max_pool_size)

        self.learnability_last_updated_global_step = -1

    def register_level(self, encoding) -> bool:
        """
        Add a new procedural level encoding to the pool (no-op if already present).
        When pool is full (>= max_pool_size), drop the oldest entry first.
        Returns True if the level was newly registered.

        encoding may be a str (iphyre) or numpy array (multigrid).
        The dict key is always hashable (_enc_key); env_names stores the original.
        """
        key = _enc_key(encoding)
        if key in self.task_info_dict:
            return False
        if len(self.env_names) >= self.max_pool_size:
            oldest = self.env_names.pop(0)
            del self.task_info_dict[_enc_key(oldest)]
        self.env_names.append(encoding)       # store original (e.g. numpy array)
        self.task_info_dict[key] = {
            'zero_shot_success_rate': 0.0,
            'last_updated_global_step_for_learnability': 0,
        }
        return True

    def update_learnability(self, env_id, global_step, success_rate):
        key = _enc_key(env_id)
        if key not in self.task_info_dict:
            raise ValueError(f"Env not found in learnability sampler (key={key!r})")

        print('update learnability for env_id: ', env_id, 'with success_rate: ', success_rate)
        self.task_info_dict[key] = {
            'zero_shot_success_rate': success_rate,
            'last_updated_global_step_for_learnability': global_step
        }
        self.learnability_last_updated_global_step = global_step

    def state_dict(self) -> dict:
        # Serialize: convert bytes keys to hex strings, numpy env_names to lists
        def _ser_key(k):
            return k.hex() if isinstance(k, bytes) else k
        def _ser_val(v):
            return v.tolist() if isinstance(v, np.ndarray) else v
        return {
            "task_info_dict": {_ser_key(k): v for k, v in self.task_info_dict.items()},
            "env_names": [_ser_val(e) for e in self.env_names],
            "learnability_last_updated_global_step": self.learnability_last_updated_global_step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.task_info_dict = state.get("task_info_dict", self.task_info_dict)
        self.learnability_last_updated_global_step = state.get(
            "learnability_last_updated_global_step", -1
        )
        # VLM mode: env_names is always rebuilt from subsampled_env_ids in __init__.
        # Procedural mode: restore the pool so sampling works immediately after resume.
        if not self._vlm_mode:
            saved = state.get("env_names", [])
            if saved:
                self.env_names = list(saved)

    def update_env_names(self, env_names):
        self.env_names = env_names
        print('update env_names: ', self.env_names)

    def wrap_level_result(self, env_name):
        """Return a level identifier for venv.reset_to_level().

        Numpy arrays (multigrid) are returned directly so reset_to_level()
        receives the actual grid encoding.  Strings (iphyre) are returned as-is.
        """
        if isinstance(env_name, np.ndarray):
            return env_name
        return str(env_name)

    def sample(self):
        if not self.env_names:
            return None  # pool empty; caller must handle (e.g. call reset_random())

        epsilon = 1e-6
        if self.learnability_alpha is None:
            raise ValueError("learnability_alpha must be set for beta priority")

        # Levels whose learnability was evaluated in the most recent update step.
        sampled_env_ids = [
            env_id
            for env_id in self.env_names
            if self.task_info_dict[_enc_key(env_id)][
                "last_updated_global_step_for_learnability"
            ] == self.learnability_last_updated_global_step
        ]

        # Before the first learnability update (or when no levels match), fall back
        # to uniform random from the full pool rather than crashing.
        if not sampled_env_ids:
            return self.wrap_level_result(random.choice(self.env_names))

        task_priorities = [
            (
                self.task_info_dict[_enc_key(env_id)]["zero_shot_success_rate"]
                + self.learnability_c
            ) ** (self.learnability_alpha)
            * (
                1
                - self.task_info_dict[_enc_key(env_id)]["zero_shot_success_rate"]
                + self.learnability_c
            ) ** (1 - self.learnability_alpha)
            + epsilon
            for env_id in sampled_env_ids
        ]

        is_stale = random.random() < self.staleness
        if is_stale:
            return self.wrap_level_result(random.choice(self.env_names))
        else:
            if self.top_k_to_sample_uniformly > 0:
                top_k_env_ids = sorted(
                    sampled_env_ids,
                    key=lambda x: task_priorities[sampled_env_ids.index(x)],
                    reverse=True,
                )[: self.top_k_to_sample_uniformly]
                return self.wrap_level_result(random.choice(top_k_env_ids))
            else:
                return self.wrap_level_result(
                    random.choices(sampled_env_ids, task_priorities, k=1)[0]
                )
