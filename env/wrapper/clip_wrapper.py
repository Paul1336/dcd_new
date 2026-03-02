import numpy as np
import torch
import gym

from .vec_env import VecEnvWrapper


class VecCLIPEmbeddingWrapper(VecEnvWrapper):
    """
    Vectorized wrapper that replaces image observations with CLIP embeddings.

    Applies a frozen CLIP vision encoder to each batch of observations produced
    by the wrapped environment, returning a flat float32 embedding vector instead
    of the raw image. The encoder runs inside torch.no_grad() so it accumulates
    no gradients during rollout collection.

    observation_space is updated to reflect the embedding shape, e.g. (512,)
    for ViT-B/32 or (768,) for ViT-L/14.

    Args:
        venv: Vectorized environment producing image observations [N, H, W, C].
        clip_model_name: CLIP model variant string passed to clip.load(),
            e.g. "ViT-B/32", "ViT-B/16", "ViT-L/14".
        clip_device: Device for the CLIP forward pass. Defaults to "cpu" to
            keep the rollout GPU budget free for the PPO network.
    """

    def __init__(
        self,
        venv,
        clip_model_name: str = "ViT-B/32",
        clip_device: str = "cpu",
    ):
        super().__init__(venv)

        try:
            import clip as openai_clip
        except ImportError:
            raise ImportError(
                "The 'clip' package is required. "
                "Install it with: pip install git+https://github.com/openai/CLIP.git"
            )

        self._clip_device = torch.device(clip_device)
        self._model, self._preprocess = openai_clip.load(
            clip_model_name, device=self._clip_device
        )
        self._model.eval()

        # Infer embedding dimension via a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self._clip_device)
            embed_dim = int(self._model.encode_image(dummy).shape[-1])

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(embed_dim,),
            dtype=np.float32,
        )

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        """
        Encode a batch of images into CLIP embeddings.

        Args:
            obs: np.ndarray [N, H, W, C] uint8 — batch of RGB images.

        Returns:
            np.ndarray [N, D] float32 — CLIP visual embeddings.
        """
        from PIL import Image

        imgs = [self._preprocess(Image.fromarray(obs[i])) for i in range(len(obs))]
        batch = torch.stack(imgs).to(self._clip_device)
        with torch.no_grad():
            emb = self._model.encode_image(batch)
        return emb.cpu().float().numpy()

    # --- VecEnv interface ---

    def reset(self) -> np.ndarray:
        return self._encode(self.venv.reset())

    def reset_agent(self) -> np.ndarray:
        return self._encode(self.venv.reset_agent())

    def reset_random(self) -> np.ndarray:
        return self._encode(self.venv.reset_random())

    def reset_to_level(self, level, index) -> np.ndarray:
        return self._encode(self.venv.reset_to_level(level, index))

    def reset_to_level_batch(self, level) -> np.ndarray:
        return self._encode(self.venv.reset_to_level_batch(level))

    def mutate_level(self, num_edits) -> np.ndarray:
        return self._encode(self.venv.mutate_level(num_edits))

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        obs = self._encode(obs)
        for i, info in enumerate(infos):
            if "truncated_obs" in info:
                # truncated_obs is a single image [H, W, C]; add/remove batch dim.
                infos[i]["truncated_obs"] = self._encode(info["truncated_obs"][np.newaxis])[0]
        return obs, rews, dones, infos
