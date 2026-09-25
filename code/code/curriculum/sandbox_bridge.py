"""MineStudio sandbox bridge for co-evolution (anonymous only).

Credentials come from the environment:

  MINESTUDIO_ENDPOINT   (default: https://model.example.com/sandboxGateway/system/714)
  MINESTUDIO_TOKEN      (required)
"""
from __future__ import annotations

import base64
import io
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


DEFAULT_ENDPOINT = "https://model.example.com/sandboxGateway/system/714"


@dataclass
class SandboxConfig:
    endpoint: str = ""
    token: str = ""
    obs_size: List[int] = field(default_factory=lambda: [128, 128])
    render_size: List[int] = field(default_factory=lambda: [640, 360])
    loading_command_steps: int = 20
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "SandboxConfig":
        return cls(
            endpoint=os.environ.get("MINESTUDIO_ENDPOINT", DEFAULT_ENDPOINT),
            token=os.environ.get("MINESTUDIO_TOKEN", "").strip(),
        )

    def validate(self) -> None:
        if not self.token:
            raise RuntimeError(
                "Missing MINESTUDIO_TOKEN. Export it before starting a sandbox run."
            )


def _ares_root() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "ares")
    )


class MineStudioSandbox:
    """Thin wrapper around ares MineStudioEnv / SandboxTool (import only, no edits)."""

    def __init__(
        self,
        cfg: Optional[SandboxConfig] = None,
        img_save_dir: str = "",
        task_id: Tuple[int, int] = (0, 0),
    ):
        self.cfg = cfg or SandboxConfig.from_env()
        self.cfg.validate()
        self.task_id = task_id
        self.img_save_dir = img_save_dir or os.path.join(
            os.path.dirname(__file__), "outputs", "sandbox_images"
        )
        os.makedirs(self.img_save_dir, exist_ok=True)
        self._env = None
        self._last_obs: Optional[Dict[str, Any]] = None

    def _ensure_env(self):
        if self._env is not None:
            return
        ares = _ares_root()
        if ares not in sys.path:
            sys.path.insert(0, ares)
        from ares.mm_agent.envs_and_tools.openworld.minestudio_env import (
            MineStudioConfig,
            MineStudioEnv,
        )
        env_cfg = MineStudioConfig(
            name="openworld/minestudio",
            application_secret_token=self.cfg.token,
            cluster_endpoint=self.cfg.endpoint,
            img_save_dir=self.img_save_dir,
            max_retries=self.cfg.max_retries,
            request_delay=1.0,
            loading_command_steps=self.cfg.loading_command_steps,
            obs_size=list(self.cfg.obs_size),
            render_size=list(self.cfg.render_size),
        )
        self._env = MineStudioEnv(task_id=self.task_id, config=env_cfg)

    def reset(self, task_config_file_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        self._ensure_env()
        obs = self._env.reset(task_config_file_path=task_config_file_path, **kwargs)
        self._last_obs = obs
        return obs

    def step(self, action: Any) -> Dict[str, Any]:
        """action: one MineStudio dict OR a list (chunk) for step_batch."""
        self._ensure_env()
        obs = self._env.step(action)
        self._last_obs = obs
        return obs

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    @property
    def last_image(self) -> Optional[Image.Image]:
        if not self._last_obs:
            return None
        return extract_pil(self._last_obs)

    @property
    def last_reward(self) -> float:
        if not self._last_obs:
            return 0.0
        r = self._last_obs.get("reward", 0.0)
        try:
            return float(r)
        except (TypeError, ValueError):
            return 0.0


def extract_pil(obs: Dict[str, Any]) -> Optional[Image.Image]:
    """Best-effort POV image from MineStudio / ARES observation dict."""
    if obs is None:
        return None
    path = obs.get("local_image_path")
    if path and os.path.exists(path):
        return Image.open(path).convert("RGB")
    for key in ("image", "pov", "rgb"):
        v = obs.get(key)
        if v is None and isinstance(obs.get("observation"), dict):
            v = obs["observation"].get(key)
        if v is None:
            continue
        if isinstance(v, Image.Image):
            return v.convert("RGB")
        if isinstance(v, np.ndarray):
            arr = v
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW
                arr = np.transpose(arr, (1, 2, 0))
            return Image.fromarray(arr[..., :3])
        if isinstance(v, (bytes, bytearray)):
            return Image.open(io.BytesIO(v)).convert("RGB")
        if isinstance(v, str) and v.startswith("data:image"):
            b64 = v.split(",", 1)[-1]
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return None


def pov_to_uint8(img: Image.Image, size: Tuple[int, int] = (640, 360)) -> np.ndarray:
    """RGB uint8 HxWx3 at render size (pretrained_wm / OpenHA default 640x360)."""
    if img.size != (size[0], size[1]):
        # PIL size is (W,H); OpenHA render_size is [W,H]
        img = img.resize((size[0], size[1]), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)
