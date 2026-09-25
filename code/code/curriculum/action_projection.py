"""Action projection bridge: VLA action space → WM conditioning.

Typical co-evolution practice (Dreamer / MuZero / VLA-WM):
  a_vla  --(encode)-->  e  --(proj)-->  a_wm / cond

We support two levels:
  1) Discrete MineStudio dict / cold-start text → pretrained_wm (kb[4], mouse[2])
     via action_codec (hard mapping for shared Minecraft keys).
  2) Learnable ActionProjection: embed richer VLA action features then
     project into WM action / latent-conditioning dims.

Without (2), WM only sees expert/external actions and never conditions on π.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from curriculum.action_codec import (
    KB_DIM,
    chunk_to_mg2,
    parse_actions_text,
    sandbox_to_mg2,
)

MOUSE_DIM = 2


# Ordered MineStudio binary keys used as a flat action feature (excl. camera).
_BINARY_KEYS = [
    "forward", "back", "left", "right", "jump", "sneak", "sprint",
    "attack", "use", "drop", "inventory", "ESC", "pickItem", "swapHands",
] + [f"hotbar.{i}" for i in range(1, 10)]


def ministudio_to_feature(action: Dict) -> torch.Tensor:
    """One MineStudio tick → flat feature [D] (binaries + camera pitch/yaw)."""
    bins = [float(action.get(k, 0)) for k in _BINARY_KEYS]
    cam = action.get("camera", [0.0, 0.0])
    pitch = float(cam[0]) / 180.0  # ~[-1,1]
    yaw = float(cam[1]) / 180.0
    return torch.tensor(bins + [pitch, yaw], dtype=torch.float32)


def chunk_to_feature(actions: List[Dict]) -> torch.Tensor:
    """List of ticks → [T, D]."""
    if not actions:
        return torch.zeros(1, len(_BINARY_KEYS) + 2)
    return torch.stack([ministudio_to_feature(a) for a in actions], dim=0)


class ActionProjection(nn.Module):
    """Learnable bridge: VLA action features → WM (kb, mouse) or cond vec.

    Forward:
      feat [B,T,D] or [B,D] → kb [B,4], mouse [B,2]  (mean-pooled over T)
      optional cond [B, cond_dim] for richer WM conditioning.
    """

    def __init__(
        self,
        in_dim: Optional[int] = None,
        hidden: int = 256,
        cond_dim: int = 64,
        mouse_scale: float = 0.4,
    ):
        super().__init__()
        in_dim = in_dim or (len(_BINARY_KEYS) + 2)
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.mouse_scale = mouse_scale
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.kb_head = nn.Linear(hidden, KB_DIM)
        self.mouse_head = nn.Linear(hidden, 2)
        self.cond_head = nn.Linear(hidden, cond_dim)
        # warm-start: bias kb toward identity on first 4 keys (w/s/a/d order)
        with torch.no_grad():
            self.kb_head.bias.zero_()
            # soft prior: projection can still learn; hard codec remains fallback

    def forward(
        self,
        feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          kb:    [B, 4] in (0,1) via sigmoid
          mouse: [B, 2] tanh * scale
          cond:  [B, cond_dim]
        """
        if feat.ndim == 3:
            feat = feat.float().mean(dim=1)  # [B,D]
        elif feat.ndim == 1:
            feat = feat.unsqueeze(0)
        h = self.enc(feat.float())
        kb = torch.sigmoid(self.kb_head(h))
        mouse = torch.tanh(self.mouse_head(h)) * self.mouse_scale
        cond = self.cond_head(h)
        return kb, mouse, cond

    def from_minestudio_chunk(
        self,
        actions: List[Dict],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = chunk_to_feature(actions).unsqueeze(0)  # [1,T,D]
        if device is not None:
            feat = feat.to(device)
        return self.forward(feat)

    def from_vla_text(
        self,
        text: str,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        actions, _ = parse_actions_text(text)
        if not actions:
            # fallback zeros
            B = 1
            z = torch.zeros(B, KB_DIM, device=device)
            m = torch.zeros(B, 2, device=device)
            c = torch.zeros(B, self.cond_dim, device=device)
            return z, m, c, []
        kb, mouse, cond = self.from_minestudio_chunk(actions, device=device)
        return kb, mouse, cond, actions


def hard_codec_mg2(actions: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Non-learnable bridge (shared Minecraft keys) — baseline / init target."""
    return chunk_to_mg2(actions)


def projection_align_loss(
    proj: ActionProjection,
    actions: List[Dict],
) -> torch.Tensor:
    """Supervise proj toward hard pretrained_wm codec so it starts as a faithful bridge."""
    feat = chunk_to_feature(actions).unsqueeze(0)
    kb_h, ms_h = hard_codec_mg2(actions)
    kb_t = kb_h.mean(0, keepdim=True)  # [1,4]
    ms_t = ms_h.mean(0, keepdim=True)  # [1,2]
    device = next(proj.parameters()).device
    kb_p, ms_p, _ = proj(feat.to(device))
    return F.mse_loss(kb_p, kb_t.to(device)) + F.mse_loss(ms_p, ms_t.to(device))
