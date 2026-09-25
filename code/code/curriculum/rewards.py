"""Reward helpers for Stage-2.

Default rewards are imagination-quality proxies (usable without game score APIs).
Plug in sandbox task rewards when available.
"""
from __future__ import annotations

from typing import Optional

import torch


def latent_sharpness(z: torch.Tensor) -> torch.Tensor:
    """High-frequency proxy on latent. z: [..., C, f, H, W] or [C, f, H, W]."""
    if z.ndim == 4:
        z = z.unsqueeze(0)
    g = z.float().mean(dim=1)          # [B,f,H,W]
    d = g[..., 1:] - g[..., :-1]
    return d.var(dim=(-1, -2, -3)).mean(dim=-1)  # [B] or scalar-ish


def dynamics_consistency_reward(
    pred_tp1: torch.Tensor,
    true_tp1: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """r = -||pred - true|| (higher is better)."""
    err = (pred_tp1.float() - true_tp1.float()).pow(2).mean(dim=tuple(range(1, pred_tp1.ndim)))
    return -scale * err


def sharpness_preserve_reward(
    z_t: torch.Tensor,
    z_tp1: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalize sharpness collapse across a step (common WM failure mode)."""
    s0 = latent_sharpness(z_t)
    s1 = latent_sharpness(z_tp1)
    drop = torch.relu(s0 - s1) / (s0.abs() + 1e-6)
    return -scale * drop


def combine_rewards(
    *parts: torch.Tensor,
    weights: Optional[list] = None,
) -> torch.Tensor:
    if not parts:
        raise ValueError("no reward parts")
    if weights is None:
        weights = [1.0] * len(parts)
    assert len(weights) == len(parts)
    acc = weights[0] * parts[0]
    for w, p in zip(weights[1:], parts[1:]):
        acc = acc + w * p
    return acc
