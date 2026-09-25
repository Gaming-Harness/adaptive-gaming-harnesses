#!/usr/bin/env python3
"""Cascaded World-Action Model (WAM) on project_root latents.

Architecture (survey: Cascaded WAM):
  World:   F_φ(z, a) → ẑ'     (existing ToyDynamics / pretrained_wm substrate)
  Action:  π_ψ(z) → a          (new InverseActionHead)

Training uses on-policy ARES dumps (z_t, a_vla→a_wm, z_{t+1}):
  L = L_dyn(F(z,a), z') + λ_act · L_act(π(z), a) [+ optional align]

Inference (cascaded):
  a = π(z);  ẑ' = F(z, a);  optionally re-act on ẑ' for H-step imagination.

This is NOT a Joint WAM (shared DiT). It is the practical path on the current
pretrained_wm + ARES stack: keep F, add action generation coupled to world latents.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from curriculum.frozen_wm import ToyDynamics


class InverseActionHead(nn.Module):
    """Decode pretrained_wm (kb[4], mouse[2]) from pooled latent z. Cascaded action model."""

    def __init__(
        self,
        latent_c: int = 16,
        kb_dim: int = 4,
        mouse_dim: int = 2,
        hidden: int = 256,
        mouse_scale: float = 0.4,
    ):
        super().__init__()
        self.mouse_scale = mouse_scale
        self.enc = nn.Sequential(
            nn.Linear(latent_c, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.kb_head = nn.Linear(hidden, kb_dim)
        self.mouse_head = nn.Linear(hidden, mouse_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: [B,C,f,H,W] or [B,C]
        returns kb [B,4] in (0,1), mouse [B,2] in [-scale, scale]
        """
        if z.ndim == 5:
            pooled = z.float().mean(dim=(2, 3, 4))
        elif z.ndim == 2:
            pooled = z.float()
        else:
            raise ValueError(f"InverseActionHead expects [B,C,…] or [B,C], got {tuple(z.shape)}")
        h = self.enc(pooled)
        kb = torch.sigmoid(self.kb_head(h))
        mouse = torch.tanh(self.mouse_head(h)) * self.mouse_scale
        return kb, mouse


class CascadedWAM(nn.Module):
    """World F + Action π, cascaded at inference."""

    def __init__(
        self,
        world: Optional[ToyDynamics] = None,
        action: Optional[InverseActionHead] = None,
        lambda_act: float = 1.0,
        lambda_open: float = 0.0,
    ):
        super().__init__()
        self.world = world if world is not None else ToyDynamics()
        self.action = action if action is not None else InverseActionHead()
        self.lambda_act = float(lambda_act)
        self.lambda_open = float(lambda_open)

    def act(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.action(z)

    def imagine(self, z: torch.Tensor, kb: torch.Tensor, mouse: torch.Tensor) -> torch.Tensor:
        return self.world(z, kb, mouse)

    def cascaded_step(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """One cascaded step: predict a from z, then imagine z'."""
        kb, mouse = self.act(z)
        z_hat = self.imagine(z, kb, mouse)
        return {"keyboard": kb, "mouse": mouse, "latent_hat": z_hat}

    def cascaded_rollout(self, z: torch.Tensor, horizon: int = 4) -> Dict[str, torch.Tensor]:
        zs = [z]
        kbs, mss = [], []
        cur = z
        for _ in range(max(1, int(horizon))):
            out = self.cascaded_step(cur)
            kbs.append(out["keyboard"])
            mss.append(out["mouse"])
            cur = out["latent_hat"]
            zs.append(cur)
        return {
            "latents": torch.stack(zs, dim=1),  # [B,H+1,…]
            "keyboard": torch.stack(kbs, dim=1),
            "mouse": torch.stack(mss, dim=1),
        }

    def loss(
        self,
        z: torch.Tensor,
        z1: torch.Tensor,
        kb_tgt: torch.Tensor,
        ms_tgt: torch.Tensor,
        *,
        train_world: bool = True,
        train_action: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Supervised cascaded losses on a real transition."""
        kb_hat, ms_hat = self.action(z)
        loss_act = F.mse_loss(kb_hat, kb_tgt) + F.mse_loss(ms_hat, ms_tgt)

        if train_world:
            # Teacher-forced world: use ground-truth a (stable), like standard WM.
            pred = self.world(z, kb_tgt, ms_tgt)
            loss_dyn = F.mse_loss(pred, z1)
        else:
            pred = z1
            loss_dyn = z.new_zeros(())

                # Open-loop term OFF by default (lambda_open=0): bad π poisons F.
        if train_action and train_world and self.lambda_open > 0:
            z_open = self.world(z, kb_hat, ms_hat)
            loss_open = F.mse_loss(z_open, z1)
        else:
            loss_open = z.new_zeros(())

        loss = loss_dyn + self.lambda_act * loss_act + self.lambda_open * loss_open
        return {
            "loss": loss,
            "L_dyn": loss_dyn.detach(),
            "L_act": loss_act.detach(),
            "L_open": loss_open.detach(),
            "pred": pred,
            "kb_hat": kb_hat,
            "ms_hat": ms_hat,
        }
