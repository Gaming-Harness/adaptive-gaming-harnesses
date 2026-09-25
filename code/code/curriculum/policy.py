"""Policy interfaces for Stage-2 VLA RL.

Provides:
  - PolicyBase: act / evaluate / update_ppo
  - LatentMLPPolicy: lightweight smoke / baseline on latent features
  - ExternalVLAPolicy: thin wrapper for an external Qwen-VLA callable

Full Qwen3-VL training stays outside (e.g. ares / SenseNova-MARS); this package
only needs a π(a|o) that returns pretrained_wm keyboard/mouse actions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


KB_DIM = 4
MOUSE_DIM = 2


def pool_latent(z: torch.Tensor) -> torch.Tensor:
    """[B,16,f,H,W] or [16,f,H,W] -> [B, feat]."""
    if z.ndim == 4:
        z = z.unsqueeze(0)
    # mean over frames + spatial
    return z.float().mean(dim=(2, 3, 4))  # [B, 16]


@dataclass
class ActResult:
    keyboard: torch.Tensor   # [B, kb] logits or probs
    mouse: torch.Tensor      # [B, 2]
    log_prob: torch.Tensor   # [B]
    value: torch.Tensor      # [B]
    entropy: torch.Tensor    # [B]


class PolicyBase(ABC, nn.Module):
    @abstractmethod
    def act(self, latent: torch.Tensor, deterministic: bool = False) -> ActResult:
        ...

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "class": self.__class__.__name__}, path)

    def load(self, path: str, map_location: str = "cpu") -> None:
        ck = torch.load(path, map_location=map_location, weights_only=False)
        self.load_state_dict(ck["state_dict"])


class LatentMLPPolicy(PolicyBase):
    """Small MLP policy on pooled latents — for curriculum smoke & baselines.

    Keyboard: Bernoulli(logit) per key.
    Mouse: Gaussian mean + fixed/learned log_std.
    """

    def __init__(
        self,
        in_dim: int = 16,
        hidden: int = 256,
        kb_dim: int = KB_DIM,
        mouse_dim: int = MOUSE_DIM,
        mouse_scale: float = 0.4,
    ):
        super().__init__()
        self.kb_dim = kb_dim
        self.mouse_dim = mouse_dim
        self.mouse_scale = mouse_scale
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.kb_head = nn.Linear(hidden, kb_dim)
        self.mouse_mean = nn.Linear(hidden, mouse_dim)
        self.mouse_log_std = nn.Parameter(torch.zeros(mouse_dim) - 1.0)
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(pool_latent(latent))
        kb_logits = self.kb_head(h)
        mouse_mu = torch.tanh(self.mouse_mean(h)) * self.mouse_scale
        value = self.v_head(h).squeeze(-1)
        return kb_logits, mouse_mu, self.mouse_log_std.expand_as(mouse_mu), value

    def act(self, latent: torch.Tensor, deterministic: bool = False) -> ActResult:
        kb_logits, mouse_mu, mouse_log_std, value = self.forward(latent)
        kb_dist = torch.distributions.Bernoulli(logits=kb_logits)
        mouse_std = mouse_log_std.exp().clamp(min=1e-4)
        mouse_dist = torch.distributions.Normal(mouse_mu, mouse_std)

        if deterministic:
            kb = (kb_logits > 0).float()
            mouse = mouse_mu
        else:
            kb = kb_dist.sample()
            mouse = mouse_dist.rsample()

        log_prob = kb_dist.log_prob(kb).sum(-1) + mouse_dist.log_prob(mouse).sum(-1)
        entropy = kb_dist.entropy().sum(-1) + mouse_dist.entropy().sum(-1)
        return ActResult(
            keyboard=kb,
            mouse=mouse,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
        )

    def evaluate_actions(
        self,
        latent: torch.Tensor,
        keyboard: torch.Tensor,
        mouse: torch.Tensor,
    ) -> ActResult:
        kb_logits, mouse_mu, mouse_log_std, value = self.forward(latent)
        kb_dist = torch.distributions.Bernoulli(logits=kb_logits)
        mouse_std = mouse_log_std.exp().clamp(min=1e-4)
        mouse_dist = torch.distributions.Normal(mouse_mu, mouse_std)
        log_prob = kb_dist.log_prob(keyboard).sum(-1) + mouse_dist.log_prob(mouse).sum(-1)
        entropy = kb_dist.entropy().sum(-1) + mouse_dist.entropy().sum(-1)
        return ActResult(
            keyboard=keyboard,
            mouse=mouse,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
        )


class ExternalVLAPolicy(PolicyBase):
    """Wrap an external VLA: fn(obs_dict) -> {keyboard, mouse, log_prob?, value?}."""

    def __init__(self, fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
        super().__init__()
        self.fn = fn
        # dummy param so .to(device) / optimizers don't break
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def act(self, latent: torch.Tensor, deterministic: bool = False) -> ActResult:
        out = self.fn({"latent": latent, "deterministic": deterministic})
        kb = out["keyboard"]
        mouse = out["mouse"]
        if not isinstance(kb, torch.Tensor):
            kb = torch.as_tensor(kb, dtype=torch.float32)
        if not isinstance(mouse, torch.Tensor):
            mouse = torch.as_tensor(mouse, dtype=torch.float32)
        B = kb.shape[0] if kb.ndim > 1 else 1
        device = self._dummy.device
        kb = kb.to(device).float()
        mouse = mouse.to(device).float()
        if kb.ndim == 1:
            kb = kb.unsqueeze(0)
        if mouse.ndim == 1:
            mouse = mouse.unsqueeze(0)
        log_prob = out.get("log_prob")
        value = out.get("value")
        if log_prob is None:
            log_prob = torch.zeros(B, device=device)
        else:
            log_prob = torch.as_tensor(log_prob, device=device).float().view(B)
        if value is None:
            value = torch.zeros(B, device=device)
        else:
            value = torch.as_tensor(value, device=device).float().view(B)
        entropy = torch.zeros(B, device=device)
        return ActResult(keyboard=kb, mouse=mouse, log_prob=log_prob, value=value, entropy=entropy)


def ppo_update(
    policy: LatentMLPPolicy,
    optimizer: torch.optim.Optimizer,
    latents: torch.Tensor,
    keyboards: torch.Tensor,
    mouses: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """One PPO epoch on a batch. Only valid for LatentMLPPolicy (has evaluate_actions)."""
    assert hasattr(policy, "evaluate_actions")
    res = policy.evaluate_actions(latents, keyboards, mouses)
    ratio = (res.log_prob - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(res.value, returns)
    entropy_loss = -res.entropy.mean()
    loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(res.entropy.mean().item()),
    }
