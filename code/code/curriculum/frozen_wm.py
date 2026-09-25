"""Frozen world-model imagination substrate (Stage 1 output / Stage 2 teacher).

Two backends:
  - full: project_root CausalInferencePipeline / ActionRolloutPipeline (expensive)
  - toy:  tiny latent residual dynamics (curriculum smoke without loading 1.3B)

Sparse reality: optional DriftPredictor trigger → caller may re-ground via sandbox.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mg2_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_paths():
    root = _mg2_root()
    cl = os.path.join(root, "close-loop")
    for p in (root, cl):
        if p not in sys.path:
            sys.path.insert(0, p)


@dataclass
class Stage1Bundle:
    """Frozen Stage-1 checkpoint set."""
    generator_ckpt: str
    config_path: str = "configs/inference_yaml/inference_universal.yaml"
    pretrained_model_path: str = "pretrained_wm"
    predictor_ckpt: Optional[str] = None
    hamiltonian_ae_ckpt: Optional[str] = None
    phase_world_ckpt: Optional[str] = None
    oracle_reground_ckpt: Optional[str] = None
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generator_ckpt": self.generator_ckpt,
            "config_path": self.config_path,
            "pretrained_model_path": self.pretrained_model_path,
            "predictor_ckpt": self.predictor_ckpt,
            "hamiltonian_ae_ckpt": self.hamiltonian_ae_ckpt,
            "phase_world_ckpt": self.phase_world_ckpt,
            "oracle_reground_ckpt": self.oracle_reground_ckpt,
            "notes": self.notes,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Stage1Bundle":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Stage1Bundle":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def verify(self, root: Optional[str] = None) -> List[str]:
        root = root or _mg2_root()
        missing = []
        for key in (
            "generator_ckpt", "config_path", "predictor_ckpt",
            "hamiltonian_ae_ckpt", "phase_world_ckpt", "oracle_reground_ckpt",
        ):
            p = getattr(self, key)
            if not p:
                continue
            full = p if os.path.isabs(p) else os.path.join(root, p)
            if not os.path.exists(full):
                missing.append(f"{key}: {full}")
        return missing


class ToyDynamics(nn.Module):
    """Cheap latent residual dynamics for curriculum smoke tests.

    F_toy(z, a) = z + MLP([pool(z), a]) broadcast into latent shape.
    """

    def __init__(self, latent_c: int = 16, kb_dim: int = 4, mouse_dim: int = 2, hidden: int = 128):
        super().__init__()
        in_dim = latent_c + kb_dim + mouse_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_c),
        )
        self.latent_c = latent_c

    def forward(self, z: torch.Tensor, keyboard: torch.Tensor, mouse: torch.Tensor) -> torch.Tensor:
        # z: [B,16,f,H,W]; actions may be [B,T,d] — mean-pool over action time
        if keyboard.ndim == 3:
            kb = keyboard.float().mean(dim=1)
        else:
            kb = keyboard.float()
        if mouse.ndim == 3:
            ms = mouse.float().mean(dim=1)
        else:
            ms = mouse.float()
        pooled = z.float().mean(dim=(2, 3, 4))  # [B,C]
        delta = self.net(torch.cat([pooled, kb, ms], dim=-1))  # [B,C]
        return z + delta.view(z.shape[0], z.shape[1], 1, 1, 1)


class FrozenWorldModel:
    """Stage-2 teacher: frozen imagination engine."""

    def __init__(
        self,
        backend: str = "toy",
        bundle: Optional[Stage1Bundle] = None,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        toy_ckpt: Optional[str] = None,
    ):
        self.backend = backend
        self.bundle = bundle
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.pipeline = None
        self.vae = None
        self.predictor = None
        self.toy: Optional[ToyDynamics] = None
        self._anchor_generator = None  # frozen copy for L_anchor (full backend)

        if backend == "toy":
            self.toy = ToyDynamics().to(self.device)
            if toy_ckpt and os.path.exists(toy_ckpt):
                ck = torch.load(toy_ckpt, map_location="cpu", weights_only=False)
                self.toy.load_state_dict(ck["toy"])
            self.toy.eval()
            for p in self.toy.parameters():
                p.requires_grad_(False)
        elif backend == "full":
            assert bundle is not None, "full backend requires Stage1Bundle"
            self._load_full(bundle)
        else:
            raise ValueError(f"unknown backend: {backend}")

    def _load_full(self, bundle: Stage1Bundle):
        _ensure_paths()
        from model_loader import load_models

        root = _mg2_root()
        os.chdir(root)
        models = load_models(
            config_path=bundle.config_path,
            checkpoint_path=bundle.generator_ckpt,
            pretrained_model_path=bundle.pretrained_model_path,
            device=str(self.device),
            dtype=str(self.dtype).replace("torch.", ""),
            load_vae_decoder=False,
        )
        self.pipeline = models.pipeline
        self.vae = models.vae
        for p in self.pipeline.generator.parameters():
            p.requires_grad_(False)
        self.pipeline.eval()

        if bundle.predictor_ckpt and os.path.exists(bundle.predictor_ckpt):
            from features import DriftPredictor
            ck = torch.load(bundle.predictor_ckpt, map_location="cpu", weights_only=False)
            dual = ck.get("dual_head", True)
            m = DriftPredictor(in_dim=ck["feature_dim"], hidden=ck.get("hidden", 128), dual_head=dual)
            try:
                m.load_state_dict(ck["predictor"])
            except RuntimeError:
                m = DriftPredictor(in_dim=ck["feature_dim"], hidden=ck.get("hidden", 128), dual_head=False)
                m.load_state_dict(ck["predictor"], strict=False)
            self.predictor = m.to(self.device).eval()

    @torch.no_grad()
    def step(
        self,
        latent_t: torch.Tensor,
        keyboard: torch.Tensor,
        mouse: torch.Tensor,
    ) -> torch.Tensor:
        """One imagination step: (z_t, a_t) -> z_{t+1}. Shapes match block latents."""
        if latent_t.ndim == 4:
            latent_t = latent_t.unsqueeze(0)
        if keyboard.ndim == 2:
            keyboard = keyboard.unsqueeze(0)
        if mouse.ndim == 2:
            mouse = mouse.unsqueeze(0)
        latent_t = latent_t.to(self.device, self.dtype if self.backend == "full" else torch.float32)
        keyboard = keyboard.to(self.device, torch.float32)
        mouse = mouse.to(self.device, torch.float32)

        if self.backend == "toy":
            return self.toy(latent_t.float(), keyboard, mouse)

        # full: use ActionRolloutPipeline conditioned on actions, seed with latent_t as init
        return self._full_step(latent_t, keyboard, mouse)

    def _full_step(self, latent_t, keyboard, mouse):
        """Autoregressive few-step rollout for the next block given action window."""
        _ensure_paths()
        from train.pipeline.action_rollout import ActionRolloutPipeline

        B, C, f, H, W = latent_t.shape
        # Build a minimal cond dict; first-frame cond from latent_t itself
        # (visual_context / cond_concat ideally from VAE; for dynamics fine-tune we
        # reuse zeros-safe placeholders when not provided — caller should pass richer
        # cond via imagine_rollout).
        raise RuntimeError(
            "FrozenWorldModel.step(full) requires imagine_rollout with full conditional_dict; "
            "use imagine_from_sample() for production imagination."
        )

    @torch.no_grad()
    def imagine_from_sample(
        self,
        sample: Dict[str, torch.Tensor],
        num_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """Roll out WM on a dataset sample's actions; return pred latent [1,16,F,H,W]."""
        if self.backend == "toy":
            z = sample["latent"]
            if z.ndim == 4:
                z = z.unsqueeze(0)
            F_lat = z.shape[2] if num_frames is None else num_frames
            nfpb = 3
            out = z[:, :, :nfpb].clone().to(self.device, torch.float32)
            kb = sample["keyboard"]
            mouse = sample["mouse"]
            if kb.ndim == 2:
                kb = kb.unsqueeze(0)
            if mouse.ndim == 2:
                mouse = mouse.unsqueeze(0)
            cur = out
            blocks = [cur]
            for b in range(1, F_lat // nfpb):
                a0 = 4 * (b - 1) * nfpb
                a1 = 4 * (b * nfpb - 1) + 1
                nxt = self.toy(cur, kb[:, a0:a1], mouse[:, a0:a1])
                blocks.append(nxt)
                cur = nxt
            return torch.cat(blocks, dim=2)

        _ensure_paths()
        from train.models.action_wrapper import ActionWanDiffusionWrapper
        from train.pipeline.action_rollout import ActionRolloutPipeline
        from omegaconf import OmegaConf

        # Prefer pipeline.generator already loaded
        gen = self.pipeline.generator
        cfg = OmegaConf.create({
            "denoising_step_list": [1000, 750, 500, 250],
            "warp_denoising_step": True,
            "context_noise": 0,
            "mode": "universal",
            "num_frame_per_block": 3,
        })
        # Wrap if needed — CausalInferencePipeline.generator is WanDiffusionWrapper
        rollout = ActionRolloutPipeline(cfg, gen)
        latent = sample["latent"]
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)
        F_lat = latent.shape[2] if num_frames is None else num_frames
        noise = torch.randn(
            [1, 16, F_lat, latent.shape[-2], latent.shape[-1]],
            device=self.device, dtype=self.dtype,
        )
        cond = {
            "cond_concat": sample["cond_concat"].unsqueeze(0).to(self.device, self.dtype)
            if sample["cond_concat"].ndim == 4 else sample["cond_concat"].to(self.device, self.dtype),
            "visual_context": sample["visual_context"].unsqueeze(0).to(self.device, self.dtype)
            if sample["visual_context"].ndim == 2 else sample["visual_context"].to(self.device, self.dtype),
            "mouse_cond": sample["mouse"].unsqueeze(0).to(self.device, self.dtype)
            if sample["mouse"].ndim == 2 else sample["mouse"].to(self.device, self.dtype),
            "keyboard_cond": sample["keyboard"].unsqueeze(0).to(self.device, self.dtype)
            if sample["keyboard"].ndim == 2 else sample["keyboard"].to(self.device, self.dtype),
        }
        pred, _, _ = rollout.rollout(noise, cond)
        return pred

    def uncertainty(self, feat: torch.Tensor) -> Optional[torch.Tensor]:
        """DriftPredictor confidence if available. Higher → more uncertain."""
        if self.predictor is None:
            return None
        with torch.no_grad():
            out = self.predictor(feat.to(self.device))
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

    def trainable_copy_toy(self) -> ToyDynamics:
        """Return an unfrozen copy of toy dynamics for Stage-3 adaptation."""
        assert self.backend == "toy" and self.toy is not None
        m = ToyDynamics().to(self.device)
        m.load_state_dict(self.toy.state_dict())
        for p in m.parameters():
            p.requires_grad_(True)
        return m

    def freeze_anchor_state_dict(self) -> Dict[str, torch.Tensor]:
        """Snapshot φ0 for L_anchor."""
        if self.backend == "toy":
            return {k: v.detach().cpu().clone() for k, v in self.toy.state_dict().items()}
        return {k: v.detach().cpu().clone() for k, v in self.pipeline.generator.state_dict().items()}
