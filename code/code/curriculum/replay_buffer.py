"""Mixed replay: expert (sandbox / VPT) + policy trajectories.

Prevents Stage-3 WM collapse onto early bad π distributions.
"""
from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch


@dataclass
class Transition:
    """One (s_t, a_t, s_{t+1}, r_t, done) in latent space."""
    latent_t: torch.Tensor       # [16, f, H, W] or [16, H, W]
    keyboard: torch.Tensor       # [T_a, 4] or [4]  (hard/fallback pretrained_wm)
    mouse: torch.Tensor          # [T_a, 2] or [2]
    latent_tp1: torch.Tensor
    reward: float = 0.0
    done: bool = False
    source: str = "expert"       # expert | policy | imagined
    # VLA MineStudio flat feature [D]; when set, update_wm trains ActionProjection via L_dyn.
    action_feat: Optional[torch.Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class MixedReplayBuffer:
    def __init__(
        self,
        expert_ratio: float = 0.5,
        capacity_policy: int = 50_000,
        capacity_imagined: int = 20_000,
    ):
        self.expert_ratio = float(expert_ratio)
        self.capacity_policy = int(capacity_policy)
        self.capacity_imagined = int(capacity_imagined)
        self.expert: List[Transition] = []
        self.policy: List[Transition] = []
        self.imagined: List[Transition] = []

    def __len__(self) -> int:
        return len(self.expert) + len(self.policy) + len(self.imagined)

    def add_expert(self, tr: Transition) -> None:
        tr.source = "expert"
        self.expert.append(tr)

    def add_policy(self, tr: Transition) -> None:
        tr.source = "policy"
        self.policy.append(tr)
        if len(self.policy) > self.capacity_policy:
            self.policy = self.policy[-self.capacity_policy:]

    def add_imagined(self, tr: Transition) -> None:
        tr.source = "imagined"
        self.imagined.append(tr)
        if len(self.imagined) > self.capacity_imagined:
            self.imagined = self.imagined[-self.capacity_imagined:]

    def load_expert_pt_dir(
        self,
        data_root: str,
        max_clips: Optional[int] = None,
        block_frames: int = 3,
        stride: int = 1,
    ) -> int:
        """Convert ActionVideoDataset .pt clips into block transitions."""
        files = sorted(glob.glob(os.path.join(data_root, "*.pt")))
        if max_clips is not None:
            files = files[:max_clips]
        n = 0
        for path in files:
            d = torch.load(path, map_location="cpu", weights_only=False)
            n += self._ingest_clip(d, source="expert", block_frames=block_frames, stride=stride)
        return n

    def _ingest_clip(
        self,
        d: Dict[str, Any],
        source: str,
        block_frames: int = 3,
        stride: int = 1,
    ) -> int:
        latent = d["latent"]           # [16, F, H, W]
        kb = d["keyboard"]             # [4*(F-1)+1, 4]
        mouse = d["mouse"]             # [4*(F-1)+1, 2]
        F = latent.shape[1]
        assert F >= 2 * block_frames
        n = 0
        for b0 in range(0, F - 2 * block_frames + 1, stride * block_frames):
            b1 = b0 + block_frames
            b2 = b1 + block_frames
            # action window covering transition from block b0 -> b1
            a0 = 4 * b0
            a1 = 4 * (b1 - 1) + 1
            tr = Transition(
                latent_t=latent[:, b0:b1].contiguous(),
                keyboard=kb[a0:a1].contiguous(),
                mouse=mouse[a0:a1].contiguous(),
                latent_tp1=latent[:, b1:b2].contiguous(),
                reward=0.0,
                done=False,
                source=source,
                meta={"F": F, "b0": b0},
            )
            if source == "expert":
                self.add_expert(tr)
            elif source == "policy":
                self.add_policy(tr)
            else:
                self.add_imagined(tr)
            n += 1
        return n

    def sample(self, batch_size: int, prefer: Optional[str] = None) -> List[Transition]:
        """Sample mixed batch. prefer=None uses expert_ratio on (expert vs policy)."""
        out: List[Transition] = []
        for _ in range(batch_size):
            if prefer == "expert" and self.expert:
                out.append(random.choice(self.expert))
            elif prefer == "policy" and self.policy:
                out.append(random.choice(self.policy))
            elif prefer == "imagined" and self.imagined:
                out.append(random.choice(self.imagined))
            else:
                use_expert = random.random() < self.expert_ratio
                if use_expert and self.expert:
                    out.append(random.choice(self.expert))
                elif self.policy:
                    out.append(random.choice(self.policy))
                elif self.expert:
                    out.append(random.choice(self.expert))
                elif self.imagined:
                    out.append(random.choice(self.imagined))
                else:
                    raise RuntimeError("MixedReplayBuffer is empty")
        return out

    def sample_real_imag_mix(
        self,
        batch_size: int,
        real_ratio: float,
    ) -> List[Transition]:
        """Stage-2 mix: real (expert+policy) vs imagined."""
        out: List[Transition] = []
        real_pool = self.expert + self.policy
        for _ in range(batch_size):
            if random.random() < real_ratio and real_pool:
                out.append(random.choice(real_pool))
            elif self.imagined:
                out.append(random.choice(self.imagined))
            elif real_pool:
                out.append(random.choice(real_pool))
            else:
                raise RuntimeError("no data for real/imag mix")
        return out
