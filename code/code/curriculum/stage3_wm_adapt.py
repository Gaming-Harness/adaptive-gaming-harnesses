#!/usr/bin/env python3
"""Stage 3: Policy-aware world-model adaptation with anchor loss.

After π shifts the visited state distribution p_π(s) away from p_data(s),
adapt F with small steps:

  L = L_dyn + λ_anchor * L_anchor + λ_expert * L_expert

where
  L_dyn    = ||F_φ(s,a) - s'||² on policy (+ expert mix) transitions
  L_anchor = ||F_φ(s,a) - F_φ0(s,a)||²   (continual-learning style)
  L_expert = dynamics loss on expert replay (prevents combat-only collapse)

Do NOT full fine-tune from scratch on policy-only data.

Usage:
  python -m curriculum.stage3_wm_adapt --config curriculum/configs/stage3_wm_adapt.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from curriculum.frozen_wm import FrozenWorldModel, Stage1Bundle, ToyDynamics
from curriculum.policy import LatentMLPPolicy
from curriculum.replay_buffer import MixedReplayBuffer, Transition


class Stage3WMAdapter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if cfg.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.nfpb = int(cfg.num_frame_per_block)
        self.lambda_anchor = float(cfg.lambda_anchor)
        self.lambda_expert = float(cfg.lambda_expert)
        self.expert_ratio = float(cfg.expert_ratio)

        backend = cfg.wm_backend
        if backend != "toy":
            raise NotImplementedError(
                "Stage-3 full-F adaptation hooks into ActionConsistencyTrainer; "
                "use wm_backend=toy for the curriculum loop, or set "
                "cfg.use_consistency_finetune=true to shell out to train.train."
            )

        # Load frozen teacher φ0
        self.teacher = FrozenWorldModel(
            backend="toy",
            device=str(self.device),
            toy_ckpt=cfg.get("toy_wm_ckpt", None),
        )
        # Student φ1 starts from φ0
        self.student: ToyDynamics = self.teacher.trainable_copy_toy()
        self.opt = torch.optim.AdamW(
            self.student.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )

        # Optional policy to collect new on-policy transitions
        self.policy = LatentMLPPolicy(in_dim=16, hidden=int(cfg.get("policy_hidden", 256))).to(self.device)
        if cfg.get("policy_ckpt") and os.path.exists(cfg.policy_ckpt):
            self.policy.load(cfg.policy_ckpt, map_location=str(self.device))
            print(f"[stage3] loaded policy {cfg.policy_ckpt}")
        self.policy.eval()

        self.buf = MixedReplayBuffer(expert_ratio=self.expert_ratio)
        n = self.buf.load_expert_pt_dir(
            cfg.data_root,
            max_clips=cfg.get("max_clips", None),
            block_frames=self.nfpb,
            stride=int(cfg.get("block_stride", 1)),
        )
        print(f"[stage3] expert transitions: {n}")

        self.logdir = cfg.logdir
        os.makedirs(self.logdir, exist_ok=True)
        self.step = 0

    @torch.no_grad()
    def collect_policy_transitions(self, n: int) -> int:
        """Roll π on expert states; next state = teacher imagination (proxy for sandbox).

        In production, replace teacher.step with sandbox.step for true (s,a,s*).
        """
        added = 0
        for _ in range(n):
            tr = self.buf.sample(1, prefer="expert")[0]
            z = tr.latent_t.unsqueeze(0).to(self.device).float()
            act = self.policy.act(z, deterministic=False)
            T_a = tr.keyboard.shape[0]
            kb_seq = act.keyboard.unsqueeze(1).expand(1, T_a, -1)
            ms_seq = act.mouse.unsqueeze(1).expand(1, T_a, -1)
            # Prefer real next if we only changed action distribution slightly:
            # here we use teacher as stand-in for sandbox dynamics under new a
            z_next = self.teacher.step(z, kb_seq, ms_seq)
            self.buf.add_policy(Transition(
                latent_t=tr.latent_t,
                keyboard=act.keyboard.squeeze(0).cpu(),
                mouse=act.mouse.squeeze(0).cpu(),
                latent_tp1=z_next.squeeze(0).cpu(),
                reward=0.0,
                source="policy",
            ))
            added += 1
        return added

    def _dyn_loss(self, model: ToyDynamics, batch) -> torch.Tensor:
        z = torch.stack([t.latent_t for t in batch]).to(self.device).float()
        z1 = torch.stack([t.latent_tp1 for t in batch]).to(self.device).float()
        kbs, mss = [], []
        for t in batch:
            kb, ms = t.keyboard.float(), t.mouse.float()
            if kb.ndim == 1:
                kb = kb.unsqueeze(0)
            if ms.ndim == 1:
                ms = ms.unsqueeze(0)
            # pad/truncate to same T — mean pool for toy
            kbs.append(kb.mean(0) if kb.ndim == 2 else kb)
            mss.append(ms.mean(0) if ms.ndim == 2 else ms)
        kb = torch.stack(kbs).to(self.device)
        ms = torch.stack(mss).to(self.device)
        # expand to [B,1,d] for toy.step interface
        pred = model(z, kb, ms)
        return F.mse_loss(pred.float(), z1.float())

    def train_step(self) -> Dict[str, float]:
        self.collect_policy_transitions(int(self.cfg.collect_policy))
        B = int(self.cfg.batch_size)

        # Policy / mixed dynamics
        mix = self.buf.sample(B)  # respects expert_ratio
        loss_dyn = self._dyn_loss(self.student, mix)

        # Pure expert dynamics (anti-collapse)
        expert_batch = self.buf.sample(B, prefer="expert")
        loss_expert = self._dyn_loss(self.student, expert_batch)

        # Anchor: student vs frozen teacher on same (s,a)
        with torch.no_grad():
            # teacher predictions
            z = torch.stack([t.latent_t for t in mix]).to(self.device).float()
            kbs, mss = [], []
            for t in mix:
                kb, ms = t.keyboard.float(), t.mouse.float()
                kbs.append(kb.mean(0) if kb.ndim == 2 else kb)
                mss.append(ms.mean(0) if ms.ndim == 2 else ms)
            kb = torch.stack(kbs).to(self.device).float()
            ms = torch.stack(mss).to(self.device).float()
            teacher_pred = self.teacher.toy(z, kb, ms)

        student_pred = self.student(z, kb, ms)
        loss_anchor = F.mse_loss(student_pred.float(), teacher_pred.float())

        loss = loss_dyn + self.lambda_anchor * loss_anchor + self.lambda_expert * loss_expert
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), float(self.cfg.max_grad_norm))
        self.opt.step()
        return {
            "loss": float(loss.item()),
            "L_dyn": float(loss_dyn.item()),
            "L_anchor": float(loss_anchor.item()),
            "L_expert": float(loss_expert.item()),
            "buf_policy": len(self.buf.policy),
        }

    def train(self):
        history = []
        t0 = time.time()
        for self.step in range(1, int(self.cfg.max_steps) + 1):
            stats = self.train_step()
            if self.step % int(self.cfg.log_interval) == 0:
                print(
                    f"[stage3] step {self.step:5d} | loss {stats['loss']:.4f} "
                    f"| dyn {stats['L_dyn']:.4f} | anc {stats['L_anchor']:.4f} "
                    f"| exp {stats['L_expert']:.4f} | pol_buf {stats['buf_policy']}",
                    flush=True,
                )
                history.append({"step": self.step, **stats})
            if self.step % int(self.cfg.save_interval) == 0:
                self.save()
        self.save()
        # Promote student → new frozen teacher checkpoint for Stage-2/4
        with open(os.path.join(self.logdir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"[stage3] done in {time.time()-t0:.1f}s")

    def save(self):
        path = os.path.join(self.logdir, f"wm_adapt_step_{self.step:06d}.pt")
        payload = {
            "toy": self.student.state_dict(),
            "step": self.step,
            "lambda_anchor": self.lambda_anchor,
            "backend": "toy",
        }
        torch.save(payload, path)
        torch.save(payload, os.path.join(self.logdir, "wm_adapt_latest.pt"))
        print(f"[stage3] saved {path}", flush=True)


def maybe_launch_consistency_finetune(cfg) -> None:
    """Optional: shell out to existing ActionConsistencyTrainer for full F."""
    if not cfg.get("use_consistency_finetune", False):
        return
    import subprocess
    cmd = [
        sys.executable, "-m", "train.train",
        "--config_path", cfg.consistency_config,
        "--logdir", cfg.consistency_logdir,
    ]
    print("[stage3] launching consistency finetune:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="curriculum/configs/stage3_wm_adapt.yaml")
    args = ap.parse_args()
    os.chdir(_ROOT)
    cfg = OmegaConf.load(args.config)
    if cfg.get("use_consistency_finetune", False) and cfg.wm_backend == "full":
        maybe_launch_consistency_finetune(cfg)
        return
    Stage3WMAdapter(cfg).train()


if __name__ == "__main__":
    main()
