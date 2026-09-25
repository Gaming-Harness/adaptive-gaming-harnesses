#!/usr/bin/env python3
"""Stage 4: Closed-loop alternating WM ↔ VLA (Dreamer / MuZero style).

Outer loop:
  1. VLA interacts (real + imagination under frozen/current WM)
  2. Update WM on real + replay (policy-aware, with L_anchor)
  3. WM generates imagination
  4. Improve VLA with RL
  5. Evaluate reality gap

Usage:
  python -m curriculum.stage4_closed_loop --config curriculum/configs/stage4_closed_loop.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict, List

import torch
from omegaconf import OmegaConf

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from curriculum.frozen_wm import FrozenWorldModel
from curriculum.policy import LatentMLPPolicy
from curriculum.replay_buffer import MixedReplayBuffer
from curriculum.rewards import dynamics_consistency_reward, sharpness_preserve_reward
from curriculum.stage2_vla_rl import Stage2Trainer
from curriculum.stage3_wm_adapt import Stage3WMAdapter


def reality_gap(wm: FrozenWorldModel, buf: MixedReplayBuffer, n: int = 32) -> float:
    """Mean ||F(s,a) - s*|| on expert transitions (lower is better)."""
    if len(buf.expert) == 0:
        return float("nan")
    batch = buf.sample(min(n, len(buf.expert)), prefer="expert")
    errs = []
    device = wm.device
    with torch.no_grad():
        for t in batch:
            z = t.latent_t.unsqueeze(0).to(device)
            kb = t.keyboard.float()
            ms = t.mouse.float()
            # reduce action window -> [4] / [2], then batch dim
            if kb.ndim == 2:
                kb = kb.mean(0)
            if ms.ndim == 2:
                ms = ms.mean(0)
            kb_m = kb.view(1, -1).to(device)
            ms_m = ms.view(1, -1).to(device)
            if wm.backend == "toy":
                pred = wm.toy(z.float(), kb_m, ms_m)
            else:
                pred = wm.step(z, kb_m.unsqueeze(1), ms_m.unsqueeze(1))
            target = t.latent_tp1.unsqueeze(0).to(device)
            errs.append(float((pred.float() - target.float()).pow(2).mean().item()))
    return float(sum(errs) / max(1, len(errs)))


class ClosedLoopRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logdir = cfg.logdir
        os.makedirs(self.logdir, exist_ok=True)
        self.history: List[Dict] = []

        # Shared toy WM path that stages overwrite
        self.wm_ckpt = os.path.join(self.logdir, "shared_wm.pt")
        self.policy_ckpt = os.path.join(self.logdir, "shared_policy.pt")

        # Initialize empty toy WM checkpoint if missing
        if not os.path.exists(self.wm_ckpt):
            wm0 = FrozenWorldModel(backend="toy", device="cpu")
            torch.save({"toy": wm0.toy.state_dict(), "step": 0}, self.wm_ckpt)

    def _stage2_cfg(self, round_i: int):
        c = OmegaConf.create(OmegaConf.to_container(self.cfg.stage2, resolve=True))
        c.toy_wm_ckpt = self.wm_ckpt
        c.policy_ckpt = self.policy_ckpt if os.path.exists(self.policy_ckpt) else None
        c.logdir = os.path.join(self.logdir, f"round{round_i:02d}_stage2")
        # gradually allow more imagination across outer rounds
        r0 = float(self.cfg.stage2.real_ratio)
        r1 = float(self.cfg.stage2.get("real_ratio_end", r0))
        R = max(1, int(self.cfg.num_rounds) - 1)
        c.real_ratio = r0 + (r1 - r0) * (round_i / R)
        c.real_ratio_end = c.real_ratio  # hold within round
        return c

    def _stage3_cfg(self, round_i: int):
        c = OmegaConf.create(OmegaConf.to_container(self.cfg.stage3, resolve=True))
        c.toy_wm_ckpt = self.wm_ckpt
        c.policy_ckpt = self.policy_ckpt
        c.logdir = os.path.join(self.logdir, f"round{round_i:02d}_stage3")
        return c

    def run(self):
        t0 = time.time()
        for r in range(int(self.cfg.num_rounds)):
            print(f"\n========== outer round {r+1}/{self.cfg.num_rounds} ==========", flush=True)

            # 1–2. Interact + improve VLA with frozen WM
            s2 = Stage2Trainer(self._stage2_cfg(r))
            s2.train()
            # publish policy
            src = os.path.join(s2.logdir, "policy_latest.pt")
            shutil.copy2(src, self.policy_ckpt)

            gap_before = reality_gap(s2.wm, s2.buf, n=int(self.cfg.eval_n))
            print(f"[stage4] reality_gap before WM adapt: {gap_before:.6f}")

            # 3. Policy-aware WM adaptation (anchor to previous WM)
            s3 = Stage3WMAdapter(self._stage3_cfg(r))
            s3.train()
            src_wm = os.path.join(s3.logdir, "wm_adapt_latest.pt")
            shutil.copy2(src_wm, self.wm_ckpt)

            # 4. Re-measure gap with updated WM
            wm_new = FrozenWorldModel(backend="toy", device=str(s2.device), toy_ckpt=self.wm_ckpt)
            gap_after = reality_gap(wm_new, s2.buf, n=int(self.cfg.eval_n))
            print(f"[stage4] reality_gap after  WM adapt: {gap_after:.6f}")

            rec = {
                "round": r,
                "reality_gap_before": gap_before,
                "reality_gap_after": gap_after,
                "policy_ckpt": self.policy_ckpt,
                "wm_ckpt": self.wm_ckpt,
            }
            self.history.append(rec)
            with open(os.path.join(self.logdir, "closed_loop_history.json"), "w") as f:
                json.dump(self.history, f, indent=2)

        print(f"\n[stage4] finished {self.cfg.num_rounds} rounds in {time.time()-t0:.1f}s")
        print(f"[stage4] artifacts in {self.logdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="curriculum/configs/stage4_closed_loop.yaml")
    args = ap.parse_args()
    os.chdir(_ROOT)
    cfg = OmegaConf.load(args.config)
    ClosedLoopRunner(cfg).run()


if __name__ == "__main__":
    main()
