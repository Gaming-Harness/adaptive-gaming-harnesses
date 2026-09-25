#!/usr/bin/env python3
"""Pretrain toy dynamics on expert .pt so Stage 2/3 have a usable F_φ0.

Usage:
  python -m curriculum.pretrain_toy_wm --data_root data/mc_vpt_long45 --out curriculum/outputs/toy_wm0.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from curriculum.frozen_wm import ToyDynamics
from curriculum.replay_buffer import MixedReplayBuffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mc_vpt_long45")
    ap.add_argument("--out", default="curriculum/outputs/toy_wm0.pt")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_clips", type=int, default=None)
    args = ap.parse_args()
    os.chdir(_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buf = MixedReplayBuffer()
    n = buf.load_expert_pt_dir(args.data_root, max_clips=args.max_clips, block_frames=3, stride=2)
    print(f"[pretrain_toy] {n} transitions")

    model = ToyDynamics().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        batch = buf.sample(args.batch_size, prefer="expert")
        z = torch.stack([t.latent_t for t in batch]).to(device).float()
        z1 = torch.stack([t.latent_tp1 for t in batch]).to(device).float()
        kbs, mss = [], []
        for t in batch:
            kb, ms = t.keyboard.float(), t.mouse.float()
            kbs.append(kb.mean(0) if kb.ndim == 2 else kb)
            mss.append(ms.mean(0) if ms.ndim == 2 else ms)
        kb = torch.stack(kbs).to(device).float()
        ms = torch.stack(mss).to(device).float()
        pred = model(z, kb, ms)
        loss = F.mse_loss(pred.float(), z1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f"  step {step} loss {loss.item():.6f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"toy": model.state_dict(), "step": args.steps}, args.out)
    print(f"[pretrain_toy] wrote {args.out}")


if __name__ == "__main__":
    main()
