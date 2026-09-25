#!/usr/bin/env python3
"""Stage 2: Frozen WM + Online RL for VLA / policy.

Critical constraint: F_φ0 is frozen. Only π_θ is updated.

Data mix (default):
  real_ratio = 0.8  (expert/sandbox + policy real)
  imag_ratio = 0.2  (WM imagination)
gradually increase imagination via imag_ratio_schedule.

Usage (toy smoke, no 1.3B load):
  python -m curriculum.stage2_vla_rl --config curriculum/configs/stage2_frozen_wm_vla.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from curriculum.frozen_wm import FrozenWorldModel, Stage1Bundle
from curriculum.policy import LatentMLPPolicy, ppo_update
from curriculum.replay_buffer import MixedReplayBuffer, Transition
from curriculum.rewards import (
    combine_rewards,
    dynamics_consistency_reward,
    sharpness_preserve_reward,
)


def _cfg_get(cfg, key, default=None):
    return OmegaConf.select(cfg, key, default=default)


class Stage2Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if cfg.device != "auto"
                                    else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.real_ratio = float(cfg.real_ratio)
        self.gamma = float(cfg.gamma)
        self.nfpb = int(cfg.num_frame_per_block)

        # --- Frozen WM teacher ---
        backend = cfg.wm_backend
        bundle = None
        if backend == "full":
            bundle = Stage1Bundle.load(cfg.stage1_bundle)
        self.wm = FrozenWorldModel(
            backend=backend,
            bundle=bundle,
            device=str(self.device),
            dtype=str(cfg.get("dtype", "bfloat16")),
            toy_ckpt=cfg.get("toy_wm_ckpt", None),
        )

        # --- Policy ---
        self.policy = LatentMLPPolicy(
            in_dim=16,
            hidden=int(cfg.policy_hidden),
        ).to(self.device)
        if cfg.get("policy_ckpt") and os.path.exists(cfg.policy_ckpt):
            self.policy.load(cfg.policy_ckpt, map_location=str(self.device))
        self.opt = torch.optim.AdamW(
            self.policy.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )

        # --- Replay ---
        self.buf = MixedReplayBuffer(
            expert_ratio=float(cfg.expert_ratio),
            capacity_policy=int(cfg.capacity_policy),
            capacity_imagined=int(cfg.capacity_imagined),
        )
        n = self.buf.load_expert_pt_dir(
            cfg.data_root,
            max_clips=cfg.get("max_clips", None),
            block_frames=self.nfpb,
            stride=int(cfg.get("block_stride", 1)),
        )
        print(f"[stage2] loaded {n} expert block transitions from {cfg.data_root}")

        self.step = 0
        self.logdir = cfg.logdir
        os.makedirs(self.logdir, exist_ok=True)

    def _imagination_ratio(self) -> float:
        """Schedule: start high-real, gradually allow more imagination."""
        start = float(self.cfg.real_ratio)
        end = float(self.cfg.get("real_ratio_end", start))
        T = max(1, int(self.cfg.max_steps))
        t = min(self.step, T) / T
        return start + (end - start) * t

    def _reward_for_transition(
        self,
        z_t: torch.Tensor,
        z_tp1_true: torch.Tensor,
        z_tp1_pred: Optional[torch.Tensor],
    ) -> float:
        parts = [sharpness_preserve_reward(z_t, z_tp1_true)]
        weights = [1.0]
        if z_tp1_pred is not None:
            parts.append(dynamics_consistency_reward(z_tp1_pred, z_tp1_true))
            weights.append(float(self.cfg.get("dyn_reward_weight", 0.5)))
        r = combine_rewards(*parts, weights=weights)
        return float(r.mean().item())

    @torch.no_grad()
    def collect_imagined(self, n: int) -> int:
        """Use frozen WM to generate imagined (s,a,s') under current π."""
        if len(self.buf.expert) == 0:
            return 0
        added = 0
        self.policy.eval()
        for _ in range(n):
            tr = self.buf.sample(1, prefer="expert")[0]
            z = tr.latent_t.unsqueeze(0).to(self.device).float()
            act = self.policy.act(z, deterministic=False)
            # expand action to action-window length expected by toy/full
            T_a = tr.keyboard.shape[0]
            kb = act.keyboard  # [1,4]
            ms = act.mouse     # [1,2]
            kb_seq = kb.unsqueeze(1).expand(1, T_a, -1).contiguous()
            ms_seq = ms.unsqueeze(1).expand(1, T_a, -1).contiguous()
            z_hat = self.wm.step(z, kb_seq, ms_seq)
            r = self._reward_for_transition(
                z, tr.latent_tp1.unsqueeze(0).to(self.device).float(), z_hat
            )
            self.buf.add_imagined(Transition(
                latent_t=tr.latent_t,
                keyboard=kb.squeeze(0).cpu(),
                mouse=ms.squeeze(0).cpu(),
                latent_tp1=z_hat.squeeze(0).cpu(),
                reward=r,
                source="imagined",
            ))
            added += 1
        return added

    @torch.no_grad()
    def collect_policy_on_expert_states(self, n: int) -> int:
        """On-policy actions on expert states; next-state from expert (real) or WM."""
        added = 0
        self.policy.eval()
        for _ in range(n):
            tr = self.buf.sample(1, prefer="expert")[0]
            z = tr.latent_t.unsqueeze(0).to(self.device).float()
            act = self.policy.act(z, deterministic=False)
            # real next from expert (behavior cloning style on-policy eval reward)
            z_next = tr.latent_tp1.unsqueeze(0).to(self.device).float()
            r = self._reward_for_transition(z, z_next, None)
            self.buf.add_policy(Transition(
                latent_t=tr.latent_t,
                keyboard=act.keyboard.squeeze(0).cpu(),
                mouse=act.mouse.squeeze(0).cpu(),
                latent_tp1=tr.latent_tp1,
                reward=r,
                source="policy",
                meta={"old_log_prob": float(act.log_prob.item()),
                      "value": float(act.value.item())},
            ))
            added += 1
        return added

    def _batch_tensors(self, transitions: List[Transition]):
        z = torch.stack([t.latent_t for t in transitions]).to(self.device).float()
        # actions may be [4] or [T,4] — take mean for MLP policy
        kbs, mss, rewards, old_lps, values = [], [], [], [], []
        for t in transitions:
            kb = t.keyboard.float()
            ms = t.mouse.float()
            if kb.ndim == 2:
                kb = kb.mean(0)
            if ms.ndim == 2:
                ms = ms.mean(0)
            kbs.append(kb)
            mss.append(ms)
            rewards.append(t.reward)
            old_lps.append(float(t.meta.get("old_log_prob", 0.0)))
            values.append(float(t.meta.get("value", 0.0)))
        kb = torch.stack(kbs).to(self.device)
        ms = torch.stack(mss).to(self.device)
        rew = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        old_lp = torch.tensor(old_lps, device=self.device, dtype=torch.float32)
        old_v = torch.tensor(values, device=self.device, dtype=torch.float32)
        # bootstrap: V(s') ≈ 0 for block transitions (short horizon)
        returns = rew
        advantages = returns - old_v
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return z, kb, ms, old_lp, returns, advantages

    def train_step(self) -> Dict[str, float]:
        real_ratio = self._imagination_ratio()
        # refresh buffers
        self.collect_policy_on_expert_states(int(self.cfg.collect_policy))
        self.collect_imagined(int(self.cfg.collect_imagined))

        batch = self.buf.sample_real_imag_mix(
            int(self.cfg.batch_size),
            real_ratio=real_ratio,
        )
        # ensure old_log_prob present
        self.policy.eval()
        with torch.no_grad():
            for t in batch:
                if "old_log_prob" not in t.meta:
                    z = t.latent_t.unsqueeze(0).to(self.device).float()
                    kb = t.keyboard.float()
                    ms = t.mouse.float()
                    if kb.ndim == 2:
                        kb = kb.mean(0)
                    if ms.ndim == 2:
                        ms = ms.mean(0)
                    res = self.policy.evaluate_actions(
                        z, kb.unsqueeze(0).to(self.device), ms.unsqueeze(0).to(self.device)
                    )
                    t.meta["old_log_prob"] = float(res.log_prob.item())
                    t.meta["value"] = float(res.value.item())

        z, kb, ms, old_lp, returns, adv = self._batch_tensors(batch)
        self.policy.train()
        stats = ppo_update(
            self.policy, self.opt, z, kb, ms, old_lp, returns, adv,
            clip_eps=float(self.cfg.clip_eps),
            vf_coef=float(self.cfg.vf_coef),
            ent_coef=float(self.cfg.ent_coef),
        )
        stats["real_ratio"] = real_ratio
        stats["buf_expert"] = len(self.buf.expert)
        stats["buf_policy"] = len(self.buf.policy)
        stats["buf_imagined"] = len(self.buf.imagined)
        return stats

    def train(self):
        t0 = time.time()
        history = []
        for self.step in range(1, int(self.cfg.max_steps) + 1):
            stats = self.train_step()
            if self.step % int(self.cfg.log_interval) == 0:
                msg = (f"[stage2] step {self.step:5d} | loss {stats['loss']:.4f} "
                       f"| pi {stats['policy_loss']:.4f} | v {stats['value_loss']:.4f} "
                       f"| H {stats['entropy']:.3f} | real_ratio {stats['real_ratio']:.2f} "
                       f"| buf E/P/I {stats['buf_expert']}/{stats['buf_policy']}/{stats['buf_imagined']}")
                print(msg, flush=True)
                history.append({"step": self.step, **stats})
            if self.step % int(self.cfg.save_interval) == 0:
                self.save()
        self.save()
        with open(os.path.join(self.logdir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"[stage2] done in {time.time()-t0:.1f}s -> {self.logdir}")

    def save(self):
        path = os.path.join(self.logdir, f"policy_step_{self.step:06d}.pt")
        self.policy.save(path)
        # also write latest
        self.policy.save(os.path.join(self.logdir, "policy_latest.pt"))
        print(f"[stage2] saved {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="curriculum/configs/stage2_frozen_wm_vla.yaml")
    args = ap.parse_args()
    os.chdir(_ROOT)
    cfg = OmegaConf.load(args.config)
    Stage2Trainer(cfg).train()


if __name__ == "__main__":
    main()
