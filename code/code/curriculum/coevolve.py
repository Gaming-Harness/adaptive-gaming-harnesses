#!/usr/bin/env python3
"""Co-Evolution loop (CoMAP-aligned) for visual WM + VLA.

CoMAP (arXiv:2606.02372) co-evolves textual WM and LLM agents via:
  1) draft action → WM future imagination → future-aware reflection
  2) on-policy trajectories → WM self-distillation

We port the *same closed-loop idea* to gaming:
  Sandbox ↔ Visual World Model ↔ VLA

Differences vs CoMAP (our opportunity):
  - visual / action-conditioned video WM (not text state)
  - VLA policy (not pure LLM tool agent)
  - long-horizon stabilization (phase / re-ground / sparse reality)
    sits *inside* the WM as belief stabilization during co-evolution

Warm-up (optional): expert pretrain of F_φ0 and π_θ  → then this loop.

Usage:
  python -m curriculum.coevolve --config curriculum/configs/coevolve.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from curriculum.frozen_wm import FrozenWorldModel, ToyDynamics
from curriculum.policy import LatentMLPPolicy, ppo_update, ActResult
from curriculum.replay_buffer import MixedReplayBuffer, Transition
from curriculum.rewards import (
    combine_rewards,
    dynamics_consistency_reward,
    sharpness_preserve_reward,
)


def _mean_action(kb: torch.Tensor, ms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if kb.ndim == 2:
        kb = kb.mean(0)
    if ms.ndim == 2:
        ms = ms.mean(0)
    return kb.float(), ms.float()


def _batch_actions(transitions: List[Transition], device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zs, kbs, mss = [], [], []
    for t in transitions:
        zs.append(t.latent_t.float())
        kb, ms = _mean_action(t.keyboard, t.mouse)
        kbs.append(kb)
        mss.append(ms)
    return (
        torch.stack(zs).to(device),
        torch.stack(kbs).to(device),
        torch.stack(mss).to(device),
    )


class CoEvolver:
    """Single process that jointly updates φ and θ every outer step."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if cfg.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.nfpb = int(cfg.num_frame_per_block)
        self.logdir = cfg.logdir
        os.makedirs(self.logdir, exist_ok=True)

        # --- World model: student + EMA teacher (CoMAP self-distill) ---
        assert cfg.wm_backend == "toy", (
            "coevolve currently ships with toy backend for the joint loop; "
            "full pretrained_wm F can be swapped in via FrozenWorldModel(backend='full')."
        )
        init_ckpt = cfg.get("toy_wm_ckpt", None)
        teacher_wrap = FrozenWorldModel(
            backend="toy", device=str(self.device), toy_ckpt=init_ckpt
        )
        self.wm_teacher: ToyDynamics = teacher_wrap.toy  # frozen EMA / φ̄
        for p in self.wm_teacher.parameters():
            p.requires_grad_(False)

        self.wm_student: ToyDynamics = ToyDynamics().to(self.device)
        self.wm_student.load_state_dict(self.wm_teacher.state_dict())
        self.wm_opt = torch.optim.AdamW(
            self.wm_student.parameters(),
            lr=float(cfg.wm_lr),
            weight_decay=float(cfg.weight_decay),
        )
        self.ema_tau = float(cfg.ema_tau)

        # --- Policy ---
        self.policy = LatentMLPPolicy(in_dim=16, hidden=int(cfg.policy_hidden)).to(self.device)
        if cfg.get("policy_ckpt") and os.path.exists(cfg.policy_ckpt):
            self.policy.load(cfg.policy_ckpt, map_location=str(self.device))
        self.pi_opt = torch.optim.AdamW(
            self.policy.parameters(),
            lr=float(cfg.pi_lr),
            weight_decay=float(cfg.weight_decay),
        )

        # --- Replay: expert warm-up + on-policy ---
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
        print(f"[coevolve] expert transitions: {n}")

        self.step = 0
        self.history: List[Dict] = []

    # ------------------------------------------------------------------ #
    # CoMAP decision loop: draft → imagine → reflect → (pseudo) execute
    # ------------------------------------------------------------------ #
    def _imagine(self, z: torch.Tensor, kb: torch.Tensor, ms: torch.Tensor) -> torch.Tensor:
        """Use *teacher* for imagination (stable foresight during decision)."""
        with torch.no_grad():
            return self.wm_teacher(z.float(), kb.float(), ms.float())

    def _reliability(self, z: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
        """Cheap reliability proxy ∈ (0,1]: low if sharpness collapses / latent explodes.

        CoMAP estimates reliability of textual future feedback; we use visual
        proxies until DriftPredictor is wired for full backend.
        """
        sharp_drop = torch.relu(
            sharpness_preserve_reward(z, z_hat, scale=-1.0)
        )  # positive drop magnitude
        # map drop → reliability
        rel = torch.exp(-float(self.cfg.reliability_temp) * sharp_drop)
        return rel.clamp(0.05, 1.0)

    def _score_future(self, z: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
        """Higher is better imagined outcome (for reflection gating)."""
        return combine_rewards(
            sharpness_preserve_reward(z, z_hat),
            weights=[1.0],
        )

    @torch.no_grad()
    def decision_step(self, z: torch.Tensor) -> Tuple[ActResult, ActResult, torch.Tensor, float]:
        """
        Returns: draft, reflected, imagined_next, reliability
        """
        self.policy.eval()
        draft = self.policy.act(z, deterministic=False)
        z_hat = self._imagine(z, draft.keyboard, draft.mouse)
        rel = self._reliability(z, z_hat)
        score_draft = self._score_future(z, z_hat)

        # Future-aware reflection: resample; keep better foresight if reliable
        reflected = self.policy.act(z, deterministic=False)
        z_hat_ref = self._imagine(z, reflected.keyboard, reflected.mouse)
        score_ref = self._score_future(z, z_hat_ref)

        # Gate: only replace draft when imagination is reliable AND reflection scores higher
        use_ref = (rel.mean() > float(self.cfg.reflect_rel_thresh)) and (score_ref > score_draft)
        final = reflected if bool(use_ref.item()) else draft
        z_final = z_hat_ref if bool(use_ref.item()) else z_hat
        return draft, final, z_final, float(rel.mean().item())

    @torch.no_grad()
    def interact_batch(self, n: int) -> Dict[str, float]:
        """On-policy interact on expert states (sandbox proxy = teacher→student target mix).

        Production: replace teacher.step with sandbox Env(s,a)→s'.
        Here teacher imagination is the 'environment' stand-in for toy co-evolution,
        while student is trained to match *privileged* next from expert when available,
        else teacher rollout under the executed action (on-policy expansion).
        """
        rels, n_reflect = [], 0
        for _ in range(n):
            tr = self.buf.sample(1, prefer="expert")[0]
            z = tr.latent_t.unsqueeze(0).to(self.device).float()
            draft, final, z_hat, rel = self.decision_step(z)
            rels.append(rel)
            if (final.keyboard - draft.keyboard).abs().sum() > 0 or (
                final.mouse - draft.mouse
            ).abs().sum() > 1e-6:
                n_reflect += 1

            # Privileged next: expert s' if we stayed near expert action, else imagined
            # In true sandbox: always Env(s, a_final)
            z_next = self._imagine(z, final.keyboard, final.mouse)
            # Blend with expert next for stability (sandbox grounding proxy)
            z_exp = tr.latent_tp1.unsqueeze(0).to(self.device).float()
            # Use expert next as privileged target when draft≈expert keys (optional)
            target = z_exp
            r = float(
                combine_rewards(
                    sharpness_preserve_reward(z, target),
                    dynamics_consistency_reward(z_hat, target),
                    weights=[1.0, float(self.cfg.dyn_reward_weight)],
                ).mean().item()
            )
            self.buf.add_policy(Transition(
                latent_t=tr.latent_t,
                keyboard=final.keyboard.squeeze(0).cpu(),
                mouse=final.mouse.squeeze(0).cpu(),
                latent_tp1=target.squeeze(0).cpu(),
                reward=r,
                source="policy",
                meta={
                    "old_log_prob": float(final.log_prob.item()),
                    "value": float(final.value.item()),
                    "reliability": rel,
                    "imagined": z_hat.squeeze(0).cpu(),
                },
            ))
            # Also store imagined transition for imagination-augmented RL
            self.buf.add_imagined(Transition(
                latent_t=tr.latent_t,
                keyboard=final.keyboard.squeeze(0).cpu(),
                mouse=final.mouse.squeeze(0).cpu(),
                latent_tp1=z_hat.squeeze(0).cpu(),
                reward=r,
                source="imagined",
                meta={"old_log_prob": float(final.log_prob.item()),
                      "value": float(final.value.item())},
            ))
        return {
            "mean_reliability": float(sum(rels) / max(1, len(rels))),
            "reflect_frac": n_reflect / max(1, n),
        }

    # ------------------------------------------------------------------ #
    # Joint updates (CoMAP: WMSD + policy reflection/RL)
    # ------------------------------------------------------------------ #
    def update_wm(self) -> Dict[str, float]:
        """L_WM = L_dyn(on-policy+expert) + λ_sd * KL/MSE(student, teacher soft target).

        CoMAP uses token-level self-distillation with privileged teacher.
        Visual analogue: teacher EMA provides soft next-latent; privileged
        target is sandbox/expert s'.
        """
        B = int(self.cfg.batch_size)
        mix = self.buf.sample(B)  # expert_ratio mix
        z, kb, ms = _batch_actions(mix, self.device)
        z1 = torch.stack([t.latent_tp1.float() for t in mix]).to(self.device)

        pred = self.wm_student(z, kb, ms)
        loss_dyn = F.mse_loss(pred, z1)

        with torch.no_grad():
            soft = self.wm_teacher(z, kb, ms)
        loss_sd = F.mse_loss(pred, soft)  # soft target distill (sg teacher)

        # Extra expert-only dyn to avoid collapse
        exp = self.buf.sample(B, prefer="expert")
        ze, kbe, mse = _batch_actions(exp, self.device)
        z1e = torch.stack([t.latent_tp1.float() for t in exp]).to(self.device)
        loss_exp = F.mse_loss(self.wm_student(ze, kbe, mse), z1e)

        loss = (
            loss_dyn
            + float(self.cfg.lambda_sd) * loss_sd
            + float(self.cfg.lambda_expert) * loss_exp
        )
        self.wm_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wm_student.parameters(), float(self.cfg.max_grad_norm))
        self.wm_opt.step()

        # EMA teacher ← student  (CoMAP teacher mode)
        with torch.no_grad():
            for p_t, p_s in zip(self.wm_teacher.parameters(), self.wm_student.parameters()):
                p_t.data.mul_(self.ema_tau).add_(p_s.data, alpha=1.0 - self.ema_tau)

        return {
            "wm_loss": float(loss.item()),
            "L_dyn": float(loss_dyn.item()),
            "L_sd": float(loss_sd.item()),
            "L_exp": float(loss_exp.item()),
        }

    def update_policy(self) -> Dict[str, float]:
        """RL on real/imag mix; imagination weight anneals (still gated by reliability)."""
        B = int(self.cfg.batch_size)
        # anneal real_ratio
        T = max(1, int(self.cfg.max_steps))
        t = min(self.step, T) / T
        real_ratio = float(self.cfg.real_ratio) + (
            float(self.cfg.real_ratio_end) - float(self.cfg.real_ratio)
        ) * t

        batch = self.buf.sample_real_imag_mix(B, real_ratio=real_ratio)
        # fill missing logprobs
        self.policy.eval()
        with torch.no_grad():
            for tr in batch:
                if "old_log_prob" not in tr.meta:
                    z = tr.latent_t.unsqueeze(0).to(self.device).float()
                    kb, ms = _mean_action(tr.keyboard, tr.mouse)
                    res = self.policy.evaluate_actions(
                        z, kb.unsqueeze(0).to(self.device), ms.unsqueeze(0).to(self.device)
                    )
                    tr.meta["old_log_prob"] = float(res.log_prob.item())
                    tr.meta["value"] = float(res.value.item())

        z = torch.stack([t.latent_t.float() for t in batch]).to(self.device)
        kbs, mss, rews, old_lps, vals = [], [], [], [], []
        for t_ in batch:
            kb, ms = _mean_action(t_.keyboard, t_.mouse)
            kbs.append(kb)
            mss.append(ms)
            rews.append(t_.reward)
            old_lps.append(float(t_.meta["old_log_prob"]))
            vals.append(float(t_.meta.get("value", 0.0)))
        kb = torch.stack(kbs).to(self.device)
        ms = torch.stack(mss).to(self.device)
        returns = torch.tensor(rews, device=self.device, dtype=torch.float32)
        old_lp = torch.tensor(old_lps, device=self.device, dtype=torch.float32)
        old_v = torch.tensor(vals, device=self.device, dtype=torch.float32)
        adv = returns - old_v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        self.policy.train()
        stats = ppo_update(
            self.policy, self.pi_opt, z, kb, ms, old_lp, returns, adv,
            clip_eps=float(self.cfg.clip_eps),
            vf_coef=float(self.cfg.vf_coef),
            ent_coef=float(self.cfg.ent_coef),
        )
        stats["real_ratio"] = real_ratio
        return stats

    def reality_gap(self, n: int = 32) -> float:
        batch = self.buf.sample(min(n, len(self.buf.expert)), prefer="expert")
        errs = []
        with torch.no_grad():
            z, kb, ms = _batch_actions(batch, self.device)
            z1 = torch.stack([t.latent_tp1.float() for t in batch]).to(self.device)
            # evaluate student (current F_φ)
            pred = self.wm_student(z, kb, ms)
            errs.append(float((pred - z1).pow(2).mean().item()))
        return errs[0]

    def train(self):
        t0 = time.time()
        for self.step in range(1, int(self.cfg.max_steps) + 1):
            inter = self.interact_batch(int(self.cfg.collect_n))
            wm_stats = self.update_wm()
            pi_stats = self.update_policy()

            if self.step % int(self.cfg.log_interval) == 0:
                gap = self.reality_gap(int(self.cfg.eval_n))
                rec = {
                    "step": self.step,
                    "reality_gap": gap,
                    **inter,
                    **wm_stats,
                    **pi_stats,
                    "buf_P": len(self.buf.policy),
                    "buf_I": len(self.buf.imagined),
                }
                self.history.append(rec)
                print(
                    f"[coevolve] step {self.step:5d} | gap {gap:.4f} | "
                    f"wm {wm_stats['wm_loss']:.4f} (dyn {wm_stats['L_dyn']:.4f} sd {wm_stats['L_sd']:.4f}) | "
                    f"pi {pi_stats['loss']:.4f} | rel {inter['mean_reliability']:.3f} | "
                    f"reflect {inter['reflect_frac']:.2f} | real_ratio {pi_stats['real_ratio']:.2f}",
                    flush=True,
                )

            if self.step % int(self.cfg.save_interval) == 0:
                self.save()

        self.save()
        with open(os.path.join(self.logdir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[coevolve] done in {time.time()-t0:.1f}s -> {self.logdir}")

    def save(self):
        torch.save(
            {"toy": self.wm_student.state_dict(), "step": self.step, "kind": "student"},
            os.path.join(self.logdir, f"wm_student_{self.step:06d}.pt"),
        )
        torch.save(
            {"toy": self.wm_teacher.state_dict(), "step": self.step, "kind": "teacher_ema"},
            os.path.join(self.logdir, "wm_teacher_latest.pt"),
        )
        torch.save(
            {"toy": self.wm_student.state_dict(), "step": self.step},
            os.path.join(self.logdir, "wm_student_latest.pt"),
        )
        self.policy.save(os.path.join(self.logdir, f"policy_{self.step:06d}.pt"))
        self.policy.save(os.path.join(self.logdir, "policy_latest.pt"))
        print(f"[coevolve] saved checkpoints @ step {self.step}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="curriculum/configs/coevolve.yaml")
    args = ap.parse_args()
    os.chdir(_ROOT)
    cfg = OmegaConf.load(args.config)
    CoEvolver(cfg).train()


if __name__ == "__main__":
    main()
