#!/usr/bin/env python3
"""Self-evolving world model (WM-only): Discover → Probe → Compress.

RP spine
--------
  1. Discover  — auto-propose capability candidates (novelty / uncertainty)
  2. Probe     — intervention tests: is the capability effective?
  3. Compress  — keep only verified knowledge in experience memory

No VLA. Sandbox provides Env ground truth.

Usage:
  python -m curriculum.self_evolve_wm --config curriculum/configs/self_evolve_wm.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from curriculum.capability_discovery import (
    CapabilityDiscoverer,
    discover_probe_compress,
)
from curriculum.frozen_wm import FrozenWorldModel, ToyDynamics
from curriculum.knowledge_memory import WorldKnowledgeMemory, pool_latent
from curriculum.probes import WorldModelProber, build_horizon_chains, _synth_action
from curriculum.replay_buffer import MixedReplayBuffer, Transition
from curriculum.sandbox_experience import InterventionPair, build_collector


def _mean_action(kb: torch.Tensor, ms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if kb.ndim == 2:
        kb = kb.mean(0)
    if ms.ndim == 2:
        ms = ms.mean(0)
    return kb.float(), ms.float()


class SelfEvolvingWM:
    """Discover capabilities → probe effectiveness → compress into experience."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device
            if cfg.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.nfpb = int(cfg.num_frame_per_block)
        assert cfg.wm_backend == "toy", (
            "self_evolve_wm currently ships with toy backend; "
            "full pretrained_wm can later reuse ActionConsistencyTrainer as student F."
        )

        teacher_wrap = FrozenWorldModel(
            backend="toy",
            device=str(self.device),
            toy_ckpt=cfg.get("toy_wm_ckpt", None),
        )
        self.teacher: ToyDynamics = teacher_wrap.toy
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.student: ToyDynamics = ToyDynamics().to(self.device)
        self.student.load_state_dict(self.teacher.state_dict())
        self.opt = torch.optim.AdamW(
            self.student.parameters(),
            lr=float(cfg.wm_lr),
            weight_decay=float(cfg.weight_decay),
        )
        self.anchor = {k: v.detach().clone() for k, v in self.student.state_dict().items()}

        self.logdir = cfg.logdir
        os.makedirs(self.logdir, exist_ok=True)

        self.buf = MixedReplayBuffer(expert_ratio=float(cfg.expert_ratio))
        n = 0
        if cfg.get("data_root") and os.path.isdir(str(cfg.data_root)):
            n = self.buf.load_expert_pt_dir(
                cfg.data_root,
                max_clips=cfg.get("max_clips", None),
                block_frames=self.nfpb,
                stride=int(cfg.get("block_stride", 1)),
            )
        print(f"[self_evolve_wm] offline expert transitions: {n}")

        self.use_sandbox = bool(cfg.get("use_sandbox", True))
        self.sandbox_backend = str(cfg.get("sandbox_backend", "stub"))
        self.collector = None
        self.intervention_buf: List[InterventionPair] = []
        if self.use_sandbox:
            self.collector = build_collector(cfg, self.logdir, self.nfpb)
            print(
                f"[self_evolve_wm] sandbox backend={self.sandbox_backend} "
                f"encoder={cfg.get('latent_encoder', 'cheap')}",
                flush=True,
            )

        if n == 0 and not self.use_sandbox:
            raise RuntimeError("need data_root expert .pt or use_sandbox=true")

        self.memory = WorldKnowledgeMemory(
            merge_thresh=float(cfg.get("memory_merge_thresh", 0.85)),
            max_items=int(cfg.get("memory_max_items", 5000)),
            min_confidence=float(cfg.get("memory_min_confidence", 0.40)),
        )
        mem_path = cfg.get("memory_init", None)
        if mem_path and os.path.exists(mem_path):
            self.memory = WorldKnowledgeMemory.load(mem_path)
            print(f"[self_evolve_wm] loaded memory ({len(self.memory)} items) from {mem_path}")

        self.prober = WorldModelProber(
            model=self.student,
            device=self.device,
            horizon=int(cfg.get("probe_horizon", 3)),
            n_intervene=int(cfg.get("n_intervene", 3)),
            transfer_pairs=int(cfg.get("transfer_pairs", 16)),
        )
        self.discoverer = CapabilityDiscoverer(
            memory=self.memory,
            student=self.student,
            teacher=self.teacher,
            device=self.device,
            under_explore_bonus=float(cfg.get("under_explore_bonus", 0.3)),
        )

        self.step = 0
        self.history: List[Dict] = []

        self.lambda_causal = float(cfg.get("lambda_causal", 0.5))
        self.lambda_horizon = float(cfg.get("lambda_horizon", 0.3))
        self.lambda_anchor = float(cfg.get("lambda_anchor", 0.1))
        self.lambda_memory = float(cfg.get("lambda_memory", 0.2))
        self.lambda_intervene = float(cfg.get("lambda_intervene", 0.5))
        self.fail_thresh = float(cfg.get("fail_thresh", 0.45))
        self.pass_thresh = float(cfg.get("pass_thresh", cfg.get("fail_thresh", 0.45)))
        self.discover_topk = int(cfg.get("discover_topk", 8))
        self.ema_tau = float(cfg.get("ema_tau", 0.99))

    # ------------------------------------------------------------------ #
    def collect_sandbox(self) -> Dict[str, float]:
        assert self.collector is not None
        max_turns = int(self.cfg.get("sandbox_max_turns", 8))
        if self.sandbox_backend == "stub":
            out = self.collector.collect_episode(self.buf.expert, max_turns=max_turns)
        elif self.sandbox_backend == "llm_api":
            out = self.collector.collect_episode(
                max_turns=max_turns,
                prefix_turns=int(self.cfg.get("llm_api_prefix_turns", 2)),
            )
        else:
            out = self.collector.collect_episode(max_turns=max_turns)

        for tr in out.get("transitions", []):
            # sandbox transitions are ground-truth Env → treat as expert
            tr.source = "expert"
            self.buf.add_expert(tr)
        for pair in out.get("interventions", []):
            self.intervention_buf.append(pair)
            # cap
            if len(self.intervention_buf) > int(self.cfg.get("max_interventions", 512)):
                self.intervention_buf = self.intervention_buf[-512:]

        return {
            "sandbox/n_trans": float(out.get("n_trans", 0)),
            "sandbox/n_intervene": float(len(out.get("interventions", []))),
            "sandbox/ep_reward": float(out.get("ep_reward", 0.0)),
            "sandbox/expert_size": float(len(self.buf.expert)),
        }

    def _ema_teacher(self):
        with torch.no_grad():
            for p_t, p_s in zip(self.teacher.parameters(), self.student.parameters()):
                p_t.data.mul_(self.ema_tau).add_(p_s.data, alpha=1.0 - self.ema_tau)

    def _priority_weight(self, probe_conf: float, novelty: float) -> float:
        fail = max(0.0, self.fail_thresh - probe_conf)
        return 1.0 + 2.0 * fail + 1.5 * novelty

    def _sample_train_batch(self, batch_size: int) -> List[Transition]:
        out: List[Transition] = []
        n_fail = min(len(self.buf.policy), max(1, batch_size // 4)) if self.buf.policy else 0
        if n_fail:
            out.extend(self.buf.sample(n_fail, prefer="policy"))
        remain = batch_size - len(out)
        if remain > 0:
            prefer = "expert" if self.buf.expert else "policy"
            out.extend(self.buf.sample(remain, prefer=prefer))
        return out

    def _pool(self) -> List[Transition]:
        return self.buf.expert if self.buf.expert else self.buf.policy

    def discover_probe_compress_step(self, n: int) -> Dict[str, float]:
        """Core RP step: Discover → Probe → Compress into experience."""
        pool = self._pool()
        if not pool:
            return {"discover/n": 0.0}
        # Prefer recent sandbox expert transitions as discovery substrate
        batch = self.buf.sample(min(n, len(pool)), prefer="expert" if self.buf.expert else "policy")
        inter = self.intervention_buf[-int(self.cfg.get("probe_interventions", 16)):]
        self.prober.model = self.student
        self.discoverer.student = self.student
        self.discoverer.teacher = self.teacher
        self.student.eval()

        cycle = discover_probe_compress(
            self.discoverer,
            self.prober,
            transitions=batch,
            pool=pool,
            interventions=inter if inter else None,
            topk=min(self.discover_topk, len(batch)),
            pass_thresh=self.pass_thresh,
        )

        # Failed probes → priority replay (do NOT compress into experience)
        for c in cycle["discarded"]:
            tr = c.to_transition()
            tr.meta = {
                **(tr.meta or {}),
                "probe_conf": float((c.meta or {}).get("probe_conf", 0.0)),
                "novelty": c.novelty,
            }
            self.buf.add_policy(tr)

        # Aggregate probe scores from verified + discarded for logging
        confs = [v.confidence for v in cycle["verified"]]
        confs += [float((c.meta or {}).get("probe_conf", 0.0)) for c in cycle["discarded"]]
        mean_conf = float(sum(confs) / max(1, len(confs)))

        # Also run a global probe snapshot for dashboards
        chains = build_horizon_chains(
            pool,
            horizon=int(self.cfg.get("probe_horizon", 3)),
            max_chains=int(self.cfg.get("max_chains", 16)),
        )
        snap = self.prober.probe_batch(
            batch, pool, chains, interventions=inter if inter else None
        )

        metrics = {
            "discover/n": float(cycle["n_discovered"]),
            "probe/n_verified": float(cycle["n_verified"]),
            "probe/n_discarded": float(cycle["n_discarded"]),
            "compress/n": float(cycle["n_compressed"]),
            "compress/memory_size": float(len(self.memory)),
            "probe/confidence": mean_conf,
            "probe/sandbox_grounded": float(snap.details.get("sandbox_grounded", 0.0)),
            "memory/accepted": float(self.memory.stats["accepted"]),
            "memory/merged": float(self.memory.stats["merged"]),
            "memory/rejected": float(self.memory.stats["rejected"]),
            "priority/size": float(len(self.buf.policy)),
            "intervene/buf": float(len(self.intervention_buf)),
        }
        metrics.update({f"probe/{k}": v for k, v in snap.scores.items()})
        # discovery score of top candidate
        if cycle["candidates"]:
            metrics["discover/top_score"] = float(cycle["candidates"][0].discovery_score)
            metrics["discover/top_novelty"] = float(cycle["candidates"][0].novelty)
        return metrics

    # backward-compatible alias
    def probe_and_consolidate(self, n: int) -> Dict[str, float]:
        return self.discover_probe_compress_step(n)

    def _causal_contrastive_loss(self, batch: Sequence[Transition]) -> torch.Tensor:
        if not batch:
            return torch.zeros((), device=self.device)
        losses = []
        for t in batch:
            z = t.latent_t.unsqueeze(0).to(self.device).float()
            z_star = t.latent_tp1.unsqueeze(0).to(self.device).float()
            kb, ms = _mean_action(t.keyboard, t.mouse)
            pred = self.student(z, kb.unsqueeze(0).to(self.device), ms.unsqueeze(0).to(self.device))
            pos = F.mse_loss(pred, z_star)
            kb_n, ms_n = _synth_action(kb.cpu(), ms.cpu())
            pred_n = self.student(
                z, kb_n.unsqueeze(0).to(self.device), ms_n.unsqueeze(0).to(self.device)
            )
            neg_err = F.mse_loss(pred_n, z_star)
            hinge = F.relu(pos - neg_err + float(self.cfg.get("causal_margin", 0.01)))
            sep = F.mse_loss(pred, pred_n)
            losses.append(pos + hinge + 0.1 * F.relu(0.01 - sep))
        return torch.stack(losses).mean()

    def _sandbox_intervene_loss(self) -> torch.Tensor:
        """Supervise F with real Env outcomes under two intervened actions."""
        if not self.intervention_buf:
            return torch.zeros((), device=self.device)
        pairs = random.sample(
            self.intervention_buf,
            min(len(self.intervention_buf), int(self.cfg.get("intervene_batch", 8))),
        )
        losses = []
        for p in pairs:
            z = p.latent_t.unsqueeze(0).to(self.device).float()
            kb_a, ms_a = _mean_action(p.kb_a, p.ms_a)
            kb_b, ms_b = _mean_action(p.kb_b, p.ms_b)
            pred_a = self.student(
                z, kb_a.unsqueeze(0).to(self.device), ms_a.unsqueeze(0).to(self.device)
            )
            pred_b = self.student(
                z, kb_b.unsqueeze(0).to(self.device), ms_b.unsqueeze(0).to(self.device)
            )
            gt_a = p.latent_a.unsqueeze(0).to(self.device).float()
            gt_b = p.latent_b.unsqueeze(0).to(self.device).float()
            losses.append(F.mse_loss(pred_a, gt_a) + F.mse_loss(pred_b, gt_b))
        return torch.stack(losses).mean()

    def _horizon_loss(self, chains: Sequence[Sequence[Transition]]) -> torch.Tensor:
        if not chains:
            return torch.zeros((), device=self.device)
        losses = []
        for chain in chains:
            z = chain[0].latent_t.unsqueeze(0).to(self.device).float()
            step_losses = []
            for t in chain:
                kb, ms = _mean_action(t.keyboard, t.mouse)
                z = self.student(
                    z, kb.unsqueeze(0).to(self.device), ms.unsqueeze(0).to(self.device)
                )
                gt = t.latent_tp1.unsqueeze(0).to(self.device).float()
                step_losses.append(F.mse_loss(z, gt))
            losses.append(torch.stack(step_losses).mean())
        return torch.stack(losses).mean()

    def _memory_prior_loss(self, batch: Sequence[Transition]) -> torch.Tensor:
        losses = []
        for t in batch:
            prior = self.memory.effect_prior(t.latent_t, t.keyboard, t.mouse)
            if prior is None:
                continue
            z = t.latent_t.unsqueeze(0).to(self.device).float()
            kb, ms = _mean_action(t.keyboard, t.mouse)
            pred = self.student(
                z, kb.unsqueeze(0).to(self.device), ms.unsqueeze(0).to(self.device)
            )
            effect = pool_latent(pred - z).squeeze(0)
            losses.append(F.mse_loss(effect, prior.to(self.device)))
        if not losses:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).mean()

    def _anchor_loss(self) -> torch.Tensor:
        loss = torch.zeros((), device=self.device)
        for k, v in self.student.state_dict().items():
            loss = loss + F.mse_loss(v.float(), self.anchor[k].to(self.device).float())
        return loss / max(1, len(self.anchor))

    def update_wm(self, batch_size: int) -> Dict[str, float]:
        self.student.train()
        if len(self._pool()) == 0:
            return {"loss/total": 0.0}
        batch = self._sample_train_batch(batch_size)
        zs, z1s, kbs, mss = [], [], [], []
        weights = []
        for t in batch:
            zs.append(t.latent_t.float())
            z1s.append(t.latent_tp1.float())
            kb, ms = _mean_action(t.keyboard, t.mouse)
            kbs.append(kb)
            mss.append(ms)
            conf = float((t.meta or {}).get("probe_conf", 0.5))
            nov = float((t.meta or {}).get("novelty", 0.5))
            weights.append(self._priority_weight(conf, nov))

        z = torch.stack(zs).to(self.device)
        z1 = torch.stack(z1s).to(self.device)
        kb = torch.stack(kbs).to(self.device)
        ms = torch.stack(mss).to(self.device)
        w = torch.tensor(weights, device=self.device, dtype=torch.float32)
        w = w / w.mean().clamp_min(1e-6)

        pred = self.student(z, kb, ms)
        per = (pred.float() - z1.float()).pow(2).reshape(len(batch), -1).mean(dim=1)
        l_dyn = (per * w).mean()

        l_causal = self._causal_contrastive_loss(batch)
        l_intervene = self._sandbox_intervene_loss()
        pool = self._pool()
        chains = build_horizon_chains(
            batch + (self.buf.sample(min(32, len(pool)), prefer="expert" if self.buf.expert else "policy") if pool else []),
            horizon=int(self.cfg.get("probe_horizon", 3)),
            max_chains=int(self.cfg.get("train_chains", 8)),
        )
        l_horizon = self._horizon_loss(chains)
        l_mem = self._memory_prior_loss(batch)
        l_anchor = self._anchor_loss()

        loss = (
            l_dyn
            + self.lambda_causal * l_causal
            + self.lambda_intervene * l_intervene
            + self.lambda_horizon * l_horizon
            + self.lambda_memory * l_mem
            + self.lambda_anchor * l_anchor
        )
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.student.parameters(), float(self.cfg.get("max_grad_norm", 1.0))
        )
        self.opt.step()
        self._ema_teacher()

        return {
            "loss/total": float(loss.item()),
            "loss/dyn": float(l_dyn.item()),
            "loss/causal": float(l_causal.item()),
            "loss/intervene": float(l_intervene.item()),
            "loss/horizon": float(l_horizon.item()),
            "loss/memory": float(l_mem.item()),
            "loss/anchor": float(l_anchor.item()),
        }

    @torch.no_grad()
    def eval_pred(self, n: int = 32) -> float:
        self.student.eval()
        pool = self._pool()
        if not pool:
            return float("nan")
        batch = self.buf.sample(min(n, len(pool)), prefer="expert" if self.buf.expert else "policy")
        errs = []
        for t in batch:
            z = t.latent_t.unsqueeze(0).to(self.device).float()
            kb, ms = _mean_action(t.keyboard, t.mouse)
            pred = self.student(
                z, kb.unsqueeze(0).to(self.device), ms.unsqueeze(0).to(self.device)
            )
            gt = t.latent_tp1.unsqueeze(0).to(self.device).float()
            errs.append(float(F.mse_loss(pred, gt).item()))
        return float(sum(errs) / max(1, len(errs)))

    def save(self, tag: str):
        path = os.path.join(self.logdir, f"wm_{tag}.pt")
        torch.save(
            {
                "student": self.student.state_dict(),
                "teacher": self.teacher.state_dict(),
                "step": self.step,
                "cfg": OmegaConf.to_container(self.cfg, resolve=True),
            },
            path,
        )
        self.memory.save(os.path.join(self.logdir, f"memory_{tag}.json"))
        with open(os.path.join(self.logdir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        return path

    def run(self):
        max_steps = int(self.cfg.max_steps)
        log_interval = int(self.cfg.log_interval)
        save_interval = int(self.cfg.save_interval)
        probe_every = int(self.cfg.get("probe_every", 1))
        collect_every = int(self.cfg.get("collect_every", 1))
        probe_n = int(self.cfg.get("probe_n", 16))
        batch_size = int(self.cfg.batch_size)

        t0 = time.time()
        try:
            for step in range(1, max_steps + 1):
                self.step = step
                metrics: Dict[str, float] = {}

                if self.use_sandbox and self.collector is not None and step % collect_every == 0:
                    try:
                        metrics.update(self.collect_sandbox())
                    except Exception as e:
                        print(f"[self_evolve_wm] sandbox collect failed: {e}", flush=True)
                        if self.sandbox_backend != "stub" and len(self._pool()) == 0:
                            raise

                if step % probe_every == 0:
                    metrics.update(self.discover_probe_compress_step(probe_n))

                metrics.update(self.update_wm(batch_size))

                if step % log_interval == 0 or step == 1:
                    metrics["eval/pred_mse"] = self.eval_pred(int(self.cfg.get("eval_n", 32)))
                    metrics["time_s"] = time.time() - t0
                    self.history.append({"step": step, **metrics})
                    nice = {
                        k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in metrics.items()
                    }
                    print(f"[self_evolve_wm] step {step}/{max_steps} {nice}", flush=True)

                if step % save_interval == 0 or step == max_steps:
                    path = self.save(f"{step:06d}")
                    print(
                        f"[self_evolve_wm] saved {path} | memory={len(self.memory)} "
                        f"| intervene={len(self.intervention_buf)}",
                        flush=True,
                    )
        finally:
            if self.collector is not None:
                try:
                    self.collector.close()
                except Exception:
                    pass

        self.save("latest")
        print(
            f"[self_evolve_wm] done. memory={len(self.memory)} "
            f"accepted={self.memory.stats['accepted']} "
            f"rejected={self.memory.stats['rejected']} "
            f"interventions={len(self.intervention_buf)}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="curriculum/configs/self_evolve_wm.yaml")
    args, _ = ap.parse_known_args()
    os.chdir(_ROOT)
    cfg = OmegaConf.load(args.config)
    runner = SelfEvolvingWM(cfg)
    runner.run()


if __name__ == "__main__":
    main()
