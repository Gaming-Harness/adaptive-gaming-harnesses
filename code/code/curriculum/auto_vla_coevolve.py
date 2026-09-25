#!/usr/bin/env python3
"""Auto sidecar for Qwen3-VL GRPO ↔ WM.

Consumes ARES dumped VLA sandbox episodes, runs probe-gated WorthLearn,
updates WM only on Learn, and writes coevolve_schedule.json so GRPO reward
optimizes **transfer essence** rather than connection-pattern MSE.

Loop
----
  VLA(π) ──sandbox──► dump episode
                         │
                         ▼
              AutoScheduler Learn|Skip|Re-verify
                         │
           Learn → WM update; Skip → no WM step
                         │
                         ▼
              schedule.json → ARES GRPO λ / transfer_λ

Usage:
  python -m curriculum.auto_vla_coevolve --dump_dir ... --logdir ... --steps 20
  python -m curriculum.run_curriculum smoke_auto_vla_coevolve
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
from omegaconf import OmegaConf

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from curriculum.ares_wm_bridge import list_episode_dirs
from curriculum.auto_scheduler import AutoScheduler
from curriculum.capability_discovery import CapabilityDiscoverer, discover_probe_compress
from curriculum.frozen_wm import ToyDynamics
from curriculum.hypothesis import Decision
from curriculum.knowledge_memory import WorldKnowledgeMemory
from curriculum.policy_transfer_probe import episode_to_transitions, score_policy_episode
from curriculum.probe_policy import probe_focus_actions
from curriculum.probes import WorldModelProber, build_horizon_chains
from curriculum.replay_buffer import MixedReplayBuffer, Transition


def synth_dump_episodes(
    dump_dir: str,
    n_eps: int = 4,
    n_frames: int = 10,
    seed: int = 0,
) -> List[str]:
    """Write fake ARES dumps (enough frames for probe-gated GRPO smoke)."""
    import numpy as np
    from PIL import Image

    os.makedirs(dump_dir, exist_ok=True)
    actions_pool = [
        "<actions> keyPress(w) ; keyPress(w) ; keyPress(w) ; keyPress(w)</actions>",
        "<actions> keyPress(a) ; keyPress(a) ; keyPress(w) ; keyPress(w)</actions>",
        "<actions> mouseMove(-8, 0) ; mouseMove(-8, 0) ; keyPress(w) ; keyPress(w)</actions>",
        "<actions> mouseMove(8, 0) ; mouseMove(8, 0) ; keyPress(d) ; keyPress(d)</actions>",
        "<actions> mouseClick(left) ; no_op ; mouseClick(left) ; no_op</actions>",
        "<actions> keyPress(space) ; keyPress(w) ; keyPress(w) ; keyPress(w)</actions>",
    ]
    out: List[str] = []
    for i in range(int(n_eps)):
        ep_id = f"ep_synth_{seed}_{i}"
        ep_dir = os.path.join(dump_dir, ep_id)
        os.makedirs(ep_dir, exist_ok=True)
        frames: List[str] = []
        for t in range(int(n_frames)):
            img = np.zeros((64, 96, 3), dtype=np.uint8)
            img[:, :, 0] = (40 + 8 * t + i * 13) % 256
            img[:, :, 1] = (80 + 12 * i) % 256
            img[:, :, 2] = (120 + 5 * t) % 256
            y0, x0 = 10 + (t * 3) % 40, 10 + (t * 5 + i * 7) % 70
            img[y0:y0 + 12, x0:x0 + 12] = 220
            name = f"frame_{t:04d}.png"
            Image.fromarray(img).save(os.path.join(ep_dir, name))
            frames.append(name)
        acts = [actions_pool[(i + t) % len(actions_pool)] for t in range(int(n_frames) - 1)]
        with open(os.path.join(ep_dir, "actions.jsonl"), "w") as f:
            for t, text in enumerate(acts):
                f.write(json.dumps({"turn": t, "action_text": text, "reward": 0.1 * (t + 1)}) + "\n")
        meta = {
            "episode_id": ep_id,
            "n_frames": int(n_frames),
            "n_actions": len(acts),
            "episode_score": 0.5 + 0.1 * i,
            "frames": frames,
            "meta": {
                "wm_bonus": 0.30 + 0.05 * i,
                "wm_mse": 0.04,
                "env_score": 0.4,
                "synth": True,
            },
        }
        with open(os.path.join(ep_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        out.append(ep_dir)
    return out


def _load_episode(ep_dir: str) -> Dict[str, Any]:
    meta = json.load(open(os.path.join(ep_dir, "meta.json")))
    frames = [os.path.join(ep_dir, f) for f in meta.get("frames", [])]
    actions: List[str] = []
    rewards: List[float] = []
    act_path = os.path.join(ep_dir, "actions.jsonl")
    if os.path.isfile(act_path):
        with open(act_path) as f:
            for line in f:
                row = json.loads(line)
                actions.append(row.get("action_text", ""))
                rewards.append(float(row.get("reward", 0.0)))
    return {
        "dir": ep_dir,
        "meta": meta,
        "frames": frames,
        "actions": actions,
        "rewards": rewards,
        "episode_score": float(meta.get("episode_score", 0.0)),
        "env_score": float((meta.get("meta") or {}).get("env_score", 0.0)),
        "wm_mse": float((meta.get("meta") or {}).get("wm_mse", 0.0)),
        "wm_bonus": float((meta.get("meta") or {}).get("wm_bonus", 0.0)),
    }


class AutoVLACoevolve:
    """Probe-gated WM experimenter sitting beside Qwen GRPO."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if cfg.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.logdir = str(cfg.logdir)
        _home = "/path/to/project"
        _abs_log = os.path.abspath(os.path.join(_ROOT, self.logdir) if not os.path.isabs(self.logdir) else self.logdir)
        if not (_abs_log == _home or _abs_log.startswith(_home + os.sep)):
            raise RuntimeError(
                f"[safety] logdir must be under {_home} (no writes outside anonymous), got {_abs_log}"
            )
        os.makedirs(self.logdir, exist_ok=True)
        self.dump_dir = str(cfg.dump_dir)
        _abs_dump = os.path.abspath(os.path.join(_ROOT, self.dump_dir) if not os.path.isabs(self.dump_dir) else self.dump_dir)
        if not (_abs_dump == _home or _abs_dump.startswith(_home + os.sep)):
            raise RuntimeError(
                f"[safety] dump_dir must be under {_home}, got {_abs_dump}"
            )
        self.nfpb = int(cfg.get("num_frame_per_block", 3))

        self.student = ToyDynamics().to(self.device)
        self.teacher = ToyDynamics().to(self.device)
        wm_ckpt = cfg.get("wm_ckpt") or os.path.join(self.logdir, "wm_student_latest.pt")
        toy0 = cfg.get("toy_wm_ckpt", "curriculum/outputs/toy_wm0.pt")
        loaded = False
        for path in (wm_ckpt, toy0):
            if path and os.path.isfile(path):
                ck = torch.load(path, map_location="cpu", weights_only=False)
                state = ck.get("toy", ck.get("student", ck))
                self.student.load_state_dict(state, strict=False)
                loaded = True
                print(f"[auto_vla] loaded WM from {path}", flush=True)
                break
        if not loaded:
            print("[auto_vla] warning: no WM ckpt, random init", flush=True)
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.AdamW(
            self.student.parameters(),
            lr=float(cfg.get("wm_lr", 1e-4)),
            weight_decay=float(cfg.get("weight_decay", 0.01)),
        )

        self.memory = WorldKnowledgeMemory(
            merge_thresh=float(cfg.get("memory_merge_thresh", 0.85)),
            max_items=int(cfg.get("memory_max_items", 5000)),
            min_confidence=float(cfg.get("memory_min_confidence", 0.35)),
        )
        mem_p = os.path.join(self.logdir, "memory_latest.json")
        if os.path.isfile(mem_p):
            self.memory = WorldKnowledgeMemory.load(mem_p)

        self.buf = MixedReplayBuffer(expert_ratio=0.5)
        self.prober = WorldModelProber(self.student, self.device)
        self.discoverer = CapabilityDiscoverer(
            self.memory, self.student, self.teacher, self.device,
        )
        self.scheduler = AutoScheduler(
            self.memory,
            student=self.student,
            teacher=self.teacher,
            device=self.device,
            lambda_cost=float(cfg.get("worth_lambda_cost", 0.15)),
            skip_thresh=float(cfg.get("worth_skip_thresh", 0.18)),
            learn_thresh=float(cfg.get("worth_learn_thresh", 0.28)),
            mastered_min_verify=int(cfg.get("mastered_min_verify", 2)),
            mastered_min_conf=float(cfg.get("mastered_min_conf", 0.50)),
            reverify_every=int(cfg.get("reverify_every", 4)),
        )
        self.pass_thresh = float(cfg.get("pass_thresh", 0.35))
        self.seen: set = set()
        self.history: List[Dict[str, Any]] = []
        self.step = 0
        self.beta = float(cfg.get("connection_beta", 0.7))
        self.base_lambda = float(cfg.get("wm_reward_lambda", 0.5))
        self.base_transfer_lambda = float(cfg.get("transfer_lambda", 0.5))

    def _ingest_new(self, max_eps: int = 8) -> List[Dict[str, Any]]:
        eps = []
        for d in list_episode_dirs(self.dump_dir):
            if d in self.seen:
                continue
            try:
                ep = _load_episode(d)
            except Exception as e:
                print(f"[auto_vla] skip bad episode {d}: {e}", flush=True)
                self.seen.add(d)
                continue
            trans = episode_to_transitions(
                ep["frames"], ep["actions"], nfpb=self.nfpb, max_turns=24,
            )
            for tr in trans:
                tr.source = "policy"
                self.buf.add_policy(tr)
            ep["transitions"] = trans
            self.seen.add(d)
            eps.append(ep)
            if len(eps) >= max_eps:
                break
        return eps

    def _score_batch(self, eps: List[Dict[str, Any]]) -> Dict[str, float]:
        if not eps:
            return {"connection": 0.0, "transfer": 0.0, "essence": 0.0, "n_eps": 0.0}
        conn, trans, ess = [], [], []
        self.student.eval()
        for ep in eps:
            conn_u = 0.0
            try:
                # reuse dumped wm_bonus_unit if present
                meta = ep.get("meta", {}).get("meta") or {}
                if "wm_bonus" in meta and self.base_lambda:
                    conn_u = float(meta.get("wm_bonus", 0.0)) / max(self.base_lambda, 1e-6)
            except Exception:
                conn_u = 0.0
            st = score_policy_episode(
                self.student,
                ep["frames"],
                ep["actions"],
                device=self.device,
                connection_unit=conn_u,
                beta=self.beta,
                nfpb=self.nfpb,
            )
            ep["transfer_stats"] = st
            conn.append(st["connection"])
            trans.append(st["transfer"])
            ess.append(st["essence"])
        return {
            "connection": float(sum(conn) / len(conn)),
            "transfer": float(sum(trans) / len(trans)),
            "essence": float(sum(ess) / len(ess)),
            "n_eps": float(len(eps)),
            "shortcut": float(max(0.0, sum(conn) / len(conn) - sum(trans) / len(trans))),
        }

    def _write_schedule(self, agg: Dict[str, float], plan_mode: str) -> None:
        """Tell GRPO: reward essence, shrink MSE if connection shortcut."""
        shortcut = float(agg.get("shortcut", 0.0))
        essence = float(agg.get("essence", 0.0))
        # If policy is mostly connection, raise transfer λ and shrink MSE λ
        mse_shrink = 1.0
        lam = self.base_lambda
        lam_t = self.base_transfer_lambda
        if shortcut > 0.20:
            mse_shrink = 0.4
            lam_t = min(1.0, self.base_transfer_lambda + 0.3)
            lam = max(0.1, self.base_lambda * 0.5)
        if essence > 0.25 and shortcut < 0.10:
            lam_t = min(1.0, self.base_transfer_lambda + 0.1)
        if plan_mode == "skip":
            lam = min(lam, 0.15)
        cov = self.memory.coverage_report()
        payload = {
            "round": self.step,
            "lambda": lam,
            "transfer_lambda": lam_t,
            "slot_lambda": float(self.cfg.get("slot_lambda", 0.3)),
            "memory_path": os.path.join(self.logdir, "memory_latest.json"),
            "connection_beta": self.beta,
            "mse_shrink": mse_shrink,
            "imag_gamma": 0.2 if shortcut > 0.15 else 0.0,
            "imag_horizon": 3 if shortcut > 0.15 else 1,
            "update_mode": plan_mode,
            "essence_mean": essence,
            "connection_mean": float(agg.get("connection", 0.0)),
            "transfer_mean": float(agg.get("transfer", 0.0)),
            "knowledge_coverage": float(cov["family_coverage"]),
            "knowledge_unknown": float(cov["family_unknown"]),
            "probe_mode": bool(self.cfg.get("probe_mode", True)),
            "probe_alpha": float(self.cfg.get("probe_alpha", 0.55)),
            "probe_focus": probe_focus_actions(self.memory, k=4),
            "ts": time.time(),
        }
        path = os.path.join(self.logdir, "coevolve_schedule.json")
        tmp = path + f".tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
        return payload

    def _update_wm(self, batch: List[Transition]) -> Dict[str, float]:
        if not batch:
            return {"loss/dyn": 0.0, "loss/ic": 0.0, "loss/mem": 0.0, "loss/total": 0.0}
        from curriculum.knowledge_consolidation import consolidate_wm_loss

        self.student.train()
        loss, stats = consolidate_wm_loss(
            self.student,
            batch,
            self.memory,
            device=self.device,
            lambda_ic=float(self.cfg.get("lambda_ic", 0.3)),
            lambda_mem=float(self.cfg.get("lambda_mem", 0.2)),
        )
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.opt.step()
        with torch.no_grad():
            tau = 0.99
            for p_t, p_s in zip(self.teacher.parameters(), self.student.parameters()):
                p_t.data.mul_(tau).add_(p_s.data, alpha=1.0 - tau)
        return stats

    def save(self):
        torch.save(
            {"toy": self.student.state_dict(), "step": self.step},
            os.path.join(self.logdir, "wm_student_latest.pt"),
        )
        torch.save(
            {"toy": self.teacher.state_dict(), "step": self.step},
            os.path.join(self.logdir, "wm_teacher_latest.pt"),
        )
        self.memory.save(os.path.join(self.logdir, "memory_latest.json"))
        with open(os.path.join(self.logdir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    def run_step(self) -> Dict[str, Any]:
        self.step += 1
        eps = self._ingest_new(max_eps=int(self.cfg.get("max_eps_per_step", 8)))
        agg = self._score_batch(eps)
        substrate: List[Transition] = []
        for ep in eps:
            substrate.extend(ep.get("transitions") or [])
        if not substrate and self.buf.policy:
            substrate = self.buf.sample(min(16, len(self.buf.policy)), prefer="policy")

        self.scheduler.student = self.student
        self.scheduler.teacher = self.teacher
        self.scheduler.memory = self.memory
        plan = self.scheduler.build_plan(
            substrate or [],
            topk=int(self.cfg.get("hyp_topk", 10)),
            sandbox_turns=0,  # VLA already acted in sandbox
            step=self.step,
        )

        metrics: Dict[str, Any] = {**agg, **plan.metrics, "step": self.step, "n_new_eps": float(len(eps))}

        # Probe + compress on Learn; failure → scheduler queue
        if plan.update_mode in ("learn", "mixed", "reverify") and substrate:
            self.prober.model = self.student
            cycle = discover_probe_compress(
                self.discoverer,
                self.prober,
                transitions=substrate[: int(self.cfg.get("probe_n", 12))],
                pool=substrate,
                pass_thresh=self.pass_thresh,
                topk=min(8, len(substrate)),
            )
            for c in cycle["discarded"]:
                self.scheduler.note_probe_failure(
                    action=c.action_name,
                    probe_scores=dict((c.meta or {}).get("probe_scores") or {}),
                    confidence=float((c.meta or {}).get("probe_conf", 0.0)),
                    latent_t=c.latent_t,
                    keyboard=c.keyboard,
                    mouse=c.mouse,
                    latent_tp1=c.latent_tp1,
                )
            metrics["probe/n_verified"] = float(cycle["n_verified"])
            metrics["probe/n_discarded"] = float(cycle["n_discarded"])
            metrics["compress/n"] = float(cycle["n_compressed"])
            metrics["compress/memory_size"] = float(len(self.memory))
            if plan.update_mode != "skip":
                metrics.update(self._update_wm(substrate[:16]))
        else:
            metrics["auto/wm_gated"] = 1.0
            metrics["compress/memory_size"] = float(len(self.memory))

        n_mast = self.scheduler.promote_mastered()
        metrics["auto/mastered_new"] = float(n_mast)
        metrics["memory/mastered"] = float(sum(1 for it in self.memory.items if it.status == "mastered"))
        cov = self.memory.coverage_report()
        metrics["know/family_coverage"] = float(cov["family_coverage"])
        metrics["know/family_unknown"] = float(cov["family_unknown"])
        metrics["know/family_stale"] = float(cov["family_stale"])
        metrics["know/n_revoked"] = float(cov["n_revoked_items"])

        sched = self._write_schedule(agg, plan.update_mode)
        metrics["sched/lambda"] = float(sched["lambda"])
        metrics["sched/transfer_lambda"] = float(sched["transfer_lambda"])
        metrics["sched/mse_shrink"] = float(sched["mse_shrink"])
        self.history.append(metrics)
        self.save()
        return metrics

    def run(self):
        steps = int(self.cfg.get("max_steps", 20))
        poll = float(self.cfg.get("poll_s", 5.0))
        wait_dump = bool(self.cfg.get("wait_dump", False))
        print(
            f"[auto_vla] dump={self.dump_dir} logdir={self.logdir} steps={steps} device={self.device}",
            flush=True,
        )
        for i in range(steps):
            if wait_dump and not list_episode_dirs(self.dump_dir):
                print(f"[auto_vla] waiting for VLA dumps in {self.dump_dir} ...", flush=True)
                time.sleep(poll)
            m = self.run_step()
            nice = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
            print(f"[auto_vla] step {self.step}/{steps} {nice}", flush=True)
        cov = self.memory.coverage_report()
        report = {
            "mechanism_id": "self_aware_wm_knowledge_boundary",
            "steps": self.step,
            "memory": len(self.memory),
            "mastered": sum(1 for it in self.memory.items if it.status == "mastered"),
            "knowledge_coverage": cov["family_coverage"],
            "knowledge_unknown": cov["family_unknown"],
            "knowledge_stale": cov["family_stale"],
            "last": self.history[-1] if self.history else {},
            "claim": (
                "Primary: knowledge boundary (coverage / unknown / revoke) + "
                "intervention-sensitivity probes. GRPO credit uses essence/slot "
                "alignment so π is not rewarded for connection-only fit. "
                "Not a claim to beat Dreamer MSE."
            ),
        }
        with open(os.path.join(self.logdir, "auto_vla_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--dump_dir", default="")
    ap.add_argument("--logdir", default="")
    ap.add_argument("--steps", type=int, default=0)
    args, _ = ap.parse_known_args()
    os.chdir(_ROOT)
    if args.config:
        cfg = OmegaConf.load(args.config)
    else:
        cfg = OmegaConf.create({
            "dump_dir": args.dump_dir or "curriculum/outputs/coevolve_vla_auto/ares_rollouts",
            "logdir": args.logdir or "curriculum/outputs/coevolve_vla_auto",
            "max_steps": args.steps or 10,
            "device": "auto",
            "toy_wm_ckpt": "curriculum/outputs/toy_wm0.pt",
            "wm_reward_lambda": 0.5,
            "transfer_lambda": 0.5,
            "connection_beta": 0.7,
            "pass_thresh": 0.35,
        })
    if args.dump_dir:
        cfg.dump_dir = args.dump_dir
    if args.logdir:
        cfg.logdir = args.logdir
    if args.steps:
        cfg.max_steps = args.steps
    AutoVLACoevolve(cfg).run()


if __name__ == "__main__":
    main()
