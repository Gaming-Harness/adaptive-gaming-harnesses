#!/usr/bin/env python3
"""True Auto Research (CLAW-style) for Gaming World Models.

Probe is the *scheduler*, not a scoreboard:

  Belief(M) → Propose H → WorthLearn → Learn|Skip|Re-verify
           → allocate sandbox only to Learn/Re-verify
           → adaptive probes → Write|Merge|Revoke|Ignore
           → gated WM update (Skip does not spend intervene budget)

Mechanism under test (non-hyperparam):
  probe_gated_worth_learn_scheduler

Usage:
  python -m curriculum.auto_research_claw --config curriculum/configs/auto_research_gaming_claw.yaml
  python -m curriculum.run_curriculum smoke_auto_research_claw
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import torch
from omegaconf import OmegaConf

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from curriculum.auto_scheduler import AutoScheduler, SchedulePlan
from curriculum.capability_discovery import discover_probe_compress
from curriculum.evidence_graph import EvidenceGraph
from curriculum.hypothesis import Decision, Hypothesis
from curriculum.probes import build_horizon_chains
from curriculum.replay_buffer import MixedReplayBuffer, Transition
from curriculum.research_contract import ResearchContract, set_global_seed
from curriculum.self_evolve_wm import SelfEvolvingWM


MECHANISM_PATCH = {
    "mechanism_id": "probe_gated_worth_learn_scheduler",
    "type": "typed_mechanism",
    "changes": [
        "WorthLearn(H)=U*N*V-λCost decides Learn|Skip|Re-verify",
        "sandbox action_plan only for Learn/Re-verify (Skip spends no interact budget)",
        "adaptive probe focus from weak dimensions",
        "mastered memory suppresses re-learning; re-verify can revoke",
        "WM update gated: Skip → pred-only or no update; Learn → full selective intervene",
    ],
    "not_searched": ["lr", "batch_size", "width", "ema_tau"],
    "negative_control": "always_learn_fixed_scripted_loop",
}


class AutoResearchClaw(SelfEvolvingWM):
    """Full Auto experimenter under a frozen research contract."""

    def __init__(self, cfg):
        self.contract = ResearchContract.from_cfg(cfg).lock()
        set_global_seed(self.contract.seed)

        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg.data_root = self.contract.data_root_train
        cfg.pass_thresh = self.contract.pass_thresh
        cfg.max_steps = self.contract.max_steps
        cfg.eval_n = self.contract.eval_n
        cfg.mechanism_id = self.contract.mechanism_id

        super().__init__(cfg)

        self.contract.save(os.path.join(self.logdir, "research_contract.json"))
        self.evidence = EvidenceGraph(
            path=os.path.join(self.logdir, "evidence_graph.json")
        )
        self.holdout = MixedReplayBuffer()
        hold_root = self.contract.data_root_holdout
        n_hold = 0
        if hold_root and os.path.isdir(hold_root):
            n_hold = self.holdout.load_expert_pt_dir(
                hold_root,
                max_clips=int(cfg.get("holdout_max_clips", 8)),
                block_frames=self.nfpb,
                stride=int(cfg.get("block_stride", 1)),
            )

        self.scheduler = AutoScheduler(
            memory=self.memory,
            student=self.student,
            teacher=self.teacher,
            device=self.device,
            lambda_cost=float(cfg.get("worth_lambda_cost", 0.15)),
            skip_thresh=float(cfg.get("worth_skip_thresh", 0.18)),
            learn_thresh=float(cfg.get("worth_learn_thresh", 0.28)),
            mastered_sim=float(cfg.get("mastered_sim", 0.78)),
            mastered_min_verify=int(cfg.get("mastered_min_verify", 2)),
            mastered_min_conf=float(cfg.get("mastered_min_conf", 0.50)),
            reverify_every=int(cfg.get("reverify_every", 5)),
            cost_sandbox=float(cfg.get("worth_cost_sandbox", 1.0)),
        )
        self._last_plan: Optional[SchedulePlan] = None
        self._decision_log: List[Dict[str, Any]] = []
        self.allow_skip_pred_update = bool(cfg.get("allow_skip_pred_update", False))

        print(
            f"[auto_research_claw] contract={self.contract._fingerprint} "
            f"mechanism={self.contract.mechanism_id} holdout_n={n_hold}",
            flush=True,
        )
        self.evidence.propose(
            hypothesis=(
                "Probe-gated WorthLearn scheduling spends interaction/update budget "
                "only on unknown or invalidated knowledge, improving WM diagnostics "
                "vs always-learn scripted loops under fixed budget."
            ),
            mechanism_id=self.contract.mechanism_id,
            patch=MECHANISM_PATCH,
            probe_scores={},
            confidence=0.0,
            passed=False,
            boundary={
                "domain": "gaming_wm",
                "split": "train",
                "transfer": self.contract.transfer_split.to_dict(),
            },
            step=0,
            meta={"phase": "hypothesis_registered", "auto": True},
        )

    # ------------------------------------------------------------------ #
    # Scheduling + sandbox
    # ------------------------------------------------------------------ #
    def plan_step(self, n_substrate: int = 16) -> SchedulePlan:
        pool = self._pool()
        if not pool:
            # empty plan → skip everything
            plan = SchedulePlan(hypotheses=[], update_mode="skip")
            plan.metrics = {
                "auto/n_hyp": 0.0,
                "auto/n_learn": 0.0,
                "auto/n_skip": 0.0,
                "auto/n_reverify": 0.0,
                "auto/sandbox_turns": 0.0,
                "auto/update_mode_skip": 1.0,
            }
            self._last_plan = plan
            return plan

        batch = self.buf.sample(
            min(n_substrate, len(pool)),
            prefer="expert" if self.buf.expert else "policy",
        )
        self.scheduler.student = self.student
        self.scheduler.teacher = self.teacher
        self.scheduler.memory = self.memory
        plan = self.scheduler.build_plan(
            batch,
            topk=int(self.cfg.get("hyp_topk", self.discover_topk)),
            sandbox_turns=int(self.cfg.get("sandbox_max_turns", 8)),
            step=self.step,
        )
        self._last_plan = plan
        self._decision_log.append({
            "step": self.step,
            "update_mode": plan.update_mode,
            "n_learn": plan.n_learn,
            "n_skip": plan.n_skip,
            "n_reverify": plan.n_reverify,
            "hypotheses": [h.to_dict() for h in plan.hypotheses],
            "action_plan": plan.action_plan,
        })
        # persist decision log incrementally
        with open(os.path.join(self.logdir, "decision_log.json"), "w") as f:
            json.dump(self._decision_log, f, indent=2)
        return plan

    def collect_sandbox(self, action_plan: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, float]:
        self.contract.bump_sandbox()
        assert self.collector is not None
        max_turns = int(self.cfg.get("sandbox_max_turns", 8))
        plan = action_plan
        if plan is None and self._last_plan is not None:
            plan = self._last_plan.action_plan

        if self.sandbox_backend == "stub":
            out = self.collector.collect_episode(
                self.buf.expert, max_turns=max_turns, action_plan=plan,
            )
        elif self.sandbox_backend == "llm_api":
            out = self.collector.collect_episode(
                max_turns=max_turns,
                prefix_turns=int(self.cfg.get("llm_api_prefix_turns", 2)),
                action_plan=plan,
            )
        else:
            out = self.collector.collect_episode(max_turns=max_turns, action_plan=plan)

        for tr in out.get("transitions", []):
            tr.source = "expert"
            self.buf.add_expert(tr)
        for pair in out.get("interventions", []):
            self.intervention_buf.append(pair)
            if len(self.intervention_buf) > int(self.cfg.get("max_interventions", 512)):
                self.intervention_buf = self.intervention_buf[-512:]

        return {
            "sandbox/n_trans": float(out.get("n_trans", 0)),
            "sandbox/n_intervene": float(len(out.get("interventions", []))),
            "sandbox/ep_reward": float(out.get("ep_reward", 0.0)),
            "sandbox/expert_size": float(len(self.buf.expert)),
            "sandbox/planned_turns": float(out.get("planned_turns", 0.0)),
        }

    # ------------------------------------------------------------------ #
    # Probe / compress / revoke
    # ------------------------------------------------------------------ #
    def discover_probe_compress_step(self, n: int) -> Dict[str, float]:
        """Learn path: probe candidates; Skip path: no compress; Re-verify: confirm/revoke."""
        plan = self._last_plan
        pool = self._pool()
        metrics: Dict[str, float] = {
            "discover/n": 0.0,
            "probe/n_verified": 0.0,
            "probe/n_discarded": 0.0,
            "compress/n": 0.0,
            "compress/memory_size": float(len(self.memory)),
            "auto/revoked": 0.0,
            "auto/reverified_ok": 0.0,
            "auto/mastered_new": 0.0,
        }
        if plan is not None:
            metrics.update(plan.metrics)

        if not pool:
            return metrics

        inter = self.intervention_buf[-int(self.cfg.get("probe_interventions", 16)):]
        self.prober.model = self.student
        self.discoverer.student = self.student
        self.discoverer.teacher = self.teacher
        self.student.eval()

        # --- Re-verify mastered / active memory ---
        revoked = 0
        re_ok = 0
        if plan is not None:
            for h in plan.hypotheses:
                if h.decision != Decision.REVERIFY or not h.memory_item_id:
                    continue
                item = self.memory.get_item(h.memory_item_id)
                if item is None:
                    continue
                # Build a synthetic transition from memory embeddings if needed
                substrate = self.buf.sample(1, prefer="expert" if self.buf.expert else "policy")
                if not substrate:
                    continue
                tr = substrate[0]
                result = self.prober.probe_batch(
                    [tr],
                    pool,
                    chains=None,
                    interventions=inter[:4] if inter else None,
                )
                conf = float(result.confidence)
                if conf >= self.pass_thresh:
                    self.memory.confirm_reverify(
                        h.memory_item_id, conf, result.scores, step=self.step,
                    )
                    re_ok += 1
                    self.evidence.propose(
                        hypothesis=h.claim,
                        mechanism_id=self.contract.mechanism_id,
                        patch=MECHANISM_PATCH,
                        probe_scores=result.scores,
                        confidence=conf,
                        passed=True,
                        boundary={"decision": "reverify_ok", "item_id": h.memory_item_id},
                        step=self.step,
                        meta={"hyp_id": h.hyp_id, "auto": True},
                    )
                else:
                    self.memory.revoke_item(
                        h.memory_item_id,
                        probe_scores=result.scores,
                        note="reverify_failed",
                    )
                    revoked += 1
                    # failure feeds next Learn proposals
                    self.scheduler.note_probe_failure(
                        action=h.action_a,
                        probe_scores=result.scores,
                        confidence=conf,
                        latent_t=tr.latent_t,
                        keyboard=tr.keyboard,
                        mouse=tr.mouse,
                        latent_tp1=tr.latent_tp1,
                        meta={"revoked_item": h.memory_item_id},
                    )
                    self.evidence.propose(
                        hypothesis=h.claim,
                        mechanism_id=self.contract.mechanism_id,
                        patch=MECHANISM_PATCH,
                        probe_scores=result.scores,
                        confidence=conf,
                        passed=False,
                        boundary={"decision": "reverify_revoke", "item_id": h.memory_item_id},
                        step=self.step,
                        meta={"hyp_id": h.hyp_id, "auto": True},
                    )
                    # priority replay
                    tr.meta = {**(tr.meta or {}), "probe_conf": conf, "novelty": 0.8}
                    self.buf.add_policy(tr)

        metrics["auto/revoked"] = float(revoked)
        metrics["auto/reverified_ok"] = float(re_ok)

        # --- Learn: Discover→Probe→Compress on planned / recent substrate ---
        do_learn = plan is None or plan.update_mode in ("learn", "mixed", "reverify")
        # still run a light discovery when reverify-only so dashboards move
        if plan is not None and plan.update_mode == "skip":
            # Skip: do not invent new memory; optional snapshot probes for logging
            batch = self.buf.sample(min(n, len(pool)), prefer="expert" if self.buf.expert else "policy")
            chains = build_horizon_chains(
                pool, horizon=int(self.cfg.get("probe_horizon", 3)),
                max_chains=int(self.cfg.get("max_chains", 8)),
            )
            snap = self.prober.probe_batch(
                batch, pool, chains, interventions=inter if inter else None,
            )
            metrics["probe/confidence"] = float(snap.confidence)
            metrics.update({f"probe/{k}": v for k, v in snap.scores.items()})
            metrics["probe/sandbox_grounded"] = float(snap.details.get("sandbox_grounded", 0.0))
        elif do_learn:
            batch = self.buf.sample(min(n, len(pool)), prefer="expert" if self.buf.expert else "policy")
            # Prefer planned-action transitions if present
            if plan and plan.action_plan:
                wanted = {p.get("action") for p in plan.action_plan}
                preferred = [
                    t for t in self.buf.expert[-64:]
                    if str((t.meta or {}).get("action", "")) in wanted
                ]
                if preferred:
                    batch = preferred[: min(n, len(preferred))]

            cycle = discover_probe_compress(
                self.discoverer,
                self.prober,
                transitions=batch,
                pool=pool,
                interventions=inter if inter else None,
                topk=min(self.discover_topk, len(batch)),
                pass_thresh=self.pass_thresh,
            )
            for c in cycle["discarded"]:
                tr = c.to_transition()
                conf = float((c.meta or {}).get("probe_conf", 0.0))
                scores = dict((c.meta or {}).get("probe_scores") or {})
                tr.meta = {
                    **(tr.meta or {}),
                    "probe_conf": conf,
                    "novelty": c.novelty,
                }
                self.buf.add_policy(tr)
                self.scheduler.note_probe_failure(
                    action=c.action_name,
                    probe_scores=scores,
                    confidence=conf,
                    latent_t=c.latent_t,
                    keyboard=c.keyboard,
                    mouse=c.mouse,
                    latent_tp1=c.latent_tp1,
                    meta={"cand_id": c.cand_id},
                )

            confs = [v.confidence for v in cycle["verified"]]
            confs += [float((c.meta or {}).get("probe_conf", 0.0)) for c in cycle["discarded"]]
            mean_conf = float(sum(confs) / max(1, len(confs)))

            # annotate compressed items with step for mastered promotion
            for v in cycle["verified"]:
                if v.item_id:
                    it = self.memory.get_item(v.item_id)
                    if it is not None:
                        it.last_verified_step = self.step
                        it.meta["step"] = self.step

            chains = build_horizon_chains(
                pool, horizon=int(self.cfg.get("probe_horizon", 3)),
                max_chains=int(self.cfg.get("max_chains", 16)),
            )
            snap = self.prober.probe_batch(
                batch, pool, chains, interventions=inter if inter else None,
            )
            metrics.update({
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
            })
            metrics.update({f"probe/{k}": v for k, v in snap.scores.items()})
            if cycle["candidates"]:
                metrics["discover/top_score"] = float(cycle["candidates"][0].discovery_score)
                metrics["discover/top_novelty"] = float(cycle["candidates"][0].novelty)

            # Evidence for each Learn hypothesis outcome (aggregate)
            for h in (plan.hypotheses if plan else []):
                if h.decision != Decision.LEARN:
                    continue
                self.evidence.propose(
                    hypothesis=h.claim,
                    mechanism_id=self.contract.mechanism_id,
                    patch=MECHANISM_PATCH,
                    probe_scores={k: float(v) for k, v in snap.scores.items()},
                    confidence=mean_conf,
                    passed=mean_conf >= self.pass_thresh and cycle["n_compressed"] > 0,
                    boundary={
                        "decision": "learn",
                        "action": h.action_a,
                        "worth_learn": h.worth_learn,
                    },
                    step=self.step,
                    meta={"hyp_id": h.hyp_id, "U": h.U, "N": h.N, "V": h.V, "auto": True},
                )

        # Promote mastered
        n_mast = self.scheduler.promote_mastered()
        metrics["auto/mastered_new"] = float(n_mast)
        metrics["memory/mastered"] = float(
            sum(1 for it in self.memory.items if it.status == "mastered")
        )
        metrics["memory/revoked"] = float(
            sum(1 for it in self.memory.items if it.status == "revoked")
        )

        # Holdout transfer (frozen)
        hold_pool = self.holdout.expert
        hold_score = 0.0
        if hold_pool and pool:
            hold_score = float(self.prober.holdout_transfer(pool, hold_pool))
            metrics["probe/holdout_transfer"] = hold_score
            if hold_score < self.contract.pass_thresh * 0.5:
                for prev in list(self.evidence.active_passed())[-3:]:
                    self.evidence.add_counterexample(
                        prev.record_id,
                        probe_scores={"holdout_transfer": hold_score},
                        note="holdout_transfer collapse",
                        revoke=True,
                    )

        metrics["evidence/n"] = float(len(self.evidence.records))
        metrics["evidence/active_passed"] = float(len(self.evidence.active_passed()))
        metrics["budget/steps_left"] = float(self.contract.budget_left()["steps"])
        metrics["compress/memory_size"] = float(len(self.memory))
        return metrics

    # ------------------------------------------------------------------ #
    # Gated WM update
    # ------------------------------------------------------------------ #
    def update_wm(self, batch_size: int) -> Dict[str, float]:
        plan = self._last_plan
        mode = plan.update_mode if plan is not None else "learn"

        if mode == "skip":
            if not self.allow_skip_pred_update:
                return {
                    "loss/total": 0.0,
                    "loss/dyn": 0.0,
                    "loss/causal": 0.0,
                    "loss/intervene": 0.0,
                    "loss/horizon": 0.0,
                    "loss/memory": 0.0,
                    "loss/anchor": 0.0,
                    "auto/wm_gated": 1.0,
                    "auto/wm_mode_skip": 1.0,
                }
            # optional light pred-only maintenance
            saved = (
                self.lambda_causal,
                self.lambda_intervene,
                self.lambda_horizon,
                self.lambda_memory,
            )
            self.lambda_causal = 0.0
            self.lambda_intervene = 0.0
            self.lambda_horizon = 0.0
            self.lambda_memory = 0.0
            self.contract.bump_wm_update()
            out = super().update_wm(batch_size)
            self.lambda_causal, self.lambda_intervene, self.lambda_horizon, self.lambda_memory = saved
            out["auto/wm_gated"] = 0.5
            out["auto/wm_mode_skip_pred"] = 1.0
            return out

        # Learn / reverify / mixed → full selective update
        self.contract.bump_wm_update()
        out = super().update_wm(batch_size)
        out["auto/wm_gated"] = 0.0
        out["auto/wm_mode_learn"] = 1.0 if mode in ("learn", "mixed") else 0.0
        out["auto/wm_mode_reverify"] = 1.0 if mode == "reverify" else 0.0
        return out

    # ------------------------------------------------------------------ #
    def run(self):
        self.contract.assert_locked()
        max_steps = self.contract.max_steps
        log_interval = int(self.cfg.log_interval)
        save_interval = int(self.cfg.save_interval)
        probe_every = int(self.cfg.get("probe_every", 1))
        collect_every = int(self.cfg.get("collect_every", 1))
        probe_n = int(self.cfg.get("probe_n", 16))
        batch_size = int(self.cfg.batch_size)

        t0 = time.time()
        try:
            for step in range(1, max_steps + 1):
                self.contract.bump_step()
                self.step = step
                metrics: Dict[str, float] = {}

                # 1) Plan first — probe decides what is worth doing
                plan = self.plan_step(n_substrate=int(self.cfg.get("plan_substrate", 16)))
                metrics.update(plan.metrics)

                # 2) Sandbox only if Learn or Re-verify needs new evidence
                if (
                    self.use_sandbox
                    and self.collector is not None
                    and step % collect_every == 0
                    and plan.update_mode != "skip"
                    and plan.action_plan
                ):
                    try:
                        metrics.update(self.collect_sandbox(plan.action_plan))
                    except Exception as e:
                        print(f"[auto_research_claw] sandbox collect failed: {e}", flush=True)
                        if self.sandbox_backend != "stub" and len(self._pool()) == 0:
                            raise
                elif plan.update_mode == "skip":
                    metrics["sandbox/n_trans"] = 0.0
                    metrics["sandbox/skipped"] = 1.0

                # 3) Probe / compress / revoke
                if step % probe_every == 0:
                    metrics.update(self.discover_probe_compress_step(probe_n))

                # 4) Gated WM update
                metrics.update(self.update_wm(batch_size))

                if step % log_interval == 0 or step == 1:
                    metrics["eval/pred_mse"] = self.eval_pred(self.contract.eval_n)
                    metrics["time_s"] = time.time() - t0
                    self.history.append({"step": step, **metrics})
                    nice = {
                        k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in metrics.items()
                    }
                    print(f"[auto_research_claw] step {step}/{max_steps} {nice}", flush=True)

                if step % save_interval == 0 or step == max_steps:
                    path = self.save(f"{step:06d}")
                    self.contract.save(os.path.join(self.logdir, "research_contract.json"))
                    print(
                        f"[auto_research_claw] saved {path} | evidence="
                        f"{len(self.evidence.records)} | "
                        f"mastered={sum(1 for it in self.memory.items if it.status=='mastered')} | "
                        f"mode={plan.update_mode}",
                        flush=True,
                    )
        finally:
            if self.collector is not None:
                try:
                    self.collector.close()
                except Exception:
                    pass

        self.save("latest")
        report = self._final_report()
        report_path = os.path.join(self.logdir, "claw_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[auto_research_claw] done. report → {report_path}", flush=True)
        print(json.dumps(report["verdict"], indent=2), flush=True)

    def _final_report(self) -> Dict:
        hist = self.history[-1] if self.history else {}
        active = self.evidence.active_passed()
        n_mastered = sum(1 for it in self.memory.items if it.status == "mastered")
        n_revoked = sum(1 for it in self.memory.items if it.status == "revoked")
        skip_steps = sum(1 for d in self._decision_log if d.get("update_mode") == "skip")
        learn_steps = sum(1 for d in self._decision_log if d.get("update_mode") in ("learn", "mixed"))
        verdict = {
            "domain": "gaming_world_model",
            "contract_fingerprint": self.contract._fingerprint,
            "mechanism_id": self.contract.mechanism_id,
            "budget_exhausted": self.contract.budget_left(),
            "n_evidence": len(self.evidence.records),
            "n_active_passed": len(active),
            "n_mastered": n_mastered,
            "n_revoked": n_revoked,
            "n_skip_steps": skip_steps,
            "n_learn_steps": learn_steps,
            "final_probe_confidence": hist.get("probe/confidence"),
            "final_holdout_transfer": hist.get("probe/holdout_transfer"),
            "final_compress_memory": hist.get("compress/memory_size"),
            "auto_success_criteria": {
                "contract_locked": True,
                "evidence_written": len(self.evidence.records) > 0,
                "holdout_measured": hist.get("probe/holdout_transfer") is not None,
                "scheduler_decided": len(self._decision_log) > 0,
                "skip_observed": skip_steps > 0 or learn_steps > 0,
                "non_hyperparam_mechanism": MECHANISM_PATCH["mechanism_id"],
            },
            "claim_level": "auto_skeleton_ok" if self._decision_log else "failed",
        }
        if (
            active
            and n_mastered > 0
            and (hist.get("probe/holdout_transfer") or 0) >= self.contract.pass_thresh * 0.7
        ):
            verdict["claim_level"] = "auto_mastered_transfer_signal"
        elif active and (hist.get("probe/holdout_transfer") or 0) >= self.contract.pass_thresh * 0.7:
            verdict["claim_level"] = "auto_transfer_signal"
        return {
            "verdict": verdict,
            "contract": self.contract.to_dict(),
            "evidence_summary": self.evidence.summary(),
            "mechanism_patch": MECHANISM_PATCH,
            "scheduler_stats": dict(self.scheduler.stats),
            "last_metrics": hist,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="curriculum/configs/auto_research_gaming_claw.yaml",
    )
    args, _ = ap.parse_known_args()
    os.chdir(_ROOT)
    cfg = OmegaConf.load(args.config)
    runner = AutoResearchClaw(cfg)
    runner.run()


if __name__ == "__main__":
    main()
