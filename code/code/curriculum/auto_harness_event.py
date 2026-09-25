#!/usr/bin/env python3
"""Auto-Harness *event* search (HarnessWAM-inspired) — parallel to auto_harness_search.

Aligned with production Auto (K=1, temp=0, fail/stall-only, paired J, holdout,
ACCEPT/REVERT, warm-start prior, summary.effective).

Optimize EventPolicy:
  mutate/compile → paired sandbox eval → ACCEPT/REVERT

  cd project_root
  # stub
  python -m curriculum.auto_harness_event \\
    --config curriculum/configs/auto_harness_event_stub.yaml

  # hard queue (mirrors run_auto_harness_queue)
  python -m curriculum.run_auto_harness_event_queue \\
    --out-root curriculum/outputs/auto_harness_event_hard_k1_t0 \\
    --only-hard-from curriculum/outputs/batch_openha_bare_k1_t0/hard_tasks.json \\
    --wait-bare
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from curriculum.harness_event_policy import (
    MUTATION_BANK,
    CompileError,
    EventPolicy,
    apply_bank_mutation,
    compile_edits,
    default_event_seed,
    strong_event_seed,
)
from curriculum.harness_event_runtime import EventHarnessRuntime


def _load_cfg(path: str) -> Dict[str, Any]:
    with open(path) as f:
        if path.endswith((".yaml", ".yml")) and yaml is not None:
            return yaml.safe_load(f) or {}
        return json.load(f)


def _runtime_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "backend",
        "vla_mode",
        "vla_model_path",
        "vla_device",
        "vla_dtype",
        "vla_temperature",
        "vla_do_sample",
        "max_steps",
        "action_chunk_len",
        "checkpoint_every",
        "ticks_per_action",
        "task_config",
        "img_save_dir",
        "success_reward_thresh",
        "soft_rollback",
        "instruction",
        "gemini_api_key",
        "gemini_temperature",
    )
    out: Dict[str, Any] = {}
    for k in keys:
        if k in cfg and cfg[k] is not None:
            out[k] = cfg[k]
    # K=1 greedy default alignment
    if "vla_temperature" in out and float(out["vla_temperature"]) <= 0:
        out["vla_temperature"] = 0.0
        out["vla_do_sample"] = False
    return out


def paired_decision(
    base: Dict[str, Any],
    cand: Dict[str, Any],
    *,
    accept_eps: float,
) -> Dict[str, Any]:
    """Paired Δ on per-seed success (and score fallback), same protocol as Auto."""
    b_eps = {
        int((e.get("meta") or {}).get("seed", i)): e
        for i, e in enumerate(base.get("episodes") or [])
    }
    c_eps = {
        int((e.get("meta") or {}).get("seed", i)): e
        for i, e in enumerate(cand.get("episodes") or [])
    }
    seeds = sorted(set(b_eps) & set(c_eps))
    d_succ = [float(c_eps[s]["success"]) - float(b_eps[s]["success"]) for s in seeds]
    mean_succ = sum(d_succ) / max(1, len(d_succ))
    score_b = float(base.get("score") or 0.0)
    score_c = float(cand.get("score") or 0.0)
    mean_score = score_c - score_b
    # Prefer success Δ; if flat, use score (matches Auto's score-based accept)
    primary = mean_succ if abs(mean_succ) > 1e-12 else mean_score
    keep = primary >= float(accept_eps)
    n_win = sum(1 for d in d_succ if d > 0)
    return {
        "keep": keep,
        "mean_delta": float(primary),
        "mean_succ_delta": float(mean_succ),
        "score_delta": float(mean_score),
        "deltas": d_succ,
        "n_win": n_win,
        "n_seeds": len(seeds),
        "seeds": seeds,
    }


class AutoEventSearch:
    """Search EventPolicy; sandbox judges; compiler gates illegal edits."""

    def __init__(
        self,
        *,
        logdir: str,
        seed: int = 0,
        n_rounds: int = 3,
        eval_seeds: Optional[Sequence[int]] = None,
        holdout_seeds: Optional[Sequence[int]] = None,
        accept_eps: float = 0.01,
        runtime_overrides: Optional[Dict[str, Any]] = None,
        policy: Optional[EventPolicy] = None,
        proposer_mode: str = "bank",  # bank | llm
        proposer_model: str = "gemini-2.5-flash",
        proposer_temperature: float = 0.0,
    ):
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self.rng = random.Random(int(seed))
        self.n_rounds = int(n_rounds)
        self.eval_seeds = list(eval_seeds or [101])
        self.holdout_seeds = list(holdout_seeds or [909])
        self.accept_eps = float(accept_eps)
        self.runtime_overrides = dict(runtime_overrides or {})
        self.policy = policy or strong_event_seed()
        self.proposer_mode = str(proposer_mode)
        self.proposer_model = str(proposer_model)
        self.proposer_temperature = float(proposer_temperature)
        self.history: List[Dict[str, Any]] = []
        self.best_policy = self.policy
        self.best_score = 0.0

    def _save_artifacts(self, policy: EventPolicy, prefix: str) -> None:
        policy.save(os.path.join(self.logdir, f"{prefix}_policy.json"))
        # Align with Auto queue promote path: also emit harness_best.json
        h = policy.to_harness_spec()
        # bind runtime overrides into projected harness
        for k, v in self.runtime_overrides.items():
            if hasattr(h.runtime, k):
                setattr(h.runtime, k, v)
        h.save(os.path.join(self.logdir, f"{prefix}_harness.json"))
        if prefix == "best":
            policy.save(os.path.join(self.logdir, "policy_best.json"))
            h.save(os.path.join(self.logdir, "harness_best.json"))

    def _eval(self, policy: EventPolicy, seeds: Sequence[int]) -> Dict[str, Any]:
        rt = EventHarnessRuntime(
            policy,
            seed=int(seeds[0]),
            harness_overrides=self.runtime_overrides,
        )
        return rt.evaluate_on_seeds(seeds, persist_memory=False)

    def _propose(self, base_metrics: Dict[str, Any]) -> Tuple[EventPolicy, Dict[str, Any]]:
        if self.proposer_mode == "llm":
            try:
                return self._propose_llm(base_metrics)
            except Exception as e:
                man = {"proposer": "llm_fallback_bank", "error": str(e)}
                mut = self.rng.choice(MUTATION_BANK)
                pol, report = apply_bank_mutation(self.policy, mut)
                man.update({"mutation": mut, "compile": report})
                return pol, man
        mut = self.rng.choice(MUTATION_BANK)
        pol, report = apply_bank_mutation(self.policy, mut)
        return pol, {"proposer": "bank", "mutation": mut, "compile": report}

    def _propose_llm(self, base_metrics: Dict[str, Any]) -> Tuple[EventPolicy, Dict[str, Any]]:
        from curriculum.harness_llm_proposer import extract_json_obj, llm_api_chat

        ev = base_metrics.get("event_counts") or {}
        prompt = (
            "Propose ONE EventPolicy patch for a frozen Minecraft VLA.\n"
            "Legal decisions: continue, observe, replan, recover, terminate.\n"
            "Legal edit paths: triggers.*, observe.*, replan.*, recover.*, "
            "belief.*, routing.*, prompts.*\n"
            "forbid_periodic must stay true. Prefer fail/stall routing edits.\n"
            "Never emit game keypresses.\n"
            f"Current routing: {self.policy.routing}\n"
            f"Current stall_steps: {self.policy.triggers.stall_steps}\n"
            f"Seed success_rate: {base_metrics.get('success_rate')}\n"
            f"Seed score: {base_metrics.get('score')}\n"
            f"Event counts: {ev}\n"
            "JSON field order: edits, claim, expect_delta, reason.\n"
        )
        text = llm_api_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You propose EventPolicy patches only. "
                        "Return ONE JSON object. Never emit game actions."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=self.proposer_model,
            temperature=self.proposer_temperature,
        )
        obj = extract_json_obj(text)
        edits = list(obj.get("edits") or [])
        pol, report = compile_edits(self.policy, edits)
        return pol, {"proposer": "llm", "raw": obj, "compile": report}

    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        self._save_artifacts(self.policy, "seed")
        base = self._eval(self.policy, self.eval_seeds)
        with open(os.path.join(self.logdir, "seed_eval.json"), "w") as f:
            json.dump(base, f, indent=2)

        self.best_policy = self.policy
        self.best_score = float(base.get("score") or 0.0)
        best_metrics = base
        accepted = 0
        reverted = 0
        proposer_failed = 0

        self.history.append({
            "round": -1,
            "decision": "SEED",
            "score": self.best_score,
            "success_rate": base.get("success_rate"),
            "event_counts": base.get("event_counts"),
            "fp": self.policy.fingerprint(),
        })

        for r in range(self.n_rounds):
            try:
                cand_pol, man = self._propose(best_metrics)
            except CompileError as e:
                proposer_failed += 1
                row = {
                    "round": r,
                    "decision": "REVERT",
                    "reason": f"compile_error:{e}",
                    "accepted": False,
                }
                self.history.append(row)
                print(f"[event-auto] r{r} REVERT compile: {e}", flush=True)
                continue

            cand = self._eval(cand_pol, self.eval_seeds)
            paired = paired_decision(best_metrics, cand, accept_eps=self.accept_eps)
            score = float(cand.get("score") or 0.0)
            keep = bool(paired["keep"])
            row = {
                "round": r,
                "decision": "ACCEPT" if keep else "REVERT",
                "accepted": keep,
                "mean_delta": paired["mean_delta"],
                "score": score,
                "score_delta": paired["score_delta"],
                "deltas": paired["deltas"],
                "n_win": paired["n_win"],
                "n_seeds": paired["n_seeds"],
                "cand_success": cand.get("success_rate"),
                "base_success": best_metrics.get("success_rate"),
                "cand_events": cand.get("event_counts"),
                "manifest": man,
                "cand_fp": cand_pol.fingerprint(),
            }
            if keep:
                accepted += 1
                self.best_policy = cand_pol
                self.best_score = score
                best_metrics = cand
                self.policy = cand_pol
                self._save_artifacts(cand_pol, "best")
                with open(os.path.join(self.logdir, "best_eval.json"), "w") as f:
                    json.dump(best_metrics, f, indent=2)
                print(
                    f"[event-auto] r{r} ACCEPT Δ={paired['mean_delta']:+.4f} "
                    f"score={score:.4f} win={paired['n_win']}/{paired['n_seeds']} "
                    f"events={cand.get('event_counts')}",
                    flush=True,
                )
            else:
                reverted += 1
                print(
                    f"[event-auto] r{r} REVERT Δ={paired['mean_delta']:+.4f} "
                    f"score={score:.4f} (falsified) events={cand.get('event_counts')}",
                    flush=True,
                )
            self.history.append(row)
            with open(os.path.join(self.logdir, "history.json"), "w") as f:
                json.dump(self.history, f, indent=2, default=str)
            with open(os.path.join(self.logdir, "history.jsonl"), "a") as f:
                f.write(json.dumps(row, default=str) + "\n")

        self._save_artifacts(self.best_policy, "best")
        hold = self._eval(self.best_policy, self.holdout_seeds)
        with open(os.path.join(self.logdir, "holdout_eval.json"), "w") as f:
            json.dump(hold, f, indent=2)

        seed_score = float(self.history[0]["score"])
        summary = {
            "mechanism_id": "training_free_event_harness_search",
            "seed_score": seed_score,
            "best_score": self.best_score,
            "holdout_score": hold.get("score"),
            "holdout_success_rate": hold.get("success_rate"),
            "eval_seeds": list(self.eval_seeds),
            "holdout_seeds": list(self.holdout_seeds),
            "accepted": accepted,
            "reverted": reverted,
            "proposer_failed": proposer_failed,
            "rounds": len(self.history) - 1,
            "best_harness_fp": self.best_policy.fingerprint(),
            "logdir": self.logdir,
            "warm_start_from": (self.policy.meta or {}).get("warm_start_from"),
            "effective": bool(
                accepted > 0 or float(self.best_score) > seed_score + 1e-9
            ),
            "seed_success": base.get("success_rate"),
            "best_success": best_metrics.get("success_rate"),
            "seed_events": base.get("event_counts"),
            "best_events": best_metrics.get("event_counts"),
            "holdout_events": hold.get("event_counts"),
            "elapsed_sec": round(time.time() - t0, 2),
        }
        with open(os.path.join(self.logdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"[event-auto] done best={self.best_score:.4f} "
            f"holdout_succ={hold.get('success_rate'):.3f} "
            f"accepted={accepted} reverted={reverted} "
            f"proposer_failed={proposer_failed}",
            flush=True,
        )
        return summary


def run_from_config(cfg_path: str) -> Dict[str, Any]:
    cfg = _load_cfg(cfg_path)
    start = str(cfg.get("start", "strong"))
    warm_from_file = False
    if start == "strong":
        pol = strong_event_seed()
    elif start == "default":
        pol = default_event_seed()
    elif os.path.isfile(start):
        # Warm-start: prefer EventPolicy JSON; else project from HarnessSpec if needed
        try:
            pol = EventPolicy.load(start)
        except Exception:
            # If prior is harness_best.json from event search, look sibling
            alt = start.replace("harness_best.json", "policy_best.json")
            if os.path.isfile(alt):
                pol = EventPolicy.load(alt)
            else:
                pol = strong_event_seed()
                print(f"[event-auto] prior not EventPolicy, using strong: {start}", flush=True)
        warm_from_file = True
        pol.name = f"warm_{os.path.basename(start)}"
        pol.meta = {**(pol.meta or {}), "warm_start_from": os.path.abspath(start)}
        print(f"[event-auto] warm-start from effective prior: {start}", flush=True)
    else:
        pol = strong_event_seed()

    preserve = bool(cfg.get("preserve_harness_body", warm_from_file))
    if not preserve:
        if "routing_on_stall" in cfg:
            pol.routing.on_stall = str(cfg["routing_on_stall"])
        if "routing_on_fail" in cfg:
            pol.routing.on_fail = str(cfg["routing_on_fail"])
        if "stall_steps" in cfg:
            pol.triggers.stall_steps = int(cfg["stall_steps"])
        if "observe_budget" in cfg:
            pol.observe.budget = int(cfg["observe_budget"])
        if cfg.get("probe_action_pool"):
            pol.observe.action_pool = [str(x) for x in list(cfg["probe_action_pool"])]
        if "recover_max_retries" in cfg:
            pol.recover.max_retries = int(cfg["recover_max_retries"])
        # fail/stall-only invariants (aligned with Auto queue cold start)
        pol.triggers.forbid_periodic = True
        pol.triggers.warmup_observes = int(cfg.get("warmup_probes", 0) or 0)
        if "probe_on_stall" in cfg:
            pol.triggers.on_stall = bool(cfg["probe_on_stall"])
        if "probe_on_fail" in cfg:
            pol.triggers.on_fail = bool(cfg["probe_on_fail"])

    search = AutoEventSearch(
        logdir=str(cfg.get("logdir") or "curriculum/outputs/auto_harness_event"),
        seed=int(cfg.get("seed", 0)),
        n_rounds=int(cfg.get("n_rounds", 3)),
        eval_seeds=list(cfg.get("eval_seeds") or [101]),
        holdout_seeds=list(cfg.get("holdout_seeds") or [909]),
        accept_eps=float(cfg.get("accept_eps", 0.01)),
        runtime_overrides=_runtime_overrides(cfg),
        policy=pol,
        proposer_mode=str(cfg.get("proposer_mode", "bank")),
        proposer_model=str(cfg.get("proposer_model") or "gemini-2.5-flash"),
        proposer_temperature=float(cfg.get("proposer_temperature", 0.0)),
    )
    return search.run()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Event-driven Auto-Harness search")
    ap.add_argument("--config", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)
    summary = run_from_config(args.config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
