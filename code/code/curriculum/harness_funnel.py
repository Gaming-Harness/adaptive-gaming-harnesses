"""Auto-Harness proposal funnel — not just “how many ACCEPTs”.

Four rates (plus invalid-reason breakdown):

  P(valid)              Gemini produced an applicable prompt patch
  P(ACCEPT | valid)     sandbox paired eval kept it
  P(holdout win | ACCEPT)  H* beats H0 on held-out episode seeds
  ΔSuccessRate          mean(best_sr − seed_sr) on searched tasks

This splits “won’t write a patch” vs “writes junk” vs “eval-only overfitting”.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional


INVALID_REASONS = ("api", "parse", "schema", "apply", "unmatched_bank", "other")


def _rate(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return float(num) / float(den)


def classify_proposer_error(exc: BaseException) -> str:
    msg = f"{type(exc).__name__}: {exc}".lower()
    if any(
        x in msg
        for x in ("llm_api", "http ", "network", "timeout", "urlerror", "429", "503", "502")
    ):
        return "api"
    if any(
        x in msg
        for x in ("json", "truncated", "no json object", "unrepairable")
    ):
        return "parse"
    if any(
        x in msg
        for x in (
            "missing diagnosis",
            "missing why",
            "missing proposal",
            "prompt/instruction",
            "knob-only",
            "not meaningful",
            "patch missing edits",
            "abandon/impossible",
            "executable behavior",
        )
    ):
        return "schema"
    if any(
        x in msg
        for x in (
            "frozen",
            "not editable",
            "no valid edits",
            "empty/invalid action_pool",
            "path too deep",
        )
    ):
        return "apply"
    return "other"


def funnel_from_counts(
    *,
    n_propose: int,
    n_valid: int,
    n_accept: int,
    n_task_accept: int = 0,
    n_holdout_win: int = 0,
    n_tasks: int = 0,
    sum_delta_sr: float = 0.0,
    invalid_reasons: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Build the 4-rate report. Counts may be round-level or mixed with task-level holdout."""
    p_valid = _rate(n_valid, n_propose)
    p_accept_given_valid = _rate(n_accept, n_valid)
    p_holdout_given_accept = _rate(n_holdout_win, n_task_accept)
    mean_delta_sr = (float(sum_delta_sr) / n_tasks) if n_tasks > 0 else None
    return {
        "n_propose": int(n_propose),
        "n_valid": int(n_valid),
        "n_accept": int(n_accept),
        "n_task_accept": int(n_task_accept),
        "n_holdout_win": int(n_holdout_win),
        "n_tasks": int(n_tasks),
        "p_valid": p_valid,
        "p_accept_given_valid": p_accept_given_valid,
        "p_holdout_given_accept": p_holdout_given_accept,
        "mean_delta_success_rate": mean_delta_sr,
        "invalid_reasons": dict(invalid_reasons or {}),
        "readout": {
            "proposal_validity": "Gemini writes an applicable prompt patch",
            "eval_acceptance": "sandbox keeps a valid patch (paired eval)",
            "holdout_generalization": "H* beats H0 on holdout seeds, given ACCEPT",
            "final_delta_sr": "mean(best_sr - seed_sr) over searched tasks",
        },
    }


def funnel_from_task_summary(s: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer explicit funnel fields; reconstruct from old summaries if needed."""
    f = s.get("funnel") if isinstance(s.get("funnel"), dict) else {}
    n_propose = int(f.get("n_propose") if f.get("n_propose") is not None else (
        int(s.get("accepted") or 0)
        + int(s.get("reverted") or 0)
        + int(s.get("proposer_failed") or 0)
    ))
    n_valid = int(f.get("n_valid") if f.get("n_valid") is not None else (
        int(s.get("accepted") or 0) + int(s.get("reverted") or 0)
    ))
    n_accept = int(f.get("n_accept") if f.get("n_accept") is not None else int(s.get("accepted") or 0))
    reasons = dict(f.get("invalid_reasons") or {})
    if not reasons and int(s.get("proposer_failed") or 0):
        reasons = {"legacy_proposer_failed": int(s.get("proposer_failed") or 0)}

    seed_sr = float(s.get("seed_success_rate") if s.get("seed_success_rate") is not None else (
        # old summaries only stored scores; leave 0 if missing
        0.0
    ))
    if "seed_success_rate" not in s and s.get("seed_score") is not None:
        # score ≈ sr + tiny reward term; not a success rate. mark unknown via None later.
        seed_sr = None  # type: ignore
    best_sr = s.get("best_success_rate")
    if best_sr is None:
        best_sr = None
    else:
        best_sr = float(best_sr)
    delta_sr = s.get("delta_success_rate")
    if delta_sr is None and best_sr is not None and seed_sr is not None:
        delta_sr = float(best_sr) - float(seed_sr)

    task_accept = bool(n_accept > 0)
    holdout_win = s.get("holdout_win")
    if holdout_win is None:
        # Legacy proxy (no H0 holdout): ACCEPT and holdout success > 0.
        holdout_win = bool(task_accept and float(s.get("holdout_success_rate") or 0.0) > 0.0)
        holdout_legacy = True
    else:
        holdout_win = bool(holdout_win)
        holdout_legacy = bool(f.get("holdout_legacy") or s.get("holdout_legacy"))

    return {
        "n_propose": n_propose,
        "n_valid": n_valid,
        "n_accept": n_accept,
        "task_accept": task_accept,
        "holdout_win": bool(holdout_win),
        "holdout_legacy": holdout_legacy,
        "delta_success_rate": delta_sr,
        "seed_success_rate": seed_sr,
        "best_success_rate": best_sr,
        "holdout_success_rate": s.get("holdout_success_rate"),
        "holdout_seed_success_rate": s.get("holdout_seed_success_rate"),
        "invalid_reasons": reasons,
    }


def aggregate_task_funnels(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r for r in rows if r]
    n_propose = sum(int(r.get("n_propose") or 0) for r in rows)
    n_valid = sum(int(r.get("n_valid") or 0) for r in rows)
    n_accept = sum(int(r.get("n_accept") or 0) for r in rows)
    n_task_accept = sum(1 for r in rows if r.get("task_accept"))
    n_holdout_win = sum(1 for r in rows if r.get("task_accept") and r.get("holdout_win"))
    deltas = [float(r["delta_success_rate"]) for r in rows if r.get("delta_success_rate") is not None]
    reasons: Counter[str] = Counter()
    for r in rows:
        for k, v in (r.get("invalid_reasons") or {}).items():
            reasons[str(k)] += int(v or 0)
    n_legacy = sum(1 for r in rows if r.get("holdout_legacy"))
    out = funnel_from_counts(
        n_propose=n_propose,
        n_valid=n_valid,
        n_accept=n_accept,
        n_task_accept=n_task_accept,
        n_holdout_win=n_holdout_win,
        n_tasks=len(rows),
        sum_delta_sr=float(sum(deltas)) if deltas else 0.0,
        invalid_reasons=dict(reasons),
    )
    if deltas:
        out["mean_delta_success_rate"] = float(sum(deltas) / len(deltas))
        out["n_tasks_with_delta_sr"] = len(deltas)
    out["n_holdout_legacy_proxy"] = n_legacy
    return out


def scan_out_root(out_root: str) -> Dict[str, Any]:
    """Aggregate funnel over search_*/summary.json under an Auto queue root."""
    rows: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    if not out_root or not os.path.isdir(out_root):
        return {"out_root": out_root, "funnel": funnel_from_counts(
            n_propose=0, n_valid=0, n_accept=0
        ), "tasks": []}
    for name in sorted(os.listdir(out_root)):
        if not name.startswith("search_"):
            continue
        sp = os.path.join(out_root, name, "summary.json")
        if not os.path.isfile(sp):
            continue
        try:
            s = json.load(open(sp))
        except Exception:
            continue
        row = funnel_from_task_summary(s)
        row["file"] = name.replace("search_", "") + ".json"
        row["logdir"] = os.path.join(out_root, name)
        rows.append(row)
        tasks.append(row)
    funnel = aggregate_task_funnels(rows)
    funnel["out_root"] = os.path.abspath(out_root)
    return {"out_root": os.path.abspath(out_root), "funnel": funnel, "tasks": tasks}


def format_funnel(funnel: Dict[str, Any]) -> str:
    def pct(x: Optional[float]) -> str:
        if x is None:
            return "n/a"
        return f"{100.0 * float(x):.1f}%"

    lines = [
        "Auto-Harness funnel",
        f"  Proposal validity     P(valid)                 "
        f"{funnel.get('n_valid')}/{funnel.get('n_propose')} = {pct(funnel.get('p_valid'))}",
        f"  Eval acceptance       P(ACCEPT | valid)        "
        f"{funnel.get('n_accept')}/{funnel.get('n_valid')} = {pct(funnel.get('p_accept_given_valid'))}",
        f"  Holdout generalization P(holdout win | ACCEPT)  "
        f"{funnel.get('n_holdout_win')}/{funnel.get('n_task_accept')} = {pct(funnel.get('p_holdout_given_accept'))}",
        f"  Final task ΔSR        mean(best_sr - seed_sr)  "
        f"{funnel.get('mean_delta_success_rate')}",
    ]
    reasons = funnel.get("invalid_reasons") or {}
    if reasons:
        bits = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()) if v)
        lines.append(f"  invalid reasons       {bits}")
    if funnel.get("n_holdout_legacy_proxy"):
        lines.append(
            f"  note: {funnel['n_holdout_legacy_proxy']} tasks used legacy "
            f"holdout proxy (ACCEPT & holdout_sr>0), not paired H0 vs H*"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Report Auto-Harness proposal funnel")
    ap.add_argument("--out-root", required=True, help="Auto queue directory")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = scan_out_root(args.out_root)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return
    print(format_funnel(report["funnel"]))
    n = len(report["tasks"])
    print(f"  tasks with summary    {n}")


if __name__ == "__main__":
    main()
