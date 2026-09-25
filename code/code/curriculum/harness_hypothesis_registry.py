"""Hypothesis Registry for Autonomous Harness Research.

Each Researcher proposal is logged as a scientific trial:

  observe → diagnose → experiment → hypothesize → modify → evaluate → learn

Secondary metric (paper-facing):
  P(Researcher prediction is correct) ≈ fraction of trials where
  sign(actual_delta) matches sign(predicted_effect) when accepted/rejected.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence


def hypothesis_registry_path(out_root: str) -> str:
    return os.path.join(out_root, "hypothesis_registry.jsonl")


def new_hypothesis_id(round_id: int = 0) -> str:
    return f"H{int(round_id)}_{uuid.uuid4().hex[:8]}"


def open_hypothesis(
    *,
    round_id: int,
    task_file: str,
    diagnosis: str = "",
    why: str = "",
    proposal: str = "",
    claim: str = "",
    suspected_cause: str = "",
    failure_clusters: Optional[Sequence[str]] = None,
    evidence: Optional[Dict[str, Any]] = None,
    proposed_change: Optional[List[Dict[str, Any]]] = None,
    predicted_effect: Optional[float] = None,
    predicted_regression: str = "",
    parent_harness_fp: str = "",
    candidate_harness_fp: str = "",
) -> Dict[str, Any]:
    """Create a registry row at proposal time (before sandbox verdict)."""
    return {
        "hypothesis_id": new_hypothesis_id(round_id),
        "round": int(round_id),
        "task": task_file,
        "timestamp_open": time.time(),
        "failure_cluster": list(failure_clusters or []),
        "suspected_cause": str(suspected_cause or "")[:160],
        "diagnosis": str(diagnosis or "")[:320],
        "why": str(why or "")[:320],
        "proposal": str(proposal or claim or "")[:400],
        "claim": str(claim or "")[:160],
        "evidence": evidence or {},
        "proposed_change": proposed_change or [],
        "predicted_effect": (
            float(predicted_effect) if predicted_effect is not None else None
        ),
        "predicted_regression": str(predicted_regression or "")[:240],
        "parent_harness_fp": parent_harness_fp,
        "candidate_harness_fp": candidate_harness_fp,
        # filled on close
        "actual_effect": None,
        "actual_success_rate": None,
        "holdout_success_rate": None,
        "accepted": None,
        "prediction_correct": None,
        "timestamp_close": None,
    }


def close_hypothesis(
    row: Dict[str, Any],
    *,
    accepted: bool,
    actual_effect: float,
    actual_success_rate: Optional[float] = None,
    holdout_success_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """Attach sandbox outcomes; score whether Researcher prediction matched."""
    out = dict(row)
    out["accepted"] = bool(accepted)
    out["actual_effect"] = float(actual_effect)
    if actual_success_rate is not None:
        out["actual_success_rate"] = float(actual_success_rate)
    if holdout_success_rate is not None:
        out["holdout_success_rate"] = float(holdout_success_rate)
    out["timestamp_close"] = time.time()
    pred = out.get("predicted_effect")
    if pred is None:
        out["prediction_correct"] = None
    else:
        # Directional agreement: both improve, both worsen, or both ~0
        eps = 0.005
        pred_s = 0 if abs(float(pred)) < eps else (1 if float(pred) > 0 else -1)
        act_s = 0 if abs(float(actual_effect)) < eps else (1 if float(actual_effect) > 0 else -1)
        out["prediction_correct"] = bool(pred_s == act_s)
    return out


def append_hypothesis(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def load_hypotheses(path: str, *, limit: int = 500) -> List[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows[-max(1, int(limit)) :]


def researcher_calibration(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Secondary metric: how often predicted effect direction matches actual."""
    closed = [r for r in rows if r.get("accepted") is not None]
    scored = [r for r in closed if r.get("prediction_correct") is not None]
    n = len(scored)
    n_ok = sum(1 for r in scored if r.get("prediction_correct"))
    n_acc = sum(1 for r in closed if r.get("accepted"))
    return {
        "n_closed": len(closed),
        "n_scored": n,
        "n_accepted": n_acc,
        "prediction_accuracy": (n_ok / n) if n else None,
        "accept_rate": (n_acc / len(closed)) if closed else None,
    }
