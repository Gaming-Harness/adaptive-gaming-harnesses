#!/usr/bin/env python3
"""Summarize causal K--P ablation metrics from per-episode event logs."""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List


ORDER = ("probe_only", "no_k_to_p", "no_p_to_k", "full")


def _safe_mean(values: List[float]) -> Any:
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(values) / len(values) if values else None


def _load_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return rows
    with open(path) as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def summarize_variant(root: str, variant: str) -> Dict[str, Any]:
    rows = _load_rows(os.path.join(root, variant, "results.jsonl"))
    probes = [event for row in rows for event in (row.get("probe_events") or [])]
    knowledge = [
        event for row in rows for event in (row.get("knowledge_events") or [])
    ]
    probed_rows = [row for row in rows if int(row.get("n_probe") or 0) > 0]
    resolved = [row for row in probed_rows if row.get("success")]
    useful_probes = [
        event for event in probes if float(event.get("immediate_utility") or 0.0) > 0.0
    ]
    useful_knowledge = [
        event
        for event in knowledge
        if float(event.get("immediate_utility") or 0.0) > 0.0
    ]
    by_type: Dict[str, Dict[str, int]] = {}
    for row in rows:
        family = str(row.get("task_type") or "unknown")
        bucket = by_type.setdefault(family, {"n": 0, "success": 0})
        bucket["n"] += 1
        bucket["success"] += int(bool(row.get("success")))
    result: Dict[str, Any] = {
        "variant": variant,
        "n": len(rows),
        "success": sum(int(bool(row.get("success"))) for row in rows),
        "success_rate": (
            sum(int(bool(row.get("success"))) for row in rows) / len(rows)
            if rows
            else None
        ),
        "by_type": by_type,
        "episodes_with_failure_probe": len(probed_rows),
        "resolved_failure_episodes": len(resolved),
        "failure_resolution_rate": (
            len(resolved) / len(probed_rows) if probed_rows else None
        ),
        "n_probes": len(probes),
        "probes_per_task": len(probes) / len(rows) if rows else None,
        "probes_per_resolved_failure": (
            sum(int(row.get("n_probe") or 0) for row in resolved) / len(resolved)
            if resolved
            else None
        ),
        "steps_to_recovery": _safe_mean(
            [float(row.get("steps")) for row in resolved if row.get("steps") is not None]
        ),
        "mean_probe_utility": _safe_mean(
            [float(event.get("immediate_utility") or 0.0) for event in probes]
        ),
        "useful_probe_fraction": (
            len(useful_probes) / len(probes) if probes else None
        ),
        "n_knowledge_selections": len(knowledge),
        "knowledge_applicability": (
            len(useful_knowledge) / len(knowledge) if knowledge else None
        ),
        "mean_knowledge_utility": _safe_mean(
            [float(event.get("immediate_utility") or 0.0) for event in knowledge]
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    rows = [summarize_variant(root, variant) for variant in ORDER]
    baseline = next(
        (row["success"] for row in rows if row["variant"] == "probe_only"), 0
    )
    for row in rows:
        row["success_gain_vs_probe_only"] = int(row["success"]) - int(baseline)
    payload = {
        "metric_definitions": {
            "failure_resolution_rate": (
                "successful episodes among episodes in which at least one "
                "failure/stall-triggered probe was executed"
            ),
            "probes_per_resolved_failure": (
                "mean number of probe actions in successful probed episodes"
            ),
            "steps_to_recovery": "mean terminal step in successful probed episodes",
            "useful_probe_fraction": (
                "fraction of probe events with immediate utility > 0 after "
                "progress, reward, novelty, cost, and failure terms"
            ),
            "knowledge_applicability": (
                "fraction of selected knowledge events with immediate utility > 0"
            ),
        },
        "variants": rows,
    }
    with open(os.path.join(root, "ablation_summary.json"), "w") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
