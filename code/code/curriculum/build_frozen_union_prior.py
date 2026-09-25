#!/usr/bin/env python3
"""Build one frozen evaluation prior from several completed Auto-Harness runs."""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

from curriculum.harness_schema import HarnessSpec
from curriculum.harness_skill_bank import load_skill_bank
from curriculum.harness_success_memory import load_success_memory


def _merge_skill_rows(roots: List[str]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for root in roots:
        for row in load_skill_bank(os.path.join(root, "skill_bank.jsonl")):
            if not row.get("promoted") or not row.get("skill_id"):
                continue
            skill_id = str(row["skill_id"])
            dst = by_id.setdefault(skill_id, dict(row))
            for key in ("source_tasks", "task_types", "triggers", "semantic_actions"):
                values = list(dst.get(key) or [])
                for value in row.get(key) or []:
                    if value not in values:
                        values.append(value)
                dst[key] = values
            dst["promoted"] = True
            dst["n_tasks"] = len(dst.get("source_tasks") or [])
            dst["n_accept"] = max(int(dst.get("n_accept") or 0), dst["n_tasks"])
    return sorted(by_id.values(), key=lambda row: str(row.get("skill_id")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", action="append", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    roots = [os.path.abspath(path) for path in args.source_root]
    # Fixed-harness is the immutable strong H0. Historical per-task best
    # harnesses are knowledge sources, not the executable base policy.
    merged = HarnessSpec.load(os.path.join(roots[0], "prior_harness_base.json"))
    instances = []
    for root in roots:
        instances.extend(load_success_memory(os.path.join(root, "success_memory.jsonl"), limit=100000))

    skills = _merge_skill_rows(roots)
    meta = dict(merged.meta or {})
    meta["prior_skills"] = [
        {
            "skill_id": row.get("skill_id"),
            "triggers": row.get("triggers"),
            "template": row.get("template"),
            "semantic_actions": row.get("semantic_actions"),
            "promoted": True,
            "source_tasks": row.get("source_tasks"),
            "n_tasks": row.get("n_tasks"),
        }
        for row in skills
    ]
    meta["prior_tips"] = [str(row.get("template")) for row in skills if row.get("template")]
    meta["frozen_prior"] = True
    meta["fixed_harness_base"] = True
    meta["n_prior_instances"] = len(instances)
    meta["prior_source_roots"] = roots
    meta["built_at"] = time.time()
    merged.meta = meta
    merged.name = "fixed_harness_with_merged_prior_t0_t07"

    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)
    prior_path = os.path.join(out_root, "prior_harness.json")
    merged.save(prior_path)
    with open(os.path.join(out_root, "skill_bank.jsonl"), "w") as f:
        for row in skills:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(out_root, "prior_knowledge.jsonl"), "w") as f:
        for row in instances:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    provenance = {
        "frozen": True,
        "base": "fixed_harness/prior_harness_base.json",
        "source_roots": roots,
        "source_prior_files": [os.path.join(root, "prior_harness_best.json") for root in roots],
        "n_prior_instances": len(instances),
        "n_promoted_skills": len(skills),
        "skills": [
            {
                "skill_id": row.get("skill_id"),
                "n_tasks": row.get("n_tasks"),
                "task_types": row.get("task_types"),
                "source_tasks": row.get("source_tasks"),
            }
            for row in skills
        ],
        "prior_path": prior_path,
    }
    with open(os.path.join(out_root, "provenance.json"), "w") as f:
        json.dump(provenance, f, ensure_ascii=False, indent=2)
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
