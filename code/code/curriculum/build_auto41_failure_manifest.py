#!/usr/bin/env python3
"""Build the t=0 Auto+Probing evaluation set from the 41/149 Auto baseline.

The historical Auto legacy number is the union of 23 bare successes and 18
successes recovered by the valid t=0 Auto legacy searches.  Auto+Probing must
run only the complementary 108 tasks; otherwise its denominator/baseline is
incorrectly reported as the 23/149 bare result.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "curriculum/outputs/batch_openha_bare_k1_t0"
ALL_TASKS = DATA / "all149_tasks_for_auto_probing.json"
BARE_RESULTS = DATA / "results.jsonl"
BASELINE_JSONL = DATA / "auto_legacy41_baseline_successes.jsonl"
FAILURES_JSON = DATA / "auto_probing_failures_from_auto41.json"
MANIFEST_JSON = DATA / "auto_legacy41_manifest.json"

AUTO_ROOTS = [
    ROOT / "curriculum/outputs/auto_harness_hard_merge_k1_t0_legacy",
    ROOT / "curriculum/outputs/auto_harness_hard_k1_t0_legacy_actionable",
    ROOT / "curriculum/outputs/auto_harness_hard_k1_t0_legacy_gemini25pro",
    ROOT / "curriculum/outputs/auto_harness_hard_k1_t0_legacy_gpt4o",
    ROOT / "curriculum/outputs/auto_harness_hard_k1_t0_legacy_gpt5",
    ROOT / "curriculum/outputs/auto_harness_hard_merge_k1_t0_legacy_actionable",
]


def load_bare_successes() -> set[str]:
    successes: set[str] = set()
    with BARE_RESULTS.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_file = str(row.get("file") or "")
            if task_file and bool(row.get("success")):
                successes.add(task_file)
    return successes


def load_auto_successes() -> tuple[set[str], dict[str, list[str]]]:
    successes: set[str] = set()
    provenance: dict[str, list[str]] = {}
    for auto_root in AUTO_ROOTS:
        if not auto_root.is_dir():
            continue
        for summary_path in sorted(auto_root.glob("search_*/summary.json")):
            try:
                summary = json.loads(summary_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if float(summary.get("best_success_rate") or 0.0) <= 0:
                continue
            stem = summary_path.parent.name[len("search_"):] if summary_path.parent.name.startswith("search_") else summary_path.parent.name
            task_file = f"{stem}.json"
            successes.add(task_file)
            provenance.setdefault(task_file, []).append(auto_root.name)
    return successes, provenance


def main() -> None:
    tasks = json.loads(ALL_TASKS.read_text())
    by_file = {str(row["file"]): row for row in tasks}
    bare = load_bare_successes()
    auto, provenance = load_auto_successes()
    overlap = bare & auto
    baseline = bare | auto

    assert len(tasks) == 149, f"expected 149 tasks, got {len(tasks)}"
    assert len(bare) == 23, f"expected 23 bare successes, got {len(bare)}"
    assert len(auto) == 18, f"expected 18 Auto successes, got {len(auto)}"
    assert not overlap, f"bare/Auto overlap is unexpected: {sorted(overlap)}"
    assert len(baseline) == 41, f"expected Auto baseline 41, got {len(baseline)}"
    assert baseline <= set(by_file), "baseline contains tasks outside all149 manifest"

    ordered_successes = [row for row in tasks if row["file"] in baseline]
    failures = [row for row in tasks if row["file"] not in baseline]
    assert len(failures) == 108, f"expected 108 Auto failures, got {len(failures)}"

    with BASELINE_JSONL.open("w") as handle:
        for row in ordered_successes:
            source = "bare" if row["file"] in bare else "auto_legacy"
            handle.write(json.dumps({
                "file": row["file"],
                "success": True,
                "source": source,
            }, ensure_ascii=False) + "\n")
    FAILURES_JSON.write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n")
    MANIFEST_JSON.write_text(json.dumps({
        "protocol": "legacy_k1_t0",
        "total_tasks": 149,
        "baseline_name": "Auto legacy",
        "baseline_success": 41,
        "baseline_success_rate": 41 / 149,
        "bare_success": 23,
        "auto_additional_success": 18,
        "auto_failure_tasks": 108,
        "bare_success_files": sorted(bare),
        "auto_success_files": sorted(auto),
        "auto_success_provenance": provenance,
        "failure_json": str(FAILURES_JSON),
        "baseline_jsonl": str(BASELINE_JSONL),
    }, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Auto legacy baseline={len(baseline)}/149 "
        f"(bare={len(bare)} + auto={len(auto)}); failures={len(failures)}"
    )
    print(FAILURES_JSON)


if __name__ == "__main__":
    main()
