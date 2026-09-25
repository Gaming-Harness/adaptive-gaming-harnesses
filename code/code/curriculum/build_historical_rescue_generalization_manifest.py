#!/usr/bin/env python3
"""Build held-out-world instances from historical Auto-K rescues.

Selection and evaluation are deliberately separated:

* Select source objectives only when the historical temperature-0.7 Bare run
  failed and Auto-K's best harness succeeded on the original DEV149 instance.
* Create deterministic, unseen Minecraft world seeds for evaluation.
* Do not inspect any held-out evaluation outcome while constructing the set.

There are fewer than 20 qualifying Mine objectives, so ``n-per-family=20``
means 20 environment instances (balanced across qualifying objectives), not
20 distinct task targets.  The manifest records both counts explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
DEV149_ROOT = Path(
    "/path/to/lab/"
    "anonymous/openha_dataset/openha_eval_50_per_type"
)
DEV149_MANIFEST = DEV149_ROOT / "manifest.json"
DEV149_TASKS = DEV149_ROOT / "tasks"
BARE_RESULTS = ROOT / "curriculum/outputs/batch_openha_bare_k1_t07/results.jsonl"
AUTO_K_ROOT = ROOT / "curriculum/outputs/auto_harness_hard_k1_t07_legacy_temp07_app2098"
DEFAULT_OUT = (
    ROOT
    / "curriculum/outputs/fixed_generalization_historical_rescues_mine20_kill20"
)


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def heldout_seed(sample_seed: int, family: str, task_name: str, slot: int) -> int:
    raw = f"{sample_seed}:{family}:{task_name}:{slot}:heldout-world"
    # MineStudio accepts positive integer world seeds. Keep the value in the
    # signed 31-bit range for compatibility with all task loaders.
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16) % (2**31 - 1) + 1


def historical_rescues(auto_k_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    manifest_rows = list(load_json(DEV149_MANIFEST).get("tasks") or [])
    by_stem = {Path(str(row["file"])).stem: row for row in manifest_rows}
    bare = {str(row.get("file")): row for row in read_jsonl(BARE_RESULTS)}
    selected: Dict[str, List[Dict[str, Any]]] = {
        "mine_block": [],
        "kill_entity": [],
    }

    for summary_path in sorted(auto_k_root.glob("search_*/summary.json")):
        summary = load_json(summary_path)
        if float(summary.get("best_success_rate") or 0.0) <= 0.0:
            continue
        search_stem = summary_path.parent.name[len("search_") :]
        source = by_stem.get(search_stem)
        if not source or source.get("task_type") not in selected:
            continue
        bare_row = bare.get(str(source.get("file")))
        if not bare_row or bool(bare_row.get("success")):
            continue
        row = dict(source)
        row.update(
            {
                "historical_bare_success": False,
                "historical_auto_k_best_success": True,
                "historical_auto_k_seed_success_rate": float(
                    summary.get("seed_success_rate") or 0.0
                ),
                "historical_auto_k_best_success_rate": float(
                    summary.get("best_success_rate") or 0.0
                ),
                "selection_summary": str(summary_path.resolve()),
            }
        )
        selected[str(source["task_type"])].append(row)

    for family in selected:
        selected[family].sort(key=lambda row: int(row["index"]))
    return selected


def build(out_root: Path, n_per_family: int, sample_seed: int, auto_k_root: Path) -> Dict[str, Any]:
    rescues = historical_rescues(auto_k_root)
    for family, rows in rescues.items():
        if not rows:
            raise RuntimeError(f"no historical Auto-K rescue sources for {family}")

    task_out = out_root / "tasks"
    task_out.mkdir(parents=True, exist_ok=True)
    samples: List[Dict[str, Any]] = []
    for family in ("mine_block", "kill_entity"):
        sources = rescues[family]
        for slot in range(int(n_per_family)):
            source = sources[slot % len(sources)]
            src = DEV149_TASKS / str(source["file"])
            task = load_json(src)
            original_seed = task.get("seed")
            world_seed = heldout_seed(
                int(sample_seed), family, str(source["task_name"]), slot
            )
            task["seed"] = world_seed
            if isinstance(task.get("openha_raw"), dict):
                task["openha_raw"]["seed"] = world_seed
            target = str(source["task_name"]).split(":", 1)[-1]
            dst_name = f"g{len(samples):03d}_{family}_{target}_ws{world_seed}.json"
            dst = task_out / dst_name
            write_json(dst, task)
            samples.append(
                {
                    "sample_id": len(samples),
                    "index811": int(source["index"]),
                    "file": dst_name,
                    "task_name": source["task_name"],
                    "task_type": family,
                    "task_path": str(dst.resolve()),
                    "source_task_path": str(src.resolve()),
                    "source_file": source["file"],
                    "generalization_axis": "unseen_world_seed_on_historical_rescue_target",
                    "in_dev149_by_task_name": True,
                    "historical_bare_success": False,
                    "historical_auto_k_best_success": True,
                    "historical_auto_k_seed_success_rate": source[
                        "historical_auto_k_seed_success_rate"
                    ],
                    "historical_auto_k_best_success_rate": source[
                        "historical_auto_k_best_success_rate"
                    ],
                    "selection_summary": source["selection_summary"],
                    "original_world_seed": original_seed,
                    "eval_world_seed": world_seed,
                }
            )

    manifest = {
        "sample_seed": int(sample_seed),
        "n_per_family": int(n_per_family),
        "n_samples": len(samples),
        "selection": "historical DEV149 Bare=0 and Auto-K Best=1",
        "selection_uses_heldout_outcomes": False,
        "evaluation_protocol": "new deterministic world seed; frozen harness and knowledge",
        "generalization_claim": "cross-world, not unseen-target",
        "source_auto_k_root": str(auto_k_root.resolve()),
        "source_target_counts": {key: len(value) for key, value in rescues.items()},
        "source_targets": {
            key: [row["task_name"] for row in value] for key, value in rescues.items()
        },
        "samples": samples,
    }
    write_json(out_root / "sample_manifest.json", manifest)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--n-per-family", type=int, default=20)
    ap.add_argument("--sample-seed", type=int, default=20260923)
    ap.add_argument("--auto-k-root", default=str(AUTO_K_ROOT))
    args = ap.parse_args()
    manifest = build(
        Path(args.out_root).resolve(),
        int(args.n_per_family),
        int(args.sample_seed),
        Path(args.auto_k_root).resolve(),
    )
    print(
        json.dumps(
            {
                "manifest": str((Path(args.out_root).resolve() / "sample_manifest.json")),
                "n_samples": manifest["n_samples"],
                "source_target_counts": manifest["source_target_counts"],
                "generalization_claim": manifest["generalization_claim"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
