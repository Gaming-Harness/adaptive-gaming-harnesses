#!/usr/bin/env python3
"""Prepare and rerun Auto-Harness tasks hit by LLMAPI AppId quota limits.

Scans search_*/run.log for「达到使用量上限」, writes a manifest, builds per-run
hard_tasks.json, and can remove stale search dirs so launch_auto_harness_parallel
will pick them up again.

Examples:
  python3 curriculum/scripts/quota_affected_retry.py list
  python3 curriculum/scripts/quota_affected_retry.py rescan
  python3 curriculum/scripts/quota_affected_retry.py prepare merge_t07
  python3 curriculum/scripts/quota_affected_retry.py clean merge_t07
  python3 curriculum/scripts/quota_affected_retry.py prepare merge_t07 --clean
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "curriculum" / "outputs"
MANIFEST_PATH = OUTPUTS / "quota_affected_tasks_manifest.json"
RETRY_ROOT = OUTPUTS / "quota_retry"
QUOTA_RE = re.compile(r"达到使用量上限")

# Short keys for CLI / RUN_KEY env in hope-job/run_quota_affected_retry_1gpu.sh
RUNS: Dict[str, Dict[str, Any]] = {
    "merge_t07": {
        "out_dir": "auto_harness_hard_merge_k1_t07_legacy",
        "source_hard": "batch_openha_bare_k1_t07/hard_tasks.json",
        "prior_mode": "merge",
        "vla_protocol": "legacy_k1_t0",
        "vla_temperature": 0.7,
        "proposer_model": "gemini-2.5-flash",
    },
    "merge_t0": {
        "out_dir": "auto_harness_hard_merge_k1_t0_legacy",
        "source_hard": "batch_openha_bare_k1_t0/hard_tasks.json",
        "prior_mode": "merge",
        "vla_protocol": "legacy_k1_t0",
        "vla_temperature": 0.0,
        "proposer_model": "gemini-2.5-flash",
    },
    "gemini25pro": {
        "out_dir": "auto_harness_hard_k1_t0_legacy_gemini25pro",
        "source_hard": "batch_openha_bare_k1_t0/hard_tasks.json",
        "prior_mode": "memory",
        "vla_protocol": "legacy_k1_t0",
        "vla_temperature": 0.0,
        "proposer_model": "gemini-2.5-pro",
    },
    "gpt4o": {
        "out_dir": "auto_harness_hard_k1_t0_legacy_gpt4o",
        "source_hard": "batch_openha_bare_k1_t0/hard_tasks.json",
        "prior_mode": "memory",
        "vla_protocol": "legacy_k1_t0",
        "vla_temperature": 0.0,
        "proposer_model": "gpt-4o",
    },
    "gpt5": {
        "out_dir": "auto_harness_hard_k1_t0_legacy_gpt5",
        "source_hard": "batch_openha_bare_k1_t0/hard_tasks.json",
        "prior_mode": "memory",
        "vla_protocol": "legacy_k1_t0",
        "vla_temperature": 0.0,
        "proposer_model": "gpt-5",
    },
}


def _task_name_from_search_dir(name: str) -> str:
    tid = name.replace("search_", "")
    return tid if tid.endswith(".json") else tid + ".json"


def _search_dir(out_root: Path, task_file: str) -> Path:
    return out_root / f"search_{task_file.replace('.json', '')}"


def _is_accepted(out_root: Path, task_file: str) -> bool:
    summary = _search_dir(out_root, task_file) / "summary.json"
    if not summary.is_file():
        return False
    try:
        data = json.loads(summary.read_text())
        return bool(data.get("accepted"))
    except Exception:
        return False


def scan_out_dir(out_dir: str) -> Dict[str, Any]:
    out_root = OUTPUTS / out_dir
    quota_tasks: Dict[str, int] = {}
    total_searches = 0
    if not out_root.is_dir():
        return {
            "quota_affected_count": 0,
            "total_searches": 0,
            "tasks": [],
            "hits_per_task": {},
        }
    for log in out_root.glob("search_*/run.log"):
        total_searches += 1
        task = _task_name_from_search_dir(log.parent.name)
        try:
            hits = len(QUOTA_RE.findall(log.read_text(errors="ignore")))
        except OSError:
            hits = 0
        if hits:
            quota_tasks[task] = hits
    return {
        "quota_affected_count": len(quota_tasks),
        "total_searches": total_searches,
        "tasks": sorted(quota_tasks.keys()),
        "hits_per_task": quota_tasks,
    }


def rescan_manifest() -> Dict[str, Any]:
    runs: Dict[str, Any] = {}
    unique: Set[str] = set()
    for key, cfg in RUNS.items():
        info = scan_out_dir(cfg["out_dir"])
        runs[cfg["out_dir"]] = info
        unique.update(info["tasks"])
    manifest = {
        "description": "Tasks with LLMAPI AppId quota errors (达到使用量上限) in run.log",
        "generated": str(date.today()),
        "runs": runs,
        "unique_tasks": sorted(unique),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def load_manifest(rescan: bool = False) -> Dict[str, Any]:
    if rescan or not MANIFEST_PATH.is_file():
        return rescan_manifest()
    return json.loads(MANIFEST_PATH.read_text())


def quota_tasks_for_run(run_key: str, manifest: Dict[str, Any]) -> List[str]:
    cfg = RUNS[run_key]
    out_dir = cfg["out_dir"]
    run_info = manifest.get("runs", {}).get(out_dir, {})
    return list(run_info.get("tasks", []))


def retry_tasks(
    run_key: str,
    manifest: Dict[str, Any],
    exclude_accepted: bool = True,
) -> List[str]:
    out_root = OUTPUTS / RUNS[run_key]["out_dir"]
    tasks = quota_tasks_for_run(run_key, manifest)
    if not exclude_accepted:
        return tasks
    return [t for t in tasks if not _is_accepted(out_root, t)]


def load_source_hard_entries(source_rel: str) -> Dict[str, Dict[str, Any]]:
    path = OUTPUTS / source_rel
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    out: Dict[str, Dict[str, Any]] = {}
    for item in data:
        if isinstance(item, dict) and item.get("file"):
            out[item["file"]] = item
        elif isinstance(item, str):
            fn = item if item.endswith(".json") else item + ".json"
            out[fn] = {"file": fn}
    return out


def build_hard_json(run_key: str, manifest: Dict[str, Any], exclude_accepted: bool = True) -> Path:
    cfg = RUNS[run_key]
    tasks = retry_tasks(run_key, manifest, exclude_accepted=exclude_accepted)
    source = load_source_hard_entries(cfg["source_hard"])
    entries: List[Dict[str, Any]] = []
    for fn in tasks:
        if fn in source:
            entries.append(source[fn])
        else:
            entries.append({"file": fn})
    out_dir = RETRY_ROOT / run_key
    out_dir.mkdir(parents=True, exist_ok=True)
    hard_path = out_dir / "hard_tasks.json"
    hard_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    meta = {
        "run_key": run_key,
        "out_dir": cfg["out_dir"],
        "task_count": len(entries),
        "hard_json": str(hard_path.relative_to(ROOT)),
        "exclude_accepted": exclude_accepted,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return hard_path


def clean_search_dirs(
    run_key: str,
    manifest: Dict[str, Any],
    dry_run: bool = False,
    exclude_accepted: bool = True,
) -> List[str]:
    cfg = RUNS[run_key]
    out_root = OUTPUTS / cfg["out_dir"]
    removed: List[str] = []
    for fn in retry_tasks(run_key, manifest, exclude_accepted=exclude_accepted):
        search = _search_dir(out_root, fn)
        if not search.exists():
            continue
        if dry_run:
            print(f"[dry-run] would remove {search}")
        else:
            shutil.rmtree(search)
            print(f"removed {search}")
        removed.append(fn)
    return removed


def cmd_list(_: argparse.Namespace) -> None:
    manifest = load_manifest(rescan=False)
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Generated: {manifest.get('generated')}")
    print(f"Unique quota-affected tasks: {len(manifest.get('unique_tasks', []))}")
    print()
    for key, cfg in RUNS.items():
        out_dir = cfg["out_dir"]
        info = manifest.get("runs", {}).get(out_dir, {})
        n_quota = info.get("quota_affected_count", 0)
        retry_n = len(retry_tasks(key, manifest))
        print(f"  {key:12}  quota={n_quota:3}  to_retry={retry_n:3}  out={out_dir}")


def cmd_rescan(_: argparse.Namespace) -> None:
    manifest = rescan_manifest()
    print(f"Rescanned -> {MANIFEST_PATH}")
    print(f"Unique tasks: {len(manifest['unique_tasks'])}")


def cmd_prepare(args: argparse.Namespace) -> None:
    manifest = load_manifest(rescan=args.rescan)
    hard = build_hard_json(args.run_key, manifest, exclude_accepted=not args.include_accepted)
    n = len(json.loads(hard.read_text()))
    print(f"Prepared {n} tasks -> {hard}")
    if args.clean:
        clean_search_dirs(
            args.run_key,
            manifest,
            dry_run=args.dry_run,
            exclude_accepted=not args.include_accepted,
        )


def cmd_clean(args: argparse.Namespace) -> None:
    manifest = load_manifest(rescan=args.rescan)
    removed = clean_search_dirs(
        args.run_key,
        manifest,
        dry_run=args.dry_run,
        exclude_accepted=not args.include_accepted,
    )
    print(f"{'Would remove' if args.dry_run else 'Removed'} {len(removed)} search dirs")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Quota-affected Auto-Harness retry helper")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Show quota counts per run").set_defaults(func=cmd_list)
    sub.add_parser("rescan", help="Rescan run.logs and update manifest").set_defaults(func=cmd_rescan)

    p_prepare = sub.add_parser("prepare", help="Build hard_tasks.json for a run")
    p_prepare.add_argument("run_key", choices=sorted(RUNS.keys()))
    p_prepare.add_argument("--rescan", action="store_true", help="Rescan before prepare")
    p_prepare.add_argument("--clean", action="store_true", help="Also remove search_* dirs")
    p_prepare.add_argument("--dry-run", action="store_true")
    p_prepare.add_argument(
        "--include-accepted",
        action="store_true",
        help="Include tasks already accepted (default: skip them)",
    )
    p_prepare.set_defaults(func=cmd_prepare)

    p_clean = sub.add_parser("clean", help="Remove search_* dirs for quota-affected tasks")
    p_clean.add_argument("run_key", choices=sorted(RUNS.keys()))
    p_clean.add_argument("--rescan", action="store_true")
    p_clean.add_argument("--dry-run", action="store_true")
    p_clean.add_argument("--include-accepted", action="store_true")
    p_clean.set_defaults(func=cmd_clean)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
