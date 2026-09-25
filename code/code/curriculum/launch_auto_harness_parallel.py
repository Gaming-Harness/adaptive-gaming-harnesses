#!/usr/bin/env python3
"""Launch N GPU-bound Auto-Harness queue workers on disjoint task shards.

Each worker sets CUDA_VISIBLE_DEVICES to one GPU and runs a serial queue on its
task list. All workers share the same out_root; promote uses a file lock so
success_memory / skill_bank / prior_best stay consistent.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Set, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)


def _python() -> str:
    env = (os.environ.get("AUTO_HARNESS_PYTHON") or "").strip()
    if env and os.path.isfile(env):
        return env
    alaya = "python"
    if os.path.isfile(alaya):
        return alaya
    return sys.executable


PY = _python()


def _hard_files(hard_json: str) -> List[str]:
    with open(hard_json) as f:
        hard = json.load(f)
    out: List[str] = []
    for h in hard:
        fn = h["file"] if isinstance(h, dict) else str(h)
        if fn:
            out.append(fn)
    return out


def _bare_results(path: str) -> Tuple[Dict[str, bool], int]:
    """Return file -> ever_succeeded and number of valid bare records."""
    by_file: Dict[str, bool] = {}
    n_rows = 0
    if path and not os.path.isfile(path):
        raise FileNotFoundError(f"bare results required for hard-only filtering: {path}")
    if not path or not os.path.isfile(path):
        return by_file, n_rows
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            fn = str(row.get("file") or "")
            if not fn:
                continue
            n_rows += 1
            by_file[fn] = bool(by_file.get(fn, False) or row.get("success", False))
    return by_file, n_rows


def _todo(out_root: str, hard_json: str, bare_jsonl: str = "") -> Tuple[List[str], List[str], Set[str], int]:
    hard = list(dict.fromkeys(_hard_files(hard_json)))
    bare, n_bare_rows = _bare_results(bare_jsonl)
    bare_success = {fn for fn, success in bare.items() if success}
    eligible = [fn for fn in hard if fn not in bare_success]
    todo: List[str] = []
    for fn in eligible:
        sp = os.path.join(out_root, f"search_{fn.replace('.json', '')}", "summary.json")
        if not os.path.isfile(sp):
            todo.append(fn)
    return todo, eligible, bare_success, n_bare_rows


def _write_accounting(
    out_root: str,
    *,
    hard_json: str,
    bare_jsonl: str,
    eligible: List[str],
    bare_success: Set[str],
    n_bare_rows: int,
    total_tasks: int,
    vla_temperature: float,
    vla_protocol: str,
    status: str,
) -> Dict[str, Any]:
    counts = {"seed": 0, "best": 0, "holdout": 0, "any": 0}
    n_done = 0
    for fn in eligible:
        path = os.path.join(out_root, f"search_{fn.replace('.json', '')}", "summary.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                summary = json.load(f)
        except Exception:
            continue
        n_done += 1
        flags = {
            "seed": float(summary.get("seed_success_rate") or 0.0) > 0,
            "best": float(summary.get("best_success_rate") or 0.0) > 0,
            "holdout": float(summary.get("holdout_success_rate") or 0.0) > 0,
        }
        flags["any"] = any(flags.values())
        for key, flag in flags.items():
            counts[key] += int(flag)
    base = len(bare_success)
    full = {
        key: {
            "success": base + value,
            "total": int(total_tasks),
            "success_rate": float(base + value) / max(1, int(total_tasks)),
        }
        for key, value in counts.items()
    }
    report: Dict[str, Any] = {
        "status": status,
        "selection": "bare_failures_only",
        "vla_temperature": float(vla_temperature),
        "vla_protocol": str(vla_protocol),
        "total_tasks": int(total_tasks),
        "bare_jsonl": os.path.abspath(bare_jsonl) if bare_jsonl else "",
        "bare_rows": int(n_bare_rows),
        "bare_success_skipped": base,
        "hard_json": os.path.abspath(hard_json),
        "hard_eligible": len(eligible),
        "auto_done": n_done,
        "auto_success_on_hard": counts,
        "combined_with_bare_success": full,
    }
    with open(os.path.join(out_root, "hard_only_accounting.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def _split_round_robin(tasks: List[str], n: int) -> List[List[str]]:
    shards: List[List[str]] = [[] for _ in range(max(1, n))]
    for i, fn in enumerate(tasks):
        shards[i % len(shards)].append(fn)
    return shards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prior-mode", choices=("memory", "merge", "replace"), default="memory")
    ap.add_argument("--hard-json", default="curriculum/outputs/batch_openha_bare_k1_t0/hard_tasks.json")
    ap.add_argument(
        "--bare-jsonl", default="",
        help="If set, forcibly exclude every task with a successful bare record.",
    )
    ap.add_argument("--total-tasks", type=int, default=149)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--vla-temperature", type=float, default=0.0)
    ap.add_argument("--proposer-temperature", type=float, default=0.0)
    ap.add_argument("--proposer-model", default="gemini-2.5-flash")
    ap.add_argument("--vla-protocol", default="minecraft_v1")
    ap.add_argument("--actionable-patches", action="store_true")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)
    n = max(1, int(args.workers))
    todo, eligible, bare_success, n_bare_rows = _todo(
        out_root, args.hard_json, args.bare_jsonl
    )
    shards = _split_round_robin(todo, n)
    accounting = _write_accounting(
        out_root,
        hard_json=args.hard_json,
        bare_jsonl=args.bare_jsonl,
        eligible=eligible,
        bare_success=bare_success,
        n_bare_rows=n_bare_rows,
        total_tasks=int(args.total_tasks),
        vla_temperature=float(args.vla_temperature),
        vla_protocol=str(args.vla_protocol),
        status="dry_run" if args.dry_run else "running",
    )

    print(
        f"[parallel] out_root={out_root} workers={n} prior_mode={args.prior_mode} "
        f"protocol={args.vla_protocol} temp={args.vla_temperature} "
        f"bare_success_skipped={len(bare_success)} eligible={len(eligible)} "
        f"todo={len(todo)} proposer={args.proposer_model} "
        f"shard_sizes={[len(s) for s in shards]}",
        flush=True,
    )
    if args.dry_run:
        print(f"[parallel] DRY RUN accounting={accounting}", flush=True)
        return
    if not todo:
        _write_accounting(
            out_root, hard_json=args.hard_json, bare_jsonl=args.bare_jsonl,
            eligible=eligible, bare_success=bare_success, n_bare_rows=n_bare_rows,
            total_tasks=int(args.total_tasks), vla_temperature=float(args.vla_temperature),
            vla_protocol=str(args.vla_protocol), status="complete",
        )
        print("[parallel] nothing todo", flush=True)
        return

    log_dir = os.path.join(_ROOT, "curriculum/outputs/logs")
    os.makedirs(log_dir, exist_ok=True)
    procs: List[subprocess.Popen] = []
    for gpu, tasks in enumerate(shards):
        if not tasks:
            print(f"[parallel] gpu={gpu} skip empty shard", flush=True)
            continue
        tag = f"{os.path.basename(out_root)}_g{gpu}"
        log_path = os.path.join(log_dir, f"auto_hard_{tag}_parallel.log")
        cmd = [
            PY, "-u", "-m", "curriculum.run_auto_harness_queue",
            "--out-root", out_root,
            "--prior-mode", str(args.prior_mode),
            "--vla-protocol", str(args.vla_protocol),
            "--tasks", ",".join(tasks),
            "--k", str(int(args.k)),
            "--vla-temperature", str(float(args.vla_temperature)),
            "--proposer-temperature", str(float(args.proposer_temperature)),
            "--proposer-model", str(args.proposer_model),
        ]
        if args.actionable_patches:
            cmd.append("--actionable-patches")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["AUTO_HARNESS_PYTHON"] = PY
        env["PYTHONUNBUFFERED"] = "1"
        print(
            f"[parallel] gpu={gpu} tasks={len(tasks)} first={tasks[0]} log={log_path}",
            flush=True,
        )
        log_fh = open(log_path, "w")
        procs.append(
            subprocess.Popen(
                cmd,
                cwd=_ROOT,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
        )

    rc = 0
    t0 = time.time()
    for i, p in enumerate(procs):
        code = p.wait()
        print(f"[parallel] worker gpu={i} exit={code}", flush=True)
        if code != 0:
            rc = code
    final_accounting = _write_accounting(
        out_root, hard_json=args.hard_json, bare_jsonl=args.bare_jsonl,
        eligible=eligible, bare_success=bare_success, n_bare_rows=n_bare_rows,
        total_tasks=int(args.total_tasks), vla_temperature=float(args.vla_temperature),
        vla_protocol=str(args.vla_protocol), status="complete" if rc == 0 else "worker_failed",
    )
    print(
        f"[parallel] ALL DONE exit={rc} elapsed={(time.time()-t0)/60:.1f}min "
        f"combined_any={final_accounting['combined_with_bare_success']['any']}",
        flush=True,
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
