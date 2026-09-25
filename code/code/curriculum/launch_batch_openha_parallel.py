#!/usr/bin/env python3
"""8-GPU parallel batch OpenHA eval (bare / static / frozen memory / frozen merge)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List

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


def _manifest_files(manifest: str) -> List[str]:
    with open(manifest) as f:
        m = json.load(f)
    return [str(t.get("file")) for t in (m.get("tasks") or []) if t.get("file")]


def _split(tasks: List[str], n: int) -> List[List[str]]:
    shards: List[List[str]] = [[] for _ in range(max(1, n))]
    for i, fn in enumerate(tasks):
        shards[i % len(shards)].append(fn)
    return shards


def _merge_jsonl(out_root: str, n: int) -> None:
    dst = os.path.join(out_root, "results.jsonl")
    seen = set()
    rows = []
    if os.path.isfile(dst):
        with open(dst) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                fn = r.get("file")
                if fn and fn not in seen:
                    seen.add(fn)
                    rows.append(r)
    for i in range(n):
        sp = os.path.join(out_root, f"shard_g{i}", "results.jsonl")
        if not os.path.isfile(sp):
            continue
        with open(sp) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                fn = r.get("file")
                if fn and fn not in seen:
                    seen.add(fn)
                    rows.append(r)
    rows.sort(key=lambda r: int(r.get("index") or 0))
    with open(dst, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_ok = sum(1 for r in rows if r.get("success"))
    summary = {
        "n": len(rows),
        "n_success": n_ok,
        "success_rate": float(n_ok) / max(1, len(rows)),
        "merged_from_shards": n,
    }
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[parallel-batch] merged {len(rows)} rows success={n_ok} rate={summary['success_rate']:.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--condition", choices=("bare", "harness"), required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--vla-protocol", default="minecraft_v1")
    ap.add_argument("--episode-seed", type=int, default=101)
    ap.add_argument("--vla-temperature", type=float, default=0.0)
    ap.add_argument("--harness-json", default="")
    ap.add_argument("--overlay-skills-from", default="")
    ap.add_argument("--persist-memory", action="store_true")
    ap.add_argument(
        "--vla-mode",
        default="hf",
        choices=("hf", "stub", "gemini", "gemini-flash", "gemini_flash"),
    )
    ap.add_argument("--vla-model-path", default="")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)
    files = _manifest_files(args.manifest)
    n = max(1, int(args.workers))
    shards = _split(files, n)
    print(
        f"[parallel-batch] out={out_root} condition={args.condition} "
        f"tasks={len(files)} workers={n} sizes={[len(s) for s in shards]}",
        flush=True,
    )

    py = _python()
    log_dir = os.path.join(_ROOT, "curriculum/outputs/logs")
    os.makedirs(log_dir, exist_ok=True)
    procs = []
    for gpu, tasks in enumerate(shards):
        if not tasks:
            continue
        shard_dir = os.path.join(out_root, f"shard_g{gpu}")
        os.makedirs(shard_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{os.path.basename(out_root)}_g{gpu}.log")
        cmd = [
            py, "-u", "-m", "curriculum.batch_openha_ablation",
            "--condition", args.condition,
            "--logdir", shard_dir,
            "--manifest", args.manifest,
            "--task-dir", args.task_dir,
            "--vla-protocol", args.vla_protocol,
            "--episode-seed", str(int(args.episode_seed)),
            "--vla-temperature", str(float(args.vla_temperature)),
            "--vla-mode", str(args.vla_mode),
            "--tasks", ",".join(tasks),
        ]
        if args.vla_model_path:
            cmd += ["--vla-model-path", args.vla_model_path]
        if args.harness_json:
            cmd += ["--harness-json", args.harness_json]
        if args.overlay_skills_from:
            cmd += ["--overlay-skills-from", args.overlay_skills_from]
        if args.persist_memory:
            cmd += ["--persist-memory"]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["AUTO_HARNESS_PYTHON"] = py
        env["PYTHONUNBUFFERED"] = "1"
        print(f"[parallel-batch] gpu={gpu} n={len(tasks)} first={tasks[0]} log={log_path}", flush=True)
        log_fh = open(log_path, "w")
        procs.append(subprocess.Popen(cmd, cwd=_ROOT, env=env, stdout=log_fh, stderr=subprocess.STDOUT))

    rc = 0
    t0 = time.time()
    for i, p in enumerate(procs):
        code = p.wait()
        print(f"[parallel-batch] gpu={i} exit={code}", flush=True)
        if code != 0:
            rc = code
    _merge_jsonl(out_root, n)
    print(f"[parallel-batch] ALL DONE exit={rc} elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
