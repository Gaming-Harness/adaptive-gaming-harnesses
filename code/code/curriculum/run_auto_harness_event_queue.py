#!/usr/bin/env python3
"""Hard-only queue for *event* Auto-Harness — mirrors run_auto_harness_queue.

Aligned protocol: K=1, VLA temp=0, fail/stall-only seed, warm-start prior,
promote effective policy_best / harness_best.

  cd project_root
  python -u -m curriculum.run_auto_harness_event_queue \\
    --out-root curriculum/outputs/auto_harness_event_hard_k1_t0 \\
    --only-hard-from curriculum/outputs/batch_openha_bare_k1_t0/hard_tasks.json \\
    --wait-bare \\
    --bare-jsonl curriculum/outputs/batch_openha_bare_k1_t0/results.jsonl \\
    --k 1 --vla-temperature 0 --proposer-temperature 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Set

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

TASK_DIR = (
    "/path/to/lab/"
    "anonymous/openha_dataset/openha_eval_50_per_type/tasks"
)
VLA = (
    "/path/to/lab/"
    "collaborator_a/ares/output/openha/20260625-cold_start_qwen3_vl_8b_vpt_gui_with_aux_"
    "weighted_ckpt_16200-openha_32tasks/ckpt/400/hf"
)
PY = "python"


def _hard_files(hard_json: str) -> List[str]:
    if not os.path.isfile(hard_json):
        return []
    with open(hard_json) as f:
        hard = json.load(f)
    out = []
    for h in hard or []:
        fn = str(h.get("file") or "")
        if fn:
            out.append(fn)
    return out


def _jsonl_n(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _prior_policy(out_root: str) -> str:
    return os.path.join(out_root, "prior_policy_best.json")


def _prior_policy_base(out_root: str) -> str:
    return os.path.join(out_root, "prior_policy_base.json")


def _prior_harness(out_root: str) -> str:
    return os.path.join(out_root, "prior_harness_best.json")


def _prior_harness_base(out_root: str) -> str:
    return os.path.join(out_root, "prior_harness_base.json")


def _merge_event_policy(base_path: str, prior_path: str, best_path: str, task_file: str):
    """Keep previous policy as foundation; fold effective routing/knobs in."""
    from curriculum.harness_event_policy import EventPolicy

    best = EventPolicy.load(best_path)
    if not os.path.isfile(base_path):
        foundation = EventPolicy.load(prior_path) if os.path.isfile(prior_path) else best
        foundation.name = "prior_policy_base"
        foundation.meta = {**(foundation.meta or {}), "prior_base": True, "base_from_task": task_file}
        foundation.save(base_path)
    cur = EventPolicy.load(prior_path) if os.path.isfile(prior_path) else EventPolicy.load(base_path)
    out = cur.clone(bump_version=True)
    out.name = "prior_policy_merged"
    # Accumulate: OR triggers, union pools, keep base prompts
    out.triggers.on_fail = bool(out.triggers.on_fail or best.triggers.on_fail)
    out.triggers.on_stall = bool(out.triggers.on_stall or best.triggers.on_stall)
    out.triggers.forbid_periodic = True
    out.observe.enabled = bool(out.observe.enabled or best.observe.enabled)
    out.observe.budget = max(int(out.observe.budget), int(best.observe.budget))
    pool = list(out.observe.action_pool or [])
    for a in best.observe.action_pool or []:
        if a not in pool:
            pool.append(a)
    out.observe.action_pool = pool
    out.recover.enabled = bool(out.recover.enabled or best.recover.enabled)
    out.recover.max_retries = max(int(out.recover.max_retries), int(best.recover.max_retries))
    out.replan.enabled = bool(out.replan.enabled or best.replan.enabled)
    # routing: only overwrite a slot if base is continue and best is richer
    for attr in ("on_stall", "on_fail", "on_warmup"):
        cur_d = getattr(out.routing, attr)
        new_d = getattr(best.routing, attr)
        if cur_d in ("continue", "") and new_d not in ("continue", ""):
            setattr(out.routing, attr, new_d)
    tips = list((out.meta or {}).get("prior_tips") or [])
    if best.replan.hint and best.replan.hint not in tips:
        tips.append(best.replan.hint[:240])
    out.meta = {
        **(out.meta or {}),
        "prior_merge": True,
        "prior_tips": tips[-8:],
        "merged_from_task": task_file,
        "merged_from_fp": best.fingerprint(),
    }
    out.save(prior_path)
    return out


def _promote_effective(out_root: str, task_file: str, logdir: str) -> bool:
    """Store effective policy into success memory; next task still starts from original H0."""
    from curriculum.harness_event_policy import EventPolicy
    from curriculum.harness_schema import HarnessSpec
    from curriculum.harness_success_memory import (
        append_success_memory,
        compact_success_entry,
        success_memory_path,
    )

    summary_p = os.path.join(logdir, "summary.json")
    best_pol = os.path.join(logdir, "policy_best.json")
    best_h = os.path.join(logdir, "harness_best.json")
    if not (os.path.isfile(summary_p) and os.path.isfile(best_pol)):
        return False
    with open(summary_p) as f:
        s = json.load(f)
    accepted = int(s.get("accepted") or 0)
    seed_score = float(s.get("seed_score") or 0.0)
    best_score = float(s.get("best_score") or 0.0)
    hold = float(s.get("holdout_success_rate") or 0.0)
    effective = bool(s.get("effective")) or accepted > 0 or best_score > seed_score + 1e-9
    if not effective:
        print(f"[event-queue] no memory (not effective) {task_file}", flush=True)
        return False
    if accepted <= 0 and hold <= 0 and best_score <= seed_score + 1e-6:
        print(f"[event-queue] no memory (weak effective) {task_file}", flush=True)
        return False

    pol = EventPolicy.load(best_pol)
    # Archive last success (not next H0)
    pol.save(_prior_policy(out_root))
    entry: Dict[str, Any]
    if os.path.isfile(best_h):
        h = HarnessSpec.load(best_h)
        h.save(_prior_harness(out_root))
        entry = compact_success_entry(
            task_file=task_file, logdir=logdir, summary=s, harness=h
        )
    else:
        entry = {
            "task": task_file,
            "logdir": logdir,
            "timestamp": time.time(),
            "accepted": accepted,
            "best_score": best_score,
            "holdout_success_rate": hold,
            "policy_fp": pol.fingerprint(),
            "lesson": {
                "routing": {
                    "on_stall": pol.routing.on_stall,
                    "on_fail": pol.routing.on_fail,
                },
                "stall_steps": pol.triggers.stall_steps,
                "action_pool": list(pol.observe.action_pool)[:12],
                "tip": pol.replan.hint[:180],
            },
        }
    mem = success_memory_path(out_root)
    append_success_memory(mem, entry)
    meta = {
        "mode": "success_memory_only",
        "source_task": task_file,
        "accepted": accepted,
        "best_score": best_score,
        "holdout_success_rate": hold,
        "success_memory": mem,
        "next_h0": "strong / original event seed",
        "timestamp": time.time(),
        "entry": entry,
    }
    with open(os.path.join(out_root, "prior_effective.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out_root, "prior_bank.jsonl"), "a") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(
        f"[event-queue] MEMORY store effective <- {task_file} "
        f"best={best_score:.4f} hold={hold} -> {mem} (next H0 stays original)",
        flush=True,
    )
    return True


def _write_cfg(
    task_file: str,
    out_cfg: str,
    logdir: str,
    *,
    k: int = 1,
    vla_temperature: float = 0.0,
    proposer_temperature: float = 0.0,
    start: str = "strong",
    preserve_harness_body: bool = False,
    proposer_mode: str = "bank",
    success_memory_jsonl: str = "",
) -> None:
    task_path = os.path.join(TASK_DIR, task_file)
    k = max(1, int(k))
    eval_seeds = [101 * (i + 1) for i in range(k)]
    holdout = [909]
    do_sample = "true" if float(vla_temperature) > 0 else "false"
    preserve = "true" if preserve_harness_body else "false"
    mem_line = (
        f"success_memory_jsonl: {success_memory_jsonl}\n" if success_memory_jsonl else ""
    )
    body = f"""seed: 0
logdir: {logdir}
n_rounds: 3
n_eval_episodes: {k}
eval_seeds: {eval_seeds}
holdout_seeds: {holdout}
accept_eps: 0.01
start: {start}
preserve_harness_body: {preserve}
{mem_line}backend: minestudio
vla_mode: hf
vla_model_path: {VLA}
vla_device: cuda
vla_dtype: bfloat16
vla_temperature: {float(vla_temperature)}
vla_do_sample: {do_sample}
action_chunk_len: 4
max_steps: 100
ticks_per_action: 4
checkpoint_every: 8
recover_max_retries: 2
probe_on_stall: true
probe_on_fail: true
warmup_probes: 0
stall_steps: 8
routing_on_stall: observe
routing_on_fail: recover
observe_budget: 8
probe_action_pool: [forward, attack, forward_attack, turn_left, look_up, look_down]
proposer_mode: {proposer_mode}
proposer_model: gemini-2.5-flash
proposer_temperature: {float(proposer_temperature)}
task_config: {task_path}
img_save_dir: {logdir}/images
success_reward_thresh: 0.5
soft_rollback: true
"""
    with open(out_cfg, "w") as f:
        f.write(body)


def _done(logdir: str) -> bool:
    return os.path.isfile(os.path.join(logdir, "summary.json"))


def _run_one(
    task_file: str,
    out_root: str,
    *,
    k: int = 1,
    vla_temperature: float = 0.0,
    proposer_temperature: float = 0.0,
    proposer_mode: str = "bank",
) -> int:
    from curriculum.harness_success_memory import success_memory_path

    stem = task_file.replace(".json", "")
    logdir = os.path.join(out_root, f"search_{stem}")
    if _done(logdir):
        print(f"[event-queue] skip done {task_file}", flush=True)
        return 0
    os.makedirs(logdir, exist_ok=True)
    cfg = os.path.join(out_root, f"cfg_{stem}.yaml")
    # Always original seed policy; successes only in memory.
    start = "strong"
    preserve = False
    mem = success_memory_path(out_root)
    print(
        f"[event-queue] H0=strong; success_memory={mem} (read-only)",
        flush=True,
    )
    _write_cfg(
        task_file,
        cfg,
        logdir,
        k=k,
        vla_temperature=vla_temperature,
        proposer_temperature=proposer_temperature,
        start=start,
        preserve_harness_body=preserve,
        proposer_mode=proposer_mode,
        success_memory_jsonl=os.path.abspath(mem),
    )
    log_path = os.path.join(logdir, "run.log")
    print(
        f"[event-queue] START {task_file} K={k} vla_t={vla_temperature} "
        f"start={start} -> {logdir}",
        flush=True,
    )
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(
            [PY, "-u", "-m", "curriculum.auto_harness_event", "--config", cfg],
            cwd=_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    ok = _done(logdir)
    if ok:
        _promote_effective(out_root, task_file, logdir)
    print(
        f"[event-queue] END {task_file} exit={proc.returncode} "
        f"summary={'ok' if ok else 'MISSING'} {elapsed/60:.1f}min",
        flush=True,
    )
    return int(proc.returncode)


def _write_index(out_root: str, tasks: List[str]) -> None:
    rows = []
    for fn in tasks:
        stem = fn.replace(".json", "")
        logdir = os.path.join(out_root, f"search_{stem}")
        sp = os.path.join(logdir, "summary.json")
        rec = {"file": fn, "done": False}
        if os.path.isfile(sp):
            try:
                s = json.load(open(sp))
                rec.update({
                    "done": True,
                    "accepted": s.get("accepted"),
                    "reverted": s.get("reverted"),
                    "seed_score": s.get("seed_score"),
                    "best_score": s.get("best_score"),
                    "holdout_success_rate": s.get("holdout_success_rate"),
                    "effective": s.get("effective"),
                    "proposer_failed": s.get("proposer_failed"),
                })
            except Exception as e:
                rec["error"] = str(e)
        rows.append(rec)
    n_done = sum(1 for r in rows if r.get("done"))
    n_acc = sum(1 for r in rows if (r.get("accepted") or 0) > 0)
    idx = {
        "n_tasks": len(tasks),
        "n_done": n_done,
        "n_with_accept": n_acc,
        "mode": "event_hard_only",
        "tasks": rows,
    }
    with open(os.path.join(out_root, "queue_index.json"), "w") as f:
        json.dump(idx, f, indent=2)
    print(f"[event-queue] index done={n_done}/{len(tasks)} with_accept={n_acc}", flush=True)


def _run_hard_poll(
    *,
    out_root: str,
    hard_json: str,
    bare_jsonl: str,
    expect_n: int,
    poll_s: int,
    k: int,
    vla_temperature: float,
    proposer_temperature: float,
    proposer_mode: str,
) -> None:
    seen: Set[str] = set()
    processed: List[str] = []
    print(
        f"[event-queue] HARD-ONLY hard={hard_json} bare_jsonl={bare_jsonl} "
        f"expect={expect_n} poll={poll_s}s",
        flush=True,
    )
    while True:
        hard = _hard_files(hard_json)
        bare_n = _jsonl_n(bare_jsonl)
        new = [f for f in hard if f not in seen]
        for fn in new:
            seen.add(fn)
            _run_one(
                fn,
                out_root,
                k=k,
                vla_temperature=vla_temperature,
                proposer_temperature=proposer_temperature,
                proposer_mode=proposer_mode,
            )
            processed.append(fn)
            _write_index(out_root, processed)
        bare_done = bare_n >= int(expect_n)
        pending = [
            f
            for f in _hard_files(hard_json)
            if not _done(os.path.join(out_root, f"search_{f.replace('.json', '')}"))
        ]
        if bare_done and not pending and not new:
            print(f"[event-queue] bare complete + all hard done n={len(processed)}", flush=True)
            break
        if not new:
            print(
                f"[event-queue] waiting… bare={bare_n}/{expect_n} hard={len(hard)}",
                flush=True,
            )
            time.sleep(max(5, int(poll_s)))
    _write_index(out_root, processed)
    print("[event-queue] ALL DONE (hard-only)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="curriculum/outputs/auto_harness_event_hard_k1_t0")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--vla-temperature", type=float, default=0.0)
    ap.add_argument("--proposer-temperature", type=float, default=0.0)
    ap.add_argument("--proposer-mode", default="bank", choices=["bank", "llm"])
    ap.add_argument("--tasks", default=None)
    ap.add_argument(
        "--only-hard-from",
        default="curriculum/outputs/batch_openha_bare_k1_t0/hard_tasks.json",
    )
    ap.add_argument("--wait-bare", action="store_true")
    ap.add_argument(
        "--bare-jsonl",
        default="curriculum/outputs/batch_openha_bare_k1_t0/results.jsonl",
    )
    ap.add_argument("--expect-bare", type=int, default=149)
    ap.add_argument("--poll-s", type=int, default=120)
    args = ap.parse_args()
    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)

    if args.tasks:
        tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
        for i, fn in enumerate(tasks):
            print(f"[event-queue] === {i+1}/{len(tasks)} {fn} ===", flush=True)
            _run_one(
                fn,
                out_root,
                k=int(args.k),
                vla_temperature=float(args.vla_temperature),
                proposer_temperature=float(args.proposer_temperature),
                proposer_mode=str(args.proposer_mode),
            )
            _write_index(out_root, tasks)
        print("[event-queue] ALL DONE", flush=True)
        return

    _run_hard_poll(
        out_root=out_root,
        hard_json=str(args.only_hard_from),
        bare_jsonl=str(args.bare_jsonl),
        expect_n=int(args.expect_bare),
        poll_s=int(args.poll_s),
        k=int(args.k),
        vla_temperature=float(args.vla_temperature),
        proposer_temperature=float(args.proposer_temperature),
        proposer_mode=str(args.proposer_mode),
    )


if __name__ == "__main__":
    main()
