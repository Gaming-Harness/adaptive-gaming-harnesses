#!/usr/bin/env python3
"""Sequential Auto-Harness over OpenHA eval tasks.

Default: only run on *bare failures* (hard_tasks.json), not all 149.
Intervening on tasks bare already solves is meaningless.

Prior modes (``--prior-mode``):
  memory  — H0 always original prior_base; successes -> success_memory.jsonl
  merge   — freeze prior_base; fold knobs into prior_best; next H0 = prior_best
  replace — prior_best := harness_best (legacy warm-start)

  # Memory-only (default out-root)
  python -u -m curriculum.run_auto_harness_queue \
    --out-root curriculum/outputs/auto_harness_hard_k1_t0 \
    --prior-mode memory \
    --only-hard-from curriculum/outputs/batch_openha_bare_k1_t0/hard_tasks.json \
    --wait-bare \
    --bare-jsonl curriculum/outputs/batch_openha_bare_k1_t0/results.jsonl \
    --k 1 --vla-temperature 0 --proposer-temperature 0

  # Merge knobs into prior (separate out-root)
  python -u -m curriculum.run_auto_harness_queue \
    --out-root curriculum/outputs/auto_harness_hard_merge_k1_t0 \
    --prior-mode merge \
    --only-hard-from curriculum/outputs/batch_openha_bare_k1_t0/hard_tasks.json \
    --wait-bare \
    --bare-jsonl curriculum/outputs/batch_openha_bare_k1_t0/results.jsonl \
    --k 1 --vla-temperature 0 --proposer-temperature 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List, Set

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

TASK_DIR = (
    "/path/to/lab/"
    "anonymous/openha_dataset/openha_eval_50_per_type/tasks"
)
MANIFEST = (
    "/path/to/lab/"
    "anonymous/openha_dataset/openha_eval_50_per_type/manifest.json"
)
VLA = (
    "/path/to/lab/"
    "collaborator_a/ares/output/openha/20260625-cold_start_qwen3_vl_8b_vpt_gui_with_aux_"
    "weighted_ckpt_16200-openha_32tasks/ckpt/400/hf"
)


def _python() -> str:
    env = (os.environ.get("AUTO_HARNESS_PYTHON") or "").strip()
    if env and os.path.isfile(env):
        return env
    alaya = "python"
    if os.path.isfile(alaya):
        return alaya
    return sys.executable


PY = _python()


def _all_tasks() -> List[str]:
    with open(MANIFEST) as f:
        m = json.load(f)
    out = []
    for t in m.get("tasks") or []:
        fn = str(t.get("file") or "")
        if fn and os.path.isfile(os.path.join(TASK_DIR, fn)):
            out.append(fn)
    return out


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


def _prior_path(out_root: str) -> str:
    """Working prior for next H0 in merge/replace modes; archive in memory mode."""
    return os.path.join(out_root, "prior_harness_best.json")


def _prior_base_path(out_root: str) -> str:
    """Frozen original prior base (immutable H0 foundation)."""
    return os.path.join(out_root, "prior_harness_base.json")


def _prior_meta_path(out_root: str) -> str:
    return os.path.join(out_root, "prior_effective.json")


def _is_effective(summary: dict) -> bool:
    accepted = int(summary.get("accepted") or 0)
    seed_score = float(summary.get("seed_score") or 0.0)
    best_score = float(summary.get("best_score") or 0.0)
    hold = float(summary.get("holdout_success_rate") or 0.0)
    effective = bool(summary.get("effective")) or accepted > 0 or best_score > seed_score + 1e-9
    if not effective:
        return False
    if accepted <= 0 and hold <= 0 and best_score <= seed_score + 1e-6:
        return False
    return True


def _promote_effective(
    out_root: str,
    task_file: str,
    logdir: str,
    *,
    prior_mode: str = "memory",
) -> bool:
    """Promote effective H* according to prior_mode.

    - memory: append success_memory; next H0 stays prior_base
    - merge: fold knobs into prior_best; base frozen; next H0 = prior_best
    - replace: copy harness_best -> prior_best; next H0 = prior_best
    """
    from curriculum.harness_io_lock import promote_lock

    prior_mode = str(prior_mode or "memory").lower().strip()
    summary_p = os.path.join(logdir, "summary.json")
    best_p = os.path.join(logdir, "harness_best.json")
    if not (os.path.isfile(summary_p) and os.path.isfile(best_p)):
        return False
    with open(summary_p) as f:
        s = json.load(f)
    if not _is_effective(s):
        print(f"[queue] no promote ({prior_mode}, not effective) {task_file}", flush=True)
        return False

    with promote_lock(out_root):
        return _promote_effective_locked(
            out_root, task_file, logdir, prior_mode=prior_mode, summary=s,
            summary_p=summary_p, best_p=best_p,
        )


def _promote_effective_locked(
    out_root: str,
    task_file: str,
    logdir: str,
    *,
    prior_mode: str,
    summary: dict,
    summary_p: str,
    best_p: str,
) -> bool:
    import shutil

    from curriculum.harness_schema import HarnessSpec
    from curriculum.harness_success_memory import (
        append_success_memory,
        compact_success_entry,
        ensure_original_prior_base,
        success_memory_path,
    )

    s = summary
    prior_mode = str(prior_mode or "memory").lower().strip()

    accepted = int(s.get("accepted") or 0)
    seed_score = float(s.get("seed_score") or 0.0)
    best_score = float(s.get("best_score") or 0.0)
    hold = float(s.get("holdout_success_rate") or 0.0)
    ensure_original_prior_base(_prior_base_path(out_root))
    best = HarnessSpec.load(best_p)
    entry = compact_success_entry(
        task_file=task_file, logdir=logdir, summary=s, harness=best
    )
    from curriculum.harness_skill_bank import record_skill

    skill = record_skill(
        out_root,
        task_file=task_file,
        logdir=logdir,
        summary=s,
        harness=best,
        diagnosis=str((entry.get("lesson") or {}).get("diagnosis") or ""),
        why=str((entry.get("lesson") or {}).get("why") or ""),
        proposal=str((entry.get("lesson") or {}).get("proposal") or ""),
    )
    if skill:
        entry["skill"] = skill
        print(
            f"[queue] SKILL {skill.get('skill_id')} promoted={skill.get('promoted')} "
            f"n_tasks={skill.get('n_tasks')} holdout={skill.get('holdout_win')} "
            f"<- {task_file}",
            flush=True,
        )
    # Always record memory for analysis / proposer (all modes).
    mem_path = success_memory_path(out_root)
    append_success_memory(mem_path, entry)

    merge_report = None
    next_h0 = _prior_base_path(out_root)
    if prior_mode == "replace":
        shutil.copy2(best_p, _prior_path(out_root))
        next_h0 = _prior_path(out_root)
        mode_tag = "replace_prior"
        print(
            f"[queue] REPLACE prior <- {task_file} best={best_score:.4f} hold={hold} "
            f"-> {_prior_path(out_root)}",
            flush=True,
        )
    elif prior_mode == "merge":
        from curriculum.harness_prior_merge import promote_merge

        _, merge_report = promote_merge(
            prior_path=_prior_path(out_root),
            base_path=_prior_base_path(out_root),
            best_path=best_p,
            task_file=task_file,
            holdout_success_rate=hold,
            holdout_win=bool(s.get("holdout_win")) or hold >= 0.5,
        )
        next_h0 = _prior_path(out_root)
        mode_tag = "merge_into_prior"
        print(
            f"[queue] MERGE effective into prior <- {task_file} "
            f"best={best_score:.4f} hold={hold} applied={merge_report.get('applied')} "
            f"-> {_prior_path(out_root)} (base={_prior_base_path(out_root)})",
            flush=True,
        )
    else:
        # memory-only: archive snapshot, H0 unchanged
        best.save(_prior_path(out_root))
        next_h0 = _prior_base_path(out_root)
        mode_tag = "success_memory_only"
        print(
            f"[queue] MEMORY store effective <- {task_file} "
            f"best={best_score:.4f} hold={hold} -> {mem_path} "
            f"(next H0 remains {_prior_base_path(out_root)})",
            flush=True,
        )

    meta = {
        "mode": mode_tag,
        "prior_mode": prior_mode,
        "source_task": task_file,
        "source_logdir": logdir,
        "accepted": accepted,
        "seed_score": seed_score,
        "best_score": best_score,
        "holdout_success_rate": s.get("holdout_success_rate"),
        "best_harness_fp": s.get("best_harness_fp"),
        "success_memory": mem_path,
        "next_h0": next_h0,
        "prior_base_path": _prior_base_path(out_root),
        "prior_path": _prior_path(out_root),
        "merge_report": merge_report,
        "timestamp": time.time(),
        "entry": entry,
    }
    with open(_prior_meta_path(out_root), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out_root, "prior_bank.jsonl"), "a") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
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
    success_memory_jsonl: str = "",
    actionable_patches: bool = False,
    vla_protocol: str = "minecraft_v1",
    proposer_model: str = "gemini-2.5-flash",
) -> None:
    task_path = os.path.join(TASK_DIR, task_file)
    k = max(1, int(k))
    # Keep the historical defaults, but allow independent full-run seed
    # replications without changing task files or silently reusing seed 101.
    eval_seed = int(os.environ.get("AUTO_HARNESS_EVAL_SEED", "101"))
    holdout_seed = int(os.environ.get("AUTO_HARNESS_HOLDOUT_SEED", "909"))
    eval_seeds = [eval_seed + 101 * i for i in range(k)]
    holdout = [holdout_seed]
    do_sample = "true" if float(vla_temperature) > 0 else "false"
    preserve = "true" if preserve_harness_body else "false"
    mem_line = (
        f"success_memory_jsonl: {success_memory_jsonl}\n"
        if success_memory_jsonl
        else ""
    )
    act_line = "actionable_patches: true\n" if actionable_patches else ""
    adaptive = os.environ.get("AUTO_HARNESS_ADAPTIVE_PROBING", "0") == "1"
    bandit_path = os.environ.get("AUTO_HARNESS_PROBE_BANDIT_STATE", "").strip()
    knowledge_adaptive = (
        os.environ.get("AUTO_HARNESS_KNOWLEDGE_PROBING", "0") == "1"
    )
    knowledge_bandit_path = os.environ.get(
        "AUTO_HARNESS_KNOWLEDGE_BANDIT_STATE", ""
    ).strip()
    adaptive_lines = ""
    if adaptive:
        if not bandit_path:
            raise ValueError("adaptive probing requires AUTO_HARNESS_PROBE_BANDIT_STATE")
        adaptive_lines = f"""probe_selection_strategy: ucb
probe_ucb_c: 1.2
probe_transfer_weight: 0.25
probe_downstream_success_reward: 1.0
probe_downstream_discount: 0.8
probe_bandit_state_path: {bandit_path}
"""
    if knowledge_adaptive:
        if not knowledge_bandit_path:
            raise ValueError(
                "knowledge probing requires AUTO_HARNESS_KNOWLEDGE_BANDIT_STATE"
            )
        adaptive_lines += f"""knowledge_probe_enabled: true
knowledge_probe_selection_strategy: ucb
knowledge_probe_ucb_c: 1.2
knowledge_probe_transfer_weight: 0.25
knowledge_probe_downstream_success_reward: 1.0
knowledge_probe_downstream_discount: 0.8
knowledge_probe_bandit_state_path: {knowledge_bandit_path}
"""
    model_id = str(proposer_model or "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    # Warm-start: keep evolved body. Cold start: fail/stall-only seed knobs.
    if preserve_harness_body:
        body = f"""seed: 0
logdir: {logdir}
n_rounds: 3
n_eval_episodes: {k}
eval_seeds: {eval_seeds}
holdout_seeds: {holdout}
accept_eps: 0.01
start: {start}
preserve_harness_body: {preserve}
{mem_line}{act_line}{adaptive_lines}backend: minestudio
vla_mode: hf
vla_model_path: {VLA}
vla_device: cuda
vla_dtype: bfloat16
vla_protocol: {vla_protocol}
vla_temperature: {float(vla_temperature)}
vla_do_sample: {do_sample}
action_chunk_len: 4
max_steps: 100
ticks_per_action: 4
checkpoint_every: 8
recover_max_retries: 2
proposer_mode: llm
proposer_model: {model_id}
proposer_temperature: {float(proposer_temperature)}
proposer_vision: true
proposer_max_frames: 4
attribution_probes: true
prefer_components: [probe, recover]
task_config: {task_path}
img_save_dir: {logdir}/images
success_reward_thresh: 0.5
soft_rollback: true
"""
    else:
        body = f"""seed: 0
logdir: {logdir}
n_rounds: 3
n_eval_episodes: {k}
eval_seeds: {eval_seeds}
holdout_seeds: {holdout}
accept_eps: 0.01
start: {start}
preserve_harness_body: false
{mem_line}{act_line}{adaptive_lines}backend: minestudio
vla_mode: hf
vla_model_path: {VLA}
vla_device: cuda
vla_dtype: bfloat16
vla_protocol: {vla_protocol}
vla_temperature: {float(vla_temperature)}
vla_do_sample: {do_sample}
action_chunk_len: 4
max_steps: 100
ticks_per_action: 4
checkpoint_every: 8
recover_max_retries: 2
probe_enabled: true
probe_periodic: false
probe_on_fail: true
probe_on_stall: true
probe_stall_steps: 8
warmup_probes: 0
recover_on_stall: true
recover_stall_steps: 12
memory_enabled: true
verify_enabled: false
recover_enabled: true
prefer_memory_action: false
probe_budget: 8
probe_every_n: 4
probe_action_pool: [forward, attack, forward_attack, turn_left, look_up, look_down]
proposer_mode: llm
proposer_model: {model_id}
proposer_temperature: {float(proposer_temperature)}
proposer_vision: true
proposer_max_frames: 4
attribution_probes: true
prefer_components: [probe, recover]
task_config: {task_path}
img_save_dir: {logdir}/images
success_reward_thresh: 0.5
soft_rollback: true
"""
    with open(out_cfg, "w") as f:
        f.write(body)


def _done(logdir: str) -> bool:
    return os.path.isfile(os.path.join(logdir, "summary.json"))


def _purge_images(logdir: str) -> None:
    """Drop step PNGs after a search; metrics live in json."""
    import shutil

    img = os.path.join(logdir, "images")
    if os.path.isdir(img):
        shutil.rmtree(img, ignore_errors=True)


def _run_one(
    task_file: str,
    out_root: str,
    *,
    k: int = 1,
    vla_temperature: float = 0.0,
    proposer_temperature: float = 0.0,
    prior_mode: str = "memory",
    actionable_patches: bool = False,
    vla_protocol: str = "minecraft_v1",
    proposer_model: str = "gemini-2.5-flash",
) -> int:
    from curriculum.harness_success_memory import (
        ensure_original_prior_base,
        success_memory_path,
    )

    prior_mode = str(prior_mode or "memory").lower().strip()
    stem = task_file.replace(".json", "")
    logdir = os.path.join(out_root, f"search_{stem}")
    if _done(logdir):
        print(f"[queue] skip done {task_file}", flush=True)
        return 0
    os.makedirs(logdir, exist_ok=True)
    cfg = os.path.join(out_root, f"cfg_{stem}.yaml")
    base = _prior_base_path(out_root)
    ensure_original_prior_base(base)
    prior = _prior_path(out_root)
    mem = success_memory_path(out_root)

    # H0 selection by prior_mode
    if prior_mode in ("merge", "replace") and os.path.isfile(prior):
        start = os.path.abspath(prior)
        h0_note = f"H0={prior_mode} prior_best {start}"
    else:
        start = os.path.abspath(base)
        h0_note = f"H0=original prior base {start}"
    adaptive = os.environ.get("AUTO_HARNESS_ADAPTIVE_PROBING", "0") == "1"
    knowledge_adaptive = (
        os.environ.get("AUTO_HARNESS_KNOWLEDGE_PROBING", "0") == "1"
    )
    preserve = not (adaptive or knowledge_adaptive)
    print(
        f"[queue] prior_mode={prior_mode}; {h0_note}; "
        f"success_memory={mem}",
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
        success_memory_jsonl=os.path.abspath(mem),
        actionable_patches=bool(actionable_patches),
        vla_protocol=str(vla_protocol),
        proposer_model=str(proposer_model or "gemini-2.5-flash"),
    )
    log_path = os.path.join(logdir, "run.log")
    print(
        f"[queue] START hard-fail {task_file} K={k} vla_t={vla_temperature} "
        f"prior_mode={prior_mode} start={start} -> {logdir}",
        flush=True,
    )
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(
            [PY, "-u", "-m", "curriculum.auto_harness_search", "--config", cfg],
            cwd=_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    ok = _done(logdir)
    if ok:
        _promote_effective(out_root, task_file, logdir, prior_mode=prior_mode)
    _purge_images(logdir)
    print(
        f"[queue] END {task_file} exit={proc.returncode} "
        f"summary={'ok' if ok else 'MISSING'} {elapsed/60:.1f}min",
        flush=True,
    )
    if int(proc.returncode) != 0 and elapsed < 30 and not ok:
        print(
            f"[queue] FATAL search died in {elapsed:.1f}s without summary "
            f"(check {log_path}); abort so later tasks are not skip-marked.",
            flush=True,
        )
        raise SystemExit(int(proc.returncode) or 1)
    return int(proc.returncode)


def _write_index(out_root: str, tasks: List[str]) -> None:
    from curriculum.harness_funnel import format_funnel, scan_out_root

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
                    "seed_success_rate": s.get("seed_success_rate"),
                    "best_success_rate": s.get("best_success_rate"),
                    "delta_success_rate": s.get("delta_success_rate"),
                    "holdout_success_rate": s.get("holdout_success_rate"),
                    "holdout_win": s.get("holdout_win"),
                    "n_propose": s.get("n_propose"),
                    "n_valid": s.get("n_valid"),
                    "proposer_failed": s.get("proposer_failed"),
                    "invalid_reasons": s.get("invalid_reasons"),
                })
            except Exception as e:
                rec["error"] = str(e)
        rows.append(rec)
    n_done = sum(1 for r in rows if r.get("done"))
    n_acc = sum(1 for r in rows if (r.get("accepted") or 0) > 0)
    report = scan_out_root(out_root)
    funnel = report.get("funnel") or {}
    idx = {
        "n_tasks": len(tasks),
        "n_done": n_done,
        "n_with_accept": n_acc,
        "mode": "hard_only",
        "funnel": funnel,
        "tasks": rows,
    }
    with open(os.path.join(out_root, "queue_index.json"), "w") as f:
        json.dump(idx, f, indent=2, default=str)
    print(f"[queue] index done={n_done}/{len(tasks)} with_accept={n_acc}", flush=True)
    if n_done:
        print(format_funnel(funnel), flush=True)


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
    prior_mode: str = "memory",
    actionable_patches: bool = False,
    vla_protocol: str = "minecraft_v1",
    proposer_model: str = "gemini-2.5-flash",
) -> None:
    """Poll bare hard_tasks; Auto only on new failures until bare finishes."""
    seen: Set[str] = set()
    processed: List[str] = []
    print(
        f"[queue] HARD-ONLY mode prior_mode={prior_mode} hard={hard_json} "
        f"bare_jsonl={bare_jsonl} expect_bare={expect_n} poll={poll_s}s",
        flush=True,
    )
    while True:
        hard = _hard_files(hard_json)
        bare_n = _jsonl_n(bare_jsonl)
        new = [f for f in hard if f not in seen]
        if new:
            print(
                f"[queue] bare_done={bare_n}/{expect_n} hard={len(hard)} "
                f"new={len(new)}: {new[:5]}{'...' if len(new)>5 else ''}",
                flush=True,
            )
        for fn in new:
            seen.add(fn)
            _run_one(
                fn,
                out_root,
                k=k,
                vla_temperature=vla_temperature,
                proposer_temperature=proposer_temperature,
                prior_mode=prior_mode,
                actionable_patches=bool(actionable_patches),
                vla_protocol=str(vla_protocol),
                proposer_model=str(proposer_model or "gemini-2.5-flash"),
            )
            processed.append(fn)
            _write_index(out_root, processed)

        bare_done = bare_n >= int(expect_n)
        pending = [f for f in _hard_files(hard_json) if not _done(
            os.path.join(out_root, f"search_{f.replace('.json', '')}")
        )]
        if bare_done and not pending and not new:
            print(
                f"[queue] bare complete ({bare_n}) and all hard Auto done "
                f"n_hard={len(processed)}",
                flush=True,
            )
            break
        if not new:
            print(
                f"[queue] waiting… bare={bare_n}/{expect_n} hard={len(hard)} "
                f"auto_done={sum(1 for f in processed if _done(os.path.join(out_root, 'search_'+f.replace('.json',''))))}",
                flush=True,
            )
            time.sleep(max(5, int(poll_s)))
    _write_index(out_root, processed)
    print("[queue] ALL DONE (hard-only)", flush=True)
    try:
        from curriculum.harness_funnel import format_funnel, scan_out_root
        print(format_funnel(scan_out_root(out_root)["funnel"]), flush=True)
    except Exception as e:
        print(f"[queue] funnel_report_failed ({e})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="curriculum/outputs/auto_harness_hard_k1_t0")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--vla-temperature", type=float, default=0.0)
    ap.add_argument("--proposer-temperature", type=float, default=0.0)
    ap.add_argument("--tasks", default=None, help="Comma-separated filenames (explicit subset).")
    ap.add_argument(
        "--only-hard-from",
        default="curriculum/outputs/batch_openha_bare_k1_t0/hard_tasks.json",
        help="Only Auto-search bare failures listed here.",
    )
    ap.add_argument(
        "--wait-bare",
        action="store_true",
        help="Poll hard_tasks as bare progresses; stop when bare jsonl reaches --expect-bare.",
    )
    ap.add_argument(
        "--bare-jsonl",
        default="curriculum/outputs/batch_openha_bare_k1_t0/results.jsonl",
    )
    ap.add_argument("--expect-bare", type=int, default=149)
    ap.add_argument("--poll-s", type=int, default=120)
    ap.add_argument(
        "--all-tasks",
        action="store_true",
        help="DANGEROUS: run Auto on all 149 (not recommended).",
    )
    ap.add_argument(
        "--prior-mode",
        choices=("memory", "merge", "replace"),
        default="memory",
        help=(
            "How effective H* affects next H0: "
            "memory=always original prior_base + success_memory; "
            "merge=freeze prior_base, fold knobs into prior_best; "
            "replace=prior_best := harness_best (legacy)."
        ),
    )
    ap.add_argument(
        "--actionable-patches",
        action="store_true",
        help="Reject abort/impossible proposer patches; require look/approach/attack edits.",
    )
    ap.add_argument(
        "--vla-protocol",
        choices=("minecraft_v1", "legacy_k1_t0"),
        default="minecraft_v1",
        help="Qwen prompt/history protocol; legacy_k1_t0 reproduces the old single-turn run.",
    )
    ap.add_argument(
        "--proposer-model",
        default="gemini-2.5-flash",
        help="LLMAPI VLM id for Auto proposer (not the actor). Actor stays Qwen ckpt/400.",
    )
    args = ap.parse_args()
    out_root = args.out_root
    prior_mode = str(args.prior_mode)
    os.makedirs(out_root, exist_ok=True)
    print(
        f"[queue] out_root={out_root} prior_mode={prior_mode} "
        f"actionable_patches={bool(args.actionable_patches)} "
        f"vla_protocol={args.vla_protocol} proposer_model={args.proposer_model}",
        flush=True,
    )

    if args.tasks:
        tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
        print(f"[queue] explicit tasks n={len(tasks)}", flush=True)
        for i, fn in enumerate(tasks):
            print(f"[queue] === {i+1}/{len(tasks)} {fn} ===", flush=True)
            _run_one(
                fn, out_root, k=int(args.k),
                vla_temperature=float(args.vla_temperature),
                proposer_temperature=float(args.proposer_temperature),
                prior_mode=prior_mode,
                actionable_patches=bool(args.actionable_patches),
                vla_protocol=str(args.vla_protocol),
                proposer_model=str(args.proposer_model),
            )
            _write_index(out_root, tasks)
        print("[queue] ALL DONE", flush=True)
        return

    if args.all_tasks:
        tasks = _all_tasks()[int(args.start_index):]
        if args.limit is not None:
            tasks = tasks[: int(args.limit)]
        print(f"[queue] ALL-TASKS mode n={len(tasks)} (not failure-gated)", flush=True)
        for i, fn in enumerate(tasks):
            print(f"[queue] === {i+1}/{len(tasks)} {fn} ===", flush=True)
            _run_one(
                fn, out_root, k=int(args.k),
                vla_temperature=float(args.vla_temperature),
                proposer_temperature=float(args.proposer_temperature),
                prior_mode=prior_mode,
                actionable_patches=bool(args.actionable_patches),
                vla_protocol=str(args.vla_protocol),
                proposer_model=str(args.proposer_model),
            )
            _write_index(out_root, tasks)
        print("[queue] ALL DONE", flush=True)
        return

    # Default: hard-only
    if args.wait_bare or True:  # always poll hard list
        _run_hard_poll(
            out_root=out_root,
            hard_json=str(args.only_hard_from),
            bare_jsonl=str(args.bare_jsonl),
            expect_n=int(args.expect_bare),
            poll_s=int(args.poll_s),
            k=int(args.k),
            vla_temperature=float(args.vla_temperature),
            proposer_temperature=float(args.proposer_temperature),
            prior_mode=prior_mode,
            actionable_patches=bool(args.actionable_patches),
            vla_protocol=str(args.vla_protocol),
            proposer_model=str(args.proposer_model),
        )
        return


if __name__ == "__main__":
    main()
