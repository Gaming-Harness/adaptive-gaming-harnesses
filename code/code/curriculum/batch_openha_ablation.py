#!/usr/bin/env python3
"""Batch OpenHA ablation: bare VLA vs seed-harness.

Resume-friendly JSONL. One condition per process (share nothing).

  # Full bare sweep (needed to discover hard tasks)
  python -m curriculum.batch_openha_ablation --condition bare

  # Seed harness: ONLY on bare failures — never touch bare successes
  python -m curriculum.batch_openha_ablation --condition harness \\
    --only-hard-from curriculum/outputs/batch_openha_bare_k1_t0/hard_tasks.json \\
    --wait-bare --bare-jsonl curriculum/outputs/batch_openha_bare_k1_t0/results.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Set

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from curriculum.auto_harness_search import strong_seed
from curriculum.harness_runtime import HarnessRuntime
from curriculum.harness_schema import HarnessSpec

DEFAULT_TASK_DIR = (
    "/path/to/lab/"
    "anonymous/openha_dataset/openha_eval_50_per_type/tasks"
)
DEFAULT_MANIFEST = (
    "/path/to/lab/"
    "anonymous/openha_dataset/openha_eval_50_per_type/manifest.json"
)
VLA_PATH = (
    "/path/to/lab/"
    "collaborator_a/ares/output/openha/20260625-cold_start_qwen3_vl_8b_vpt_gui_with_aux_"
    "weighted_ckpt_16200-openha_32tasks/ckpt/400/hf"
)


def _load_manifest(path: str, task_dir: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        m = json.load(f)
    out = []
    for t in m.get("tasks") or []:
        fn = str(t.get("file") or "")
        fp = os.path.join(task_dir, fn)
        if not os.path.isfile(fp):
            continue
        out.append({
            "index": int(t.get("index", len(out))),
            "file": fn,
            "path": fp,
            "task_name": t.get("task_name"),
            "task_type": t.get("task_type"),
        })
    return out


def _done_keys(jsonl: str) -> set:
    done = set()
    if not os.path.isfile(jsonl):
        return done
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("file"):
                done.add(str(rec["file"]))
    return done


def _hard_files(hard_json: str) -> List[str]:
    if not hard_json or not os.path.isfile(hard_json):
        return []
    with open(hard_json) as f:
        hard = json.load(f)
    out: List[str] = []
    for h in hard or []:
        if isinstance(h, dict) and h.get("file"):
            out.append(str(h["file"]))
        elif isinstance(h, str):
            out.append(h)
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


def _make_harness(
    condition: str,
    task_config: str,
    img_dir: str,
    *,
    vla_temperature: float = 0.0,
    vla_protocol: str = "legacy_k1_t0",
    harness_json: str = "",
    overlay_skills_from: str = "",
    vla_mode: str = "hf",
    vla_model_path: str = "",
) -> HarnessSpec:
    if harness_json and os.path.isfile(harness_json) and condition != "bare":
        h = HarnessSpec.load(harness_json)
        h.name = f"frozen_{os.path.basename(os.path.dirname(harness_json)) or 'harness'}"
    else:
        h = strong_seed()
    h.runtime.backend = "minestudio"
    mode = str(vla_mode or "hf").strip().lower() or "hf"
    h.runtime.vla_mode = mode
    h.runtime.vla_model_path = str(vla_model_path or "").strip() or (
        "" if mode in ("gemini", "gemini-flash", "gemini_flash") else VLA_PATH
    )
    h.runtime.vla_device = "cuda"
    h.runtime.vla_dtype = "bfloat16"
    h.runtime.vla_temperature = float(vla_temperature)
    h.runtime.vla_do_sample = bool(float(vla_temperature) > 0)
    if mode in ("gemini", "gemini-flash", "gemini_flash"):
        h.runtime.gemini_temperature = float(vla_temperature) if float(vla_temperature) > 0 else 0.7
    h.runtime.vla_protocol = str(vla_protocol or "legacy_k1_t0")
    h.runtime.action_chunk_len = 4
    h.runtime.max_steps = 100
    h.runtime.ticks_per_action = 4
    h.runtime.checkpoint_every = 8
    h.runtime.task_config = task_config
    h.runtime.img_save_dir = img_dir
    h.runtime.success_reward_thresh = 0.5
    h.runtime.soft_rollback = True
    if condition == "bare":
        h.name = "bare_vla"
        h.probe.enabled = False
        h.memory.enabled = False
        h.memory.inject_into_prompt = False
        h.memory.write_success = False
        h.memory.write_failure = False
        h.memory.write_probe = False
        h.verify.enabled = False
        h.verify.prefer_memory_action = False
        h.recover.enabled = False
    elif not (harness_json and os.path.isfile(harness_json)):
        h.name = "seed_harness"
        # Intervene only when failing / stalled — do not poke healthy rollouts
        h.probe.enabled = True
        h.probe.probe_periodic = False
        h.probe.probe_on_fail = True
        h.probe.probe_on_stall = True
        h.probe.stall_steps = 8
        h.probe.warmup_probes = 0
        h.probe.every_n_steps = 4  # unused when probe_periodic=False
        h.probe.budget_per_episode = 8
        h.probe.action_pool = [
            "forward", "attack", "forward_attack", "turn_left", "look_up", "look_down",
        ]
        h.memory.enabled = True
        h.memory.inject_into_prompt = True
        h.verify.enabled = False
        h.verify.prefer_memory_action = False
        h.recover.enabled = True
        h.recover.recover_on_stall = True
        h.recover.stall_steps = 12
        h.recover.max_retries = 2
    if overlay_skills_from and condition != "bare":
        from curriculum.harness_skill_bank import overlay_promoted_skills

        overlay_promoted_skills(h, overlay_skills_from)
    return h


def _instruction_from_task(task_config: str) -> str:
    try:
        with open(task_config) as f:
            d = json.load(f)
        for k in ("instruction", "task", "goal", "prompt"):
            if d.get(k):
                return str(d[k])
        if d.get("task_name"):
            return f"Complete the Minecraft task: {d['task_name']}"
    except Exception:
        pass
    return "Complete the Minecraft task."


def run_batch(
    *,
    condition: str,
    logdir: str,
    episode_seed: int = 101,
    limit: Optional[int] = None,
    start_index: int = 0,
    vla_temperature: float = 0.0,
    vla_protocol: str = "legacy_k1_t0",
    only_hard_from: str = "",
    wait_bare: bool = False,
    bare_jsonl: str = "",
    expect_bare: int = 149,
    poll_s: int = 60,
    task_dir: str = "",
    manifest: str = "",
    harness_json: str = "",
    overlay_skills_from: str = "",
    tasks: Optional[List[str]] = None,
    vla_mode: str = "hf",
    vla_model_path: str = "",
    persist_memory: bool = False,
) -> Dict[str, Any]:
    os.makedirs(logdir, exist_ok=True)
    jsonl = os.path.join(logdir, "results.jsonl")
    summary_path = os.path.join(logdir, "summary.json")
    hard_path = os.path.join(logdir, "hard_tasks.json")
    img_dir = os.path.join(logdir, "images")
    os.makedirs(img_dir, exist_ok=True)

    man_path = manifest or DEFAULT_MANIFEST
    tdir = task_dir or DEFAULT_TASK_DIR
    all_tasks = _load_manifest(man_path, tdir)
    by_file = {t["file"]: t for t in all_tasks}
    shared_vla = None
    seen_hard: Set[str] = set()
    print(
        f"[batch] manifest={man_path} task_dir={tdir} n={len(all_tasks)} "
        f"vla_mode={vla_mode} vla_model={vla_model_path or '-'} "
        f"harness_json={harness_json or '-'} overlay_skills={overlay_skills_from or '-'}",
        flush=True,
    )

    def _run_task_list(todo: List[Dict[str, Any]], *, tag: str) -> None:
        nonlocal shared_vla
        if not todo:
            return
        if shared_vla is None:
            first = _make_harness(
                condition, todo[0]["path"], img_dir,
                vla_temperature=vla_temperature, vla_protocol=vla_protocol,
                harness_json=harness_json,
                overlay_skills_from=overlay_skills_from,
                vla_mode=vla_mode,
                vla_model_path=vla_model_path,
            )
            first.runtime.instruction = _instruction_from_task(todo[0]["path"])
            shared = HarnessRuntime(first, seed=int(episode_seed))
            shared_vla = shared.vla
            print(f"[batch] VLA ready; {tag} n={len(todo)}", flush=True)
        for i, t in enumerate(todo):
            h = _make_harness(
                condition, t["path"], img_dir,
                vla_temperature=vla_temperature, vla_protocol=vla_protocol,
                harness_json=harness_json,
                overlay_skills_from=overlay_skills_from,
                vla_mode=vla_mode,
                vla_model_path=vla_model_path,
            )
            h.runtime.instruction = _instruction_from_task(t["path"])
            rt = HarnessRuntime(h, seed=int(episode_seed), vla=shared_vla)
            t0 = time.time()
            err = None
            try:
                metrics = rt.evaluate_on_seeds(
                    [int(episode_seed)], persist_memory=bool(persist_memory)
                )
                ep = (metrics.get("episodes") or [{}])[0]
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                ep = {}
                print(f"[batch] FAIL {t['file']}: {err}", flush=True)
                traceback.print_exc()
            rec = {
                "condition": condition,
                "index": t["index"],
                "file": t["file"],
                "task_name": t["task_name"],
                "task_type": t["task_type"],
                "episode_seed": int(episode_seed),
                "vla_temperature": float(vla_temperature),
                "vla_mode": str(vla_mode),
                "vla_model_path": str(vla_model_path or ""),
                "success": bool(ep.get("success")) if not err else False,
                "steps": ep.get("steps"),
                "reward": ep.get("reward"),
                "n_probe": ep.get("n_probe"),
                "n_recover": ep.get("n_recover"),
                "n_memory_write": ep.get("n_memory_write"),
                "cascade_fail": ep.get("cascade_fail"),
                "ablation_variant": (h.meta or {}).get("ablation_variant"),
                "probe_events": list((ep.get("meta") or {}).get("probe_events") or []),
                "knowledge_events": list(
                    (ep.get("meta") or {}).get("knowledge_events") or []
                ),
                "persist_bandit_state": bool(persist_memory),
                "elapsed_s": round(time.time() - t0, 2),
                "error": err,
                "timestamp": time.time(),
                "hard_only": bool(only_hard_from),
            }
            with open(jsonl, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"[batch] [{i+1}/{len(todo)}] {t['file']} succ={rec['success']} "
                f"steps={rec['steps']} reward={rec['reward']} "
                f"probe={rec['n_probe']} {rec['elapsed_s']}s",
                flush=True,
            )
            _finalize(jsonl, summary_path, hard_path, condition)

    # Hard-only: never touch bare successes (aligned with Auto / Event queues)
    if only_hard_from:
        if condition != "harness":
            print(
                "[batch] WARNING: --only-hard-from is intended for harness; "
                f"got condition={condition}",
                flush=True,
            )
        print(
            f"[batch] HARD-ONLY from {only_hard_from} "
            f"wait_bare={wait_bare} bare_jsonl={bare_jsonl}",
            flush=True,
        )
        while True:
            hard = _hard_files(only_hard_from)
            done = _done_keys(jsonl)
            new_files = [f for f in hard if f not in seen_hard and f not in done]
            todo: List[Dict[str, Any]] = []
            for fn in new_files:
                if fn in by_file:
                    todo.append(by_file[fn])
                    seen_hard.add(fn)
                else:
                    print(f"[batch] skip missing task file {fn}", flush=True)
                    seen_hard.add(fn)
            if limit is not None:
                todo = todo[: int(limit)]
            if todo:
                print(
                    f"[batch] new hard={len(todo)} total_hard={len(hard)} "
                    f"done={len(done)}",
                    flush=True,
                )
                _run_task_list(todo, tag="hard-only")
            bare_n = _jsonl_n(bare_jsonl) if bare_jsonl else int(expect_bare)
            bare_done = (not wait_bare) or (bare_n >= int(expect_bare))
            pending = [
                f for f in _hard_files(only_hard_from) if f not in _done_keys(jsonl)
            ]
            if bare_done and not pending and not new_files:
                print(
                    f"[batch] hard-only complete bare={bare_n}/{expect_bare} "
                    f"harness_done={len(_done_keys(jsonl))}",
                    flush=True,
                )
                break
            if not todo:
                print(
                    f"[batch] waiting for bare hard… bare={bare_n}/{expect_bare} "
                    f"hard={len(hard)} harness_done={len(done)}",
                    flush=True,
                )
                time.sleep(max(5, int(poll_s)))
        return _finalize(jsonl, summary_path, hard_path, condition)

    # Full sweep (bare, or legacy full harness)
    if tasks:
        want = set(tasks)
        task_list = [by_file[f] for f in tasks if f in by_file]
        missing = [f for f in tasks if f not in by_file]
        for fn in missing:
            print(f"[batch] skip missing explicit task {fn}", flush=True)
    else:
        task_list = [t for t in all_tasks if int(t["index"]) >= int(start_index)]
        if limit is not None:
            task_list = task_list[: int(limit)]
    done = _done_keys(jsonl)
    todo = [t for t in task_list if t["file"] not in done]
    print(
        f"[batch] condition={condition} total={len(task_list)} done={len(done)} todo={len(todo)} "
        f"seed={episode_seed} vla_temp={vla_temperature}",
        flush=True,
    )
    if not todo:
        return _finalize(jsonl, summary_path, hard_path, condition)
    _run_task_list(todo, tag="full")
    return _finalize(jsonl, summary_path, hard_path, condition)


def _finalize(jsonl: str, summary_path: str, hard_path: str, condition: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    if os.path.isfile(jsonl):
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    n = len(rows)
    n_ok = sum(1 for r in rows if r.get("success"))
    by_type: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        tt = str(r.get("task_type") or "unknown")
        by_type.setdefault(tt, {"n": 0, "ok": 0})
        by_type[tt]["n"] += 1
        if r.get("success"):
            by_type[tt]["ok"] += 1
    for v in by_type.values():
        v["succ"] = v["ok"] / max(1, v["n"])
    hard = [
        {
            "file": r["file"],
            "task_name": r.get("task_name"),
            "task_type": r.get("task_type"),
            "index": r.get("index"),
        }
        for r in rows
        if not r.get("success")
    ]
    summary = {
        "condition": condition,
        "n": n,
        "n_success": n_ok,
        "success_rate": n_ok / max(1, n),
        "by_type": by_type,
        "n_hard": len(hard),
        "hard_tasks_path": hard_path,
        "jsonl": jsonl,
        "hard_only": any(r.get("hard_only") for r in rows),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(hard_path, "w") as f:
        json.dump(hard, f, indent=2, ensure_ascii=False)
    print(
        f"[batch] summary condition={condition} succ={n_ok}/{n} "
        f"rate={summary['success_rate']:.3f} hard={len(hard)}",
        flush=True,
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["bare", "harness"], required=True)
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--episode-seed", type=int, default=101)
    ap.add_argument("--vla-temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument(
        "--only-hard-from",
        default="",
        help="Only run these bare-failure tasks (harness must not touch bare wins).",
    )
    ap.add_argument(
        "--wait-bare",
        action="store_true",
        help="Poll bare hard_tasks.json until bare jsonl reaches --expect-bare.",
    )
    ap.add_argument(
        "--bare-jsonl",
        default="curriculum/outputs/batch_openha_bare_k1_t0/results.jsonl",
    )
    ap.add_argument("--expect-bare", type=int, default=149)
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument(
        "--vla-protocol",
        choices=("minecraft_v1", "legacy_k1_t0"),
        default="legacy_k1_t0",
        help="Qwen prompt/history protocol. Default legacy_k1_t0 matches old Bare.",
    )
    ap.add_argument("--task-dir", default="", help="Override OpenHA task JSON directory.")
    ap.add_argument("--manifest", default="", help="Override manifest.json path.")
    ap.add_argument(
        "--harness-json",
        default="",
        help="Frozen harness JSON (memory/merge prior). Ignored for bare.",
    )
    ap.add_argument(
        "--overlay-skills-from",
        default="",
        help="Evolve out_root whose skill_bank.jsonl is overlaid onto harness.",
    )
    ap.add_argument(
        "--persist-memory",
        action="store_true",
        help="Persist action/knowledge bandit deltas across tasks.",
    )
    ap.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task filenames (explicit subset / shard).",
    )
    ap.add_argument(
        "--vla-mode",
        default="hf",
        choices=("hf", "stub", "gemini", "gemini-flash", "gemini_flash"),
        help="VLA backend. Default hf (local Qwen). Use gemini for LLMAPI API.",
    )
    ap.add_argument(
        "--vla-model-path",
        default="",
        help="HF checkpoint dir, or Gemini model id (e.g. gemini-2.5-flash).",
    )
    args = ap.parse_args()
    if args.only_hard_from:
        default_log = "curriculum/outputs/batch_openha_harness_hard_k1_t0"
    else:
        default_log = f"curriculum/outputs/batch_openha_{args.condition}_k1_t0"
    logdir = args.logdir or default_log
    task_list = [t.strip() for t in str(args.tasks or "").split(",") if t.strip()]
    run_batch(
        condition=args.condition,
        logdir=logdir,
        episode_seed=int(args.episode_seed),
        limit=args.limit,
        start_index=int(args.start_index),
        vla_temperature=float(args.vla_temperature),
        vla_protocol=str(args.vla_protocol),
        only_hard_from=str(args.only_hard_from or ""),
        wait_bare=bool(args.wait_bare),
        bare_jsonl=str(args.bare_jsonl),
        expect_bare=int(args.expect_bare),
        poll_s=int(args.poll_s),
        task_dir=str(args.task_dir or ""),
        manifest=str(args.manifest or ""),
        harness_json=str(args.harness_json or ""),
        overlay_skills_from=str(args.overlay_skills_from or ""),
        persist_memory=bool(args.persist_memory),
        tasks=task_list or None,
        vla_mode=str(args.vla_mode or "hf"),
        vla_model_path=str(args.vla_model_path or ""),
    )


if __name__ == "__main__":
    main()
