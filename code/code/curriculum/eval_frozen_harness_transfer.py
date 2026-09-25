#!/usr/bin/env python3
"""Frozen H0 vs H* transfer eval on unseen OpenHA tasks (no search).

Evolution (merge search) happens on an evolve split. This script freezes those
harnesses and scores them on held-out hard tasks — the actual evolution test:

  Harness improvement → cross-task generalization

  python -m curriculum.eval_frozen_harness_transfer \
    --evolve-root curriculum/outputs/auto_harness_hard_merge_k1_t0 \
    --out-root curriculum/outputs/frozen_transfer_merge_k1_t0 \
    --hard-json curriculum/outputs/batch_openha_bare_k1_t0/hard_tasks.json \
    --vla-protocol minecraft_v1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from curriculum.harness_runtime import HarnessRuntime
from curriculum.harness_schema import HarnessSpec

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


def _load_manifest() -> List[Dict[str, Any]]:
    with open(MANIFEST) as f:
        m = json.load(f)
    out: List[Dict[str, Any]] = []
    for t in m.get("tasks") or []:
        fn = str(t.get("file") or "")
        fp = os.path.join(TASK_DIR, fn)
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


def _hard_files(hard_json: str) -> List[str]:
    with open(hard_json) as f:
        hard = json.load(f)
    out: List[str] = []
    for h in hard or []:
        fn = str(h.get("file") if isinstance(h, dict) else h or "")
        if fn:
            out.append(fn)
    return out


def _evolve_files(evolve_root: str) -> Tuple[List[str], List[str]]:
    """Return (done evolve tasks, incomplete/in-progress searches)."""
    done: List[str] = []
    incomplete: List[str] = []
    if not os.path.isdir(evolve_root):
        return done, incomplete
    for name in sorted(os.listdir(evolve_root)):
        if not name.startswith("search_"):
            continue
        stem = name[len("search_"):]
        fn = stem + ".json"
        logdir = os.path.join(evolve_root, name)
        if os.path.isfile(os.path.join(logdir, "summary.json")):
            done.append(fn)
        else:
            incomplete.append(fn)
    return done, incomplete


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


def _leaf_diff(a: Any, b: Any, prefix: str = "") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if type(a) is not type(b):
        out.append({"path": prefix, "h0": a, "hstar": b})
        return out
    if isinstance(a, dict):
        keys = sorted(set(a) | set(b))
        for k in keys:
            p = f"{prefix}.{k}" if prefix else str(k)
            if k not in a:
                out.append({"path": p, "h0": None, "hstar": b[k]})
            elif k not in b:
                out.append({"path": p, "h0": a[k], "hstar": None})
            else:
                out.extend(_leaf_diff(a[k], b[k], p))
        return out
    if isinstance(a, list):
        if a != b:
            out.append({"path": prefix, "h0": a, "hstar": b})
        return out
    if a != b:
        out.append({"path": prefix, "h0": a, "hstar": b})
    return out


_RUNTIME_LOCAL = {
    "task_config", "img_save_dir", "instruction",
    "backend", "vla_mode", "vla_model_path", "vla_device", "vla_dtype",
    "gemini_api_key",
}


def capability_diff(h0: HarnessSpec, hstar: HarnessSpec) -> Dict[str, Any]:
    d0, d1 = h0.to_dict(), hstar.to_dict()
    # Drop task-local runtime so the report is about transferable abilities.
    for d in (d0, d1):
        rt = d.get("runtime") or {}
        for k in list(_RUNTIME_LOCAL):
            rt.pop(k, None)
    diffs = [
        x for x in _leaf_diff(d0, d1)
        if not str(x.get("path") or "").startswith("meta")
        and not str(x.get("path") or "").startswith("name")
        and not str(x.get("path") or "").startswith("version")
    ]
    return {
        "n_diff": len(diffs),
        "diffs": diffs,
        "h0_fp": h0.fingerprint(),
        "hstar_fp": hstar.fingerprint(),
        "note": (
            "Transferable harness delta (task_config/instruction stripped). "
            "Empty / knob-only diffs mean evolution did not add a new ability."
        ),
    }


def bind_frozen(
    spec_path: str,
    *,
    task_path: str,
    img_dir: str,
    vla_protocol: str,
    vla_temperature: float = 0.0,
    evolve_root: str = "",
    apply_skills: bool = False,
) -> HarnessSpec:
    h = HarnessSpec.load(spec_path)
    h.runtime.backend = "minestudio"
    h.runtime.vla_mode = "hf"
    h.runtime.vla_model_path = VLA
    h.runtime.vla_device = "cuda"
    h.runtime.vla_dtype = "bfloat16"
    h.runtime.vla_protocol = str(vla_protocol)
    h.runtime.vla_temperature = float(vla_temperature)
    h.runtime.vla_do_sample = bool(float(vla_temperature) > 0)
    h.runtime.action_chunk_len = 4
    h.runtime.max_steps = 100
    h.runtime.ticks_per_action = 4
    h.runtime.checkpoint_every = 8
    h.runtime.task_config = task_path
    h.runtime.img_save_dir = img_dir
    h.runtime.instruction = _instruction_from_task(task_path)
    h.runtime.success_reward_thresh = 0.5
    h.runtime.soft_rollback = True
    if apply_skills and evolve_root:
        try:
            from curriculum.harness_skill_bank import overlay_promoted_skills

            skills = overlay_promoted_skills(h, evolve_root)
            if skills:
                print(
                    f"[frozen] overlay skills={[s.get('skill_id') for s in skills]} "
                    f"target={(h.meta or {}).get('current_target')}",
                    flush=True,
                )
        except Exception as e:
            print(f"[frozen] skill overlay skipped: {e}", flush=True)
    return h


def _done_pairs(jsonl: str) -> set:
    done = set()
    if not os.path.isfile(jsonl):
        return done
    seen: Dict[str, set] = {}
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            fn = str(rec.get("file") or "")
            cond = str(rec.get("condition") or "")
            if fn and cond:
                seen.setdefault(fn, set()).add(cond)
    for fn, conds in seen.items():
        if "h0" in conds and "hstar" in conds:
            done.add(fn)
    return done


def _summarize(jsonl: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    if os.path.isfile(jsonl):
        with open(jsonl) as f:
            for line in f:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    by: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(str(r.get("file")), {})[str(r.get("condition"))] = r
    n = 0
    n_h0 = n_star = n_both = n_star_only = 0
    for fn, m in by.items():
        if "h0" not in m or "hstar" not in m:
            continue
        n += 1
        s0 = bool(m["h0"].get("success"))
        s1 = bool(m["hstar"].get("success"))
        n_h0 += int(s0)
        n_star += int(s1)
        n_both += int(s0 and s1)
        n_star_only += int(s1 and not s0)
    return {
        "n_paired": n,
        "h0_success": n_h0,
        "hstar_success": n_star,
        "h0_sr": (n_h0 / n) if n else None,
        "hstar_sr": (n_star / n) if n else None,
        "delta_sr": ((n_star - n_h0) / n) if n else None,
        "hstar_only_rescues": n_star_only,
        "both_success": n_both,
    }


def run(
    *,
    evolve_root: str,
    out_root: str,
    hard_json: str,
    vla_protocol: str = "minecraft_v1",
    episode_seed: int = 101,
    holdout_seed: int = 909,
    use_holdout_seed: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    os.makedirs(out_root, exist_ok=True)
    h0_src = os.path.join(evolve_root, "prior_harness_base.json")
    star_src = os.path.join(evolve_root, "prior_harness_best.json")
    if not os.path.isfile(h0_src) or not os.path.isfile(star_src):
        raise FileNotFoundError(f"missing frozen specs in {evolve_root}")
    h0_path = os.path.join(out_root, "frozen_h0.json")
    star_path = os.path.join(out_root, "frozen_hstar.json")
    shutil.copy2(h0_src, h0_path)
    shutil.copy2(star_src, star_path)

    evolve_done, incomplete = _evolve_files(evolve_root)
    hard = _hard_files(hard_json)
    blocked = set(evolve_done) | set(incomplete)
    test = [f for f in hard if f not in blocked]
    if limit is not None:
        test = test[: int(limit)]

    split = {
        "evolve_root": os.path.abspath(evolve_root),
        "evolve_tasks": evolve_done,
        "incomplete_excluded": incomplete,
        "test_tasks": test,
        "n_evolve": len(evolve_done),
        "n_test": len(test),
        "episode_seed": int(episode_seed),
        "holdout_seed": int(holdout_seed) if use_holdout_seed else None,
        "vla_protocol": vla_protocol,
        "note": (
            "No search on test tasks. H0=prior_base, H*=prior_best at freeze time. "
            "Incomplete evolve searches are excluded from test."
        ),
    }
    with open(os.path.join(out_root, "split.json"), "w") as f:
        json.dump(split, f, indent=2)

    cap = capability_diff(HarnessSpec.load(h0_path), HarnessSpec.load(star_path))
    with open(os.path.join(out_root, "capability_diff.json"), "w") as f:
        json.dump(cap, f, indent=2, default=str)
    print(
        f"[frozen] evolve={len(evolve_done)} incomplete_excl={incomplete} "
        f"test={len(test)} cap_diffs={cap.get('n_diff')}",
        flush=True,
    )
    for d in (cap.get("diffs") or [])[:12]:
        print(f"  delta {d.get('path')}: {d.get('h0')!r} -> {d.get('hstar')!r}", flush=True)

    jsonl = os.path.join(out_root, "results.jsonl")
    img_dir = os.path.join(out_root, "images")
    os.makedirs(img_dir, exist_ok=True)
    by_file = {t["file"]: t for t in _load_manifest()}
    seeds: Sequence[int] = (
        [int(episode_seed), int(holdout_seed)] if use_holdout_seed else [int(episode_seed)]
    )

    done = _done_pairs(jsonl)
    shared_vla = None
    n_todo = sum(1 for fn in test if fn not in done)
    print(f"[frozen] resume done={len(done)} remaining={n_todo}", flush=True)

    for i, fn in enumerate(test):
        if fn in done:
            continue
        t = by_file.get(fn)
        if not t:
            print(f"[frozen] skip missing {fn}", flush=True)
            continue
        for cond, spec_p in (("h0", h0_path), ("hstar", star_path)):
            # skip a condition already written (partial crash)
            already = False
            if os.path.isfile(jsonl):
                with open(jsonl) as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        if rec.get("file") == fn and rec.get("condition") == cond:
                            already = True
                            break
            if already:
                continue
            h = bind_frozen(
                spec_p,
                task_path=t["path"],
                img_dir=os.path.join(img_dir, cond, fn.replace(".json", "")),
                vla_protocol=vla_protocol,
                evolve_root=evolve_root,
                apply_skills=(cond == "hstar"),
            )
            os.makedirs(h.runtime.img_save_dir, exist_ok=True)
            t0 = time.time()
            err = None
            try:
                rt = HarnessRuntime(h, seed=int(episode_seed), vla=shared_vla)
                if shared_vla is None:
                    shared_vla = rt.vla
                    print("[frozen] VLA ready", flush=True)
                metrics = rt.evaluate_on_seeds(list(seeds), persist_memory=False)
                ep = (metrics.get("episodes") or [{}])[0]
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                metrics = {}
                ep = {}
                print(f"[frozen] FAIL {cond} {fn}: {err}", flush=True)
                traceback.print_exc()
            rec = {
                "condition": cond,
                "file": fn,
                "index": t["index"],
                "task_name": t["task_name"],
                "task_type": t["task_type"],
                "episode_seed": int(episode_seed),
                "seeds": list(seeds),
                "success": bool(ep.get("success")) if not err else False,
                "success_rate": metrics.get("success_rate") if not err else 0.0,
                "steps": ep.get("steps"),
                "reward": ep.get("reward"),
                "n_probe": ep.get("n_probe"),
                "n_recover": ep.get("n_recover"),
                "elapsed_s": round(time.time() - t0, 2),
                "error": err,
                "harness_fp": h.fingerprint(),
                "timestamp": time.time(),
            }
            with open(jsonl, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"[frozen] [{i+1}/{len(test)}] {cond} {fn} "
                f"succ={rec['success']} sr={rec['success_rate']} "
                f"{rec['elapsed_s']}s",
                flush=True,
            )
        summ = _summarize(jsonl)
        with open(os.path.join(out_root, "summary.json"), "w") as f:
            json.dump({**split, **summ, "capability_diff": cap}, f, indent=2, default=str)
        print(
            f"[frozen] paired={summ['n_paired']} "
            f"H0={summ['h0_success']} H*={summ['hstar_success']} "
            f"ΔSR={summ['delta_sr']} rescues={summ['hstar_only_rescues']}",
            flush=True,
        )
    summ = _summarize(jsonl)
    out = {**split, **summ, "capability_diff": cap}
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("[frozen] done", json.dumps(summ), flush=True)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Frozen H0 vs H* cross-task transfer eval")
    ap.add_argument(
        "--evolve-root",
        default="curriculum/outputs/auto_harness_hard_merge_k1_t0",
    )
    ap.add_argument(
        "--out-root",
        default="curriculum/outputs/frozen_transfer_merge_k1_t0",
    )
    ap.add_argument(
        "--hard-json",
        default="curriculum/outputs/batch_openha_bare_k1_t0/hard_tasks.json",
    )
    ap.add_argument("--vla-protocol", default="minecraft_v1")
    ap.add_argument("--episode-seed", type=int, default=101)
    ap.add_argument("--holdout-seed", type=int, default=909)
    ap.add_argument(
        "--also-holdout-seed",
        action="store_true",
        help="Eval both 101 and 909 (2x cost). Default: seed 101 only.",
    )
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)
    run(
        evolve_root=str(args.evolve_root),
        out_root=str(args.out_root),
        hard_json=str(args.hard_json),
        vla_protocol=str(args.vla_protocol),
        episode_seed=int(args.episode_seed),
        holdout_seed=int(args.holdout_seed),
        use_holdout_seed=bool(args.also_holdout_seed),
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
