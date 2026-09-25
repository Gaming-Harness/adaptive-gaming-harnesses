#!/usr/bin/env python3
"""Fixed-harness generalization evaluation on OpenHA-811 samples.

The evaluation is deliberately *not* a harness search.  It compares three
paired conditions on the same task config, VLA sampling seed, and frozen VLA:

  bare                 frozen VLA without the recovery harness
  auto_k               frozen Auto-K harness/knowledge + coverage probe
  auto_k_adaptive      frozen co-evolved harness/knowledge/UCB priors

Bandit files are read as priors but never written. Mine and Craft samples are
targets outside the 149-task development benchmark. Since the 149 benchmark
already contains all 49 OpenHA-811 Kill targets, Kill generalization is tested
on new Minecraft world seeds and excludes targets that produced promoted
knowledge. Sampling is deterministic and restricted to historical Bare
failures, so the reported quantity is conditional rescue/generalization.
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from curriculum.batch_openha_ablation import VLA_PATH
from curriculum.harness_runtime import HarnessRuntime
from curriculum.harness_schema import HarnessSpec
from curriculum.harness_skill_bank import load_skill_bank, overlay_promoted_skills
from curriculum.replay_bare_auto_probing_triple import (
    _configure_shared_vla,
    _seed_everything,
)


OPENHA811_MANIFEST = _ROOT / "curriculum/outputs/openha_811/manifest.json"
OPENHA811_TASK_DIR = Path(
    "/path/to/lab/"
    "collaborator_a/ares/tests/openha/_openha_tasks_all"
)
DEV149_MANIFEST = Path(
    "/path/to/lab/"
    "anonymous/openha_dataset/openha_eval_50_per_type/manifest.json"
)
BARE811_ROOT = _ROOT / "curriculum/outputs/openha811_bare_mtv1"
AUTO_K_ROOT = _ROOT / "curriculum/outputs/auto_harness_hard_k1_t07_legacy_temp07_app2098"
ADAPTIVE_ROOT = (
    _ROOT
    / "curriculum/outputs/auto_harness_knowledge_coevolve_hard_k1_t07_legacy_temp07_app2098"
)
DEFAULT_OUT = _ROOT / "curriculum/outputs/frozen_transfer_mine20_kill20_craft20"

CONDITIONS = ("bare", "auto_k", "auto_k_adaptive")


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def _read_jsonl(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open() as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def _target(task_name: str) -> str:
    return str(task_name or "").split(":", 1)[-1]


def _promoted_source_targets(roots: Sequence[Path], task_type: str) -> set[str]:
    out: set[str] = set()
    marker = f"{task_type}_"
    for root in roots:
        for row in load_skill_bank(str(root / "skill_bank.jsonl")):
            if not row.get("promoted"):
                continue
            for source in row.get("source_tasks") or []:
                stem = Path(str(source)).stem
                if marker in stem:
                    out.add(stem.split(marker, 1)[1])
    return out


def _new_world_seed(sample_seed: int, task_name: str) -> int:
    digest = hashlib.sha256(f"{sample_seed}:{task_name}:kill-heldout".encode()).hexdigest()
    return int(digest[:15], 16) + 1


def build_sample_manifest(
    out_root: Path,
    *,
    sample_seed: int,
    n_per_family: int,
    auto_k_root: Path,
    adaptive_root: Path,
) -> Dict[str, Any]:
    """Create/reuse an audited Mine + Kill + Craft paired sample manifest."""
    sample_path = out_root / "sample_manifest.json"
    if sample_path.is_file():
        existing = _load_json(sample_path)
        if (
            int(existing.get("sample_seed", -1)) == int(sample_seed)
            and int(existing.get("n_per_family", -1)) == int(n_per_family)
        ):
            return existing

    all811 = list((_load_json(OPENHA811_MANIFEST).get("tasks") or []))
    dev149 = list((_load_json(DEV149_MANIFEST).get("tasks") or []))
    dev_names = {str(row.get("task_name")) for row in dev149}
    bare_rows = _read_jsonl(sorted(BARE811_ROOT.glob("shard_g*/results.jsonl")))
    bare_by_file = {str(row.get("file")): row for row in bare_rows if row.get("file")}

    mine_pool = [
        row
        for row in all811
        if row.get("task_type") == "mine_block"
        and str(row.get("task_name")) not in dev_names
        and str(row.get("file")) in bare_by_file
        and not bool(bare_by_file[str(row.get("file"))].get("success"))
    ]

    craft_pool = [
        row
        for row in all811
        if row.get("task_type") == "craft_item"
        and str(row.get("task_name")) not in dev_names
        and str(row.get("file")) in bare_by_file
        and not bool(bare_by_file[str(row.get("file"))].get("success"))
    ]

    # All 49 kill targets in OpenHA-811 are already present in DEV149.  Use
    # unseen world seeds, and avoid targets that generated promoted skills.
    knowledge_source_targets = _promoted_source_targets(
        [auto_k_root, adaptive_root], "kill_entity"
    )
    kill_pool = [
        row
        for row in all811
        if row.get("task_type") == "kill_entity"
        and str(row.get("file")) in bare_by_file
        and not bool(bare_by_file[str(row.get("file"))].get("success"))
        and _target(str(row.get("task_name"))) not in knowledge_source_targets
    ]
    if (
        len(mine_pool) < n_per_family
        or len(kill_pool) < n_per_family
        or len(craft_pool) < n_per_family
    ):
        raise RuntimeError(
            f"insufficient pools: mine={len(mine_pool)} kill={len(kill_pool)} "
            f"craft={len(craft_pool)} "
            f"need={n_per_family}"
        )

    rng = random.Random(int(sample_seed))
    mine = sorted(rng.sample(mine_pool, n_per_family), key=lambda row: int(row["index"]))
    kill = sorted(rng.sample(kill_pool, n_per_family), key=lambda row: int(row["index"]))
    craft = sorted(rng.sample(craft_pool, n_per_family), key=lambda row: int(row["index"]))
    task_out = out_root / "tasks"
    task_out.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, Any]] = []
    for row in mine + kill + craft:
        src = OPENHA811_TASK_DIR / str(row["file"])
        task = _load_json(src)
        original_seed = task.get("seed")
        generalization_axis = "unseen_task_target"
        new_seed = original_seed
        if row.get("task_type") == "kill_entity":
            new_seed = _new_world_seed(sample_seed, str(row.get("task_name")))
            task["seed"] = new_seed
            if isinstance(task.get("openha_raw"), dict):
                task["openha_raw"]["seed"] = new_seed
            generalization_axis = "unseen_world_seed"
        dst = task_out / str(row["file"])
        _write_json(dst, task)
        samples.append(
            {
                "sample_id": len(samples),
                "index811": int(row["index"]),
                "file": str(row["file"]),
                "task_name": row.get("task_name"),
                "task_type": row.get("task_type"),
                "task_path": str(dst.resolve()),
                "source_task_path": str(src),
                "generalization_axis": generalization_axis,
                "in_dev149_by_task_name": str(row.get("task_name")) in dev_names,
                "promoted_knowledge_source_target": (
                    _target(str(row.get("task_name"))) in knowledge_source_targets
                ),
                "historical_bare_success": False,
                "original_world_seed": original_seed,
                "eval_world_seed": new_seed,
            }
        )

    manifest = {
        "sample_seed": int(sample_seed),
        "n_per_family": int(n_per_family),
        "n_samples": len(samples),
        "sampling": "deterministic random sample from historical Bare failures",
        "mine_protocol": "task_name excluded from DEV149",
        "craft_protocol": "task_name excluded from DEV149",
        "kill_protocol": (
            "DEV149 contains all 49 OpenHA-811 kill targets; evaluate new world "
            "seeds and exclude promoted-knowledge source targets"
        ),
        "mine_pool_size": len(mine_pool),
        "kill_pool_size": len(kill_pool),
        "craft_pool_size": len(craft_pool),
        "excluded_kill_knowledge_source_targets": sorted(knowledge_source_targets),
        "samples": samples,
    }
    _write_json(sample_path, manifest)
    return manifest


def _task_instruction(task_path: Path) -> str:
    task = _load_json(task_path)
    for key in ("instruction", "task", "goal", "prompt"):
        if task.get(key):
            return str(task[key])
    return f"Complete the Minecraft task: {task.get('task_name', task_path.stem)}"


def _bind_common_runtime(
    h: HarnessSpec,
    *,
    task_path: Path,
    image_dir: Path,
    temperature: float,
    protocol: str,
) -> None:
    h.runtime.backend = "minestudio"
    h.runtime.vla_mode = "hf"
    h.runtime.vla_model_path = VLA_PATH
    h.runtime.vla_device = "cuda"
    h.runtime.vla_dtype = "bfloat16"
    h.runtime.vla_protocol = protocol
    h.runtime.vla_temperature = float(temperature)
    h.runtime.vla_do_sample = bool(temperature > 0)
    h.runtime.action_chunk_len = 4
    h.runtime.max_steps = 100
    h.runtime.ticks_per_action = 4
    h.runtime.checkpoint_every = 8
    h.runtime.task_config = str(task_path)
    h.runtime.img_save_dir = str(image_dir)
    h.runtime.instruction = _task_instruction(task_path)
    h.runtime.success_reward_thresh = 0.5
    h.runtime.soft_rollback = True

    # Immutable execution harness shared by Auto-K and our method.
    h.probe.enabled = True
    h.probe.every_n_steps = 4
    h.probe.budget_per_episode = 8
    h.probe.action_pool = [
        "forward", "attack", "forward_attack", "turn_left", "look_up", "look_down"
    ]
    h.probe.probe_periodic = False
    h.probe.probe_on_fail = True
    h.probe.probe_on_stall = True
    h.probe.stall_steps = 8
    h.probe.warmup_probes = 0
    h.memory.enabled = True
    h.memory.inject_into_prompt = True
    h.verify.enabled = False
    h.verify.prefer_memory_action = False
    h.recover.enabled = True
    h.recover.recover_on_stall = True
    h.recover.stall_steps = 12
    h.recover.max_retries = 2


def _make_harness(
    condition: str,
    *,
    fixed_harness: Path,
    auto_k_harness: Optional[Path],
    adaptive_harness: Optional[Path],
    auto_k_root: Path,
    adaptive_root: Path,
    task_path: Path,
    image_dir: Path,
    temperature: float,
    protocol: str,
) -> Tuple[HarnessSpec, Dict[str, Any]]:
    harness_path = fixed_harness
    if condition == "auto_k" and auto_k_harness is not None:
        harness_path = auto_k_harness
    elif condition == "auto_k_adaptive" and adaptive_harness is not None:
        harness_path = adaptive_harness
    h = HarnessSpec.load(str(harness_path))
    _bind_common_runtime(
        h,
        task_path=task_path,
        image_dir=image_dir,
        temperature=temperature,
        protocol=protocol,
    )
    injection: Dict[str, Any] = {
        "fixed_harness": str(harness_path.resolve()),
        "knowledge_root": None,
        "probe_policy": "disabled",
        "probe_bandit": None,
        "knowledge_policy": "disabled",
        "knowledge_bandit": None,
        "promoted_skills": [],
        "writeback": False,
    }

    task_type = ""
    for candidate in ("mine_block", "kill_entity", "craft_item"):
        if candidate in task_path.stem:
            task_type = candidate
            break

    if condition == "bare":
        h.name = "bare_frozen_vla"
        h.probe.enabled = False
        h.knowledge_probe.enabled = False
        h.memory.enabled = False
        h.memory.inject_into_prompt = False
        h.memory.write_success = False
        h.memory.write_failure = False
        h.memory.write_probe = False
        h.verify.enabled = False
        h.verify.prefer_memory_action = False
        h.recover.enabled = False
        h.meta = {
            "condition": condition,
            "frozen_eval": True,
            "current_task_type": task_type,
            "current_target": _target(task_path.stem),
        }
        return h, injection

    source_root = auto_k_root if condition == "auto_k" else adaptive_root
    skills = overlay_promoted_skills(h, str(source_root))
    injection["knowledge_root"] = str(source_root.resolve())
    injection["promoted_skills"] = [str(row.get("skill_id")) for row in skills]
    h.meta = dict(h.meta or {})
    h.meta.update(
        {
            "condition": condition,
            "frozen_eval": True,
            "writeback": False,
            "current_task_type": task_type,
            "current_target": _target(task_path.stem),
        }
    )

    if condition == "auto_k":
        h.name = "fixed_harness_auto_k_prior"
        h.probe.selection_strategy = "coverage"
        h.probe.bandit_state_path = ""
        h.knowledge_probe.enabled = False
        h.knowledge_probe.bandit_state_path = ""
        injection["probe_policy"] = "coverage"
        injection["knowledge_policy"] = "promoted_skill_binding"
        return h, injection

    if condition != "auto_k_adaptive":
        raise ValueError(f"unknown condition: {condition}")
    probe_state = adaptive_root / "probe_bandit.json"
    knowledge_state = adaptive_root / "knowledge_bandit.json"
    if not probe_state.is_file() or not knowledge_state.is_file():
        raise FileNotFoundError(
            f"missing frozen adaptive priors: {probe_state}, {knowledge_state}"
        )
    h.name = "fixed_harness_auto_k_adaptive_prior"
    h.probe.selection_strategy = "ucb"
    h.probe.bandit_state_path = str(probe_state.resolve())
    h.knowledge_probe.enabled = True
    h.knowledge_probe.selection_strategy = "ucb"
    h.knowledge_probe.bandit_state_path = str(knowledge_state.resolve())
    injection.update(
        {
            "probe_policy": "contextual_ucb",
            "probe_bandit": str(probe_state.resolve()),
            "knowledge_policy": "contextual_ucb",
            "knowledge_bandit": str(knowledge_state.resolve()),
        }
    )
    return h, injection


def _done_keys(path: Path) -> set[Tuple[str, str]]:
    return {
        (str(row.get("file")), str(row.get("condition")))
        for row in _read_jsonl([path])
        if row.get("file") and row.get("condition")
    }


def _episode_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    ep = (metrics.get("episodes") or [{}])[0]
    meta = ep.get("meta") or {}
    probe_events = list(meta.get("probe_events") or [])
    knowledge_events = list(meta.get("knowledge_events") or [])
    return {
        "success": bool(ep.get("success")),
        "steps": ep.get("steps"),
        "reward": ep.get("reward"),
        "n_probe": ep.get("n_probe"),
        "n_recover": ep.get("n_recover"),
        "n_memory_write": ep.get("n_memory_write"),
        "cascade_fail": ep.get("cascade_fail"),
        "probe_actions": [event.get("action") for event in probe_events],
        "probe_contexts": [event.get("context") for event in probe_events],
        "knowledge_ids": [event.get("knowledge_id") for event in knowledge_events],
        "knowledge_contexts": [event.get("context") for event in knowledge_events],
        "probe_events": probe_events,
        "knowledge_events": knowledge_events,
        "score": metrics.get("score"),
    }


def run_worker(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    manifest = build_sample_manifest(
        out_root,
        sample_seed=args.sample_seed,
        n_per_family=args.n_per_family,
        auto_k_root=Path(args.auto_k_root),
        adaptive_root=Path(args.adaptive_root),
    )
    samples = [
        row
        for i, row in enumerate(manifest["samples"])
        if i % int(args.workers) == int(args.worker_rank)
    ]
    shard = out_root / f"shard_w{int(args.worker_rank)}"
    shard.mkdir(parents=True, exist_ok=True)
    results_path = shard / "results.jsonl"
    done = _done_keys(results_path)
    shared_vla: Optional[Any] = None
    print(
        f"[worker {args.worker_rank}/{args.workers}] samples={len(samples)} "
        f"done={len(done)} temperature={args.temperature} seed={args.episode_seed}",
        flush=True,
    )

    conditions = _selected_conditions(args.conditions)
    for sample in samples:
        task_path = Path(sample["task_path"])
        for condition in conditions:
            key = (str(sample["file"]), condition)
            if key in done:
                continue
            condition_dir = shard / "episodes" / Path(str(sample["file"])).stem / condition
            condition_dir.mkdir(parents=True, exist_ok=True)
            h, injection = _make_harness(
                condition,
                fixed_harness=Path(args.fixed_harness),
                auto_k_harness=Path(args.auto_k_harness) if args.auto_k_harness else None,
                adaptive_harness=Path(args.adaptive_harness) if args.adaptive_harness else None,
                auto_k_root=Path(args.auto_k_root),
                adaptive_root=Path(args.adaptive_root),
                task_path=task_path,
                image_dir=condition_dir / "frames",
                temperature=args.temperature,
                protocol=args.protocol,
            )
            h.save(str(condition_dir / "harness.json"))
            _write_json(condition_dir / "injection.json", injection)
            _seed_everything(int(args.episode_seed))
            _configure_shared_vla(shared_vla, h)
            started = time.time()
            error = None
            metrics: Dict[str, Any] = {}
            try:
                runtime = HarnessRuntime(h, seed=int(args.episode_seed), vla=shared_vla)
                if shared_vla is None:
                    shared_vla = runtime.vla
                    print(f"[worker {args.worker_rank}] VLA ready", flush=True)
                metrics = runtime.evaluate_on_seeds(
                    [int(args.episode_seed)], persist_memory=False
                )
                outcome = _episode_summary(metrics)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                outcome = {
                    "success": False,
                    "steps": None,
                    "reward": None,
                    "n_probe": None,
                    "n_recover": None,
                    "probe_actions": [],
                    "knowledge_ids": [],
                }
                traceback.print_exc()
            _write_json(condition_dir / "metrics.json", metrics)
            record = {
                **{key: sample.get(key) for key in (
                    "sample_id", "index811", "file", "task_name", "task_type",
                    "generalization_axis", "original_world_seed", "eval_world_seed",
                )},
                "condition": condition,
                "episode_seed": int(args.episode_seed),
                "temperature": float(args.temperature),
                "protocol": str(args.protocol),
                "frozen_vla": True,
                "fixed_harness": condition != "bare",
                "search_on_test": False,
                "bandit_writeback": False,
                "injection": injection,
                **outcome,
                "elapsed_s": round(time.time() - started, 2),
                "error": error,
                "timestamp": time.time(),
            }
            with results_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            done.add(key)
            print(
                f"[worker {args.worker_rank}] {sample['file']} {condition} "
                f"success={record['success']} steps={record['steps']} "
                f"probe={record['n_probe']} knowledge={record['knowledge_ids']} "
                f"elapsed={record['elapsed_s']}s error={error}",
                flush=True,
            )
    return 0


def _summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_condition: Dict[str, Any] = {}
    by_family: Dict[str, Any] = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row.get("condition") == condition]
        ok = sum(bool(row.get("success")) for row in subset)
        by_condition[condition] = {
            "n": len(subset),
            "success": ok,
            "success_rate": ok / len(subset) if subset else None,
            "errors": sum(bool(row.get("error")) for row in subset),
        }
        for family in ("mine_block", "kill_entity", "craft_item"):
            part = [row for row in subset if row.get("task_type") == family]
            pok = sum(bool(row.get("success")) for row in part)
            by_family.setdefault(family, {})[condition] = {
                "n": len(part),
                "success": pok,
                "success_rate": pok / len(part) if part else None,
            }
    paired: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        paired.setdefault(str(row.get("file")), {})[str(row.get("condition"))] = row
    complete = [case for case in paired.values() if all(c in case for c in CONDITIONS)]
    auto_rescues = sum(
        not case["bare"].get("success") and case["auto_k"].get("success")
        for case in complete
    )
    adaptive_rescues = sum(
        not case["bare"].get("success") and case["auto_k_adaptive"].get("success")
        for case in complete
    )
    adaptive_only = sum(
        not case["bare"].get("success")
        and not case["auto_k"].get("success")
        and case["auto_k_adaptive"].get("success")
        for case in complete
    )
    return {
        "n_rows": len(rows),
        "n_paired_complete": len(complete),
        "by_condition": by_condition,
        "by_family": by_family,
        "auto_k_rescues_over_bare": auto_rescues,
        "adaptive_rescues_over_bare": adaptive_rescues,
        "strict_adaptive_only": adaptive_only,
        "updated_at": time.time(),
    }


def merge_results(out_root: Path, workers: int) -> Dict[str, Any]:
    rows = _read_jsonl(
        out_root / f"shard_w{rank}" / "results.jsonl" for rank in range(workers)
    )
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        by_key[(str(row.get("file")), str(row.get("condition")))] = row
    merged = sorted(
        by_key.values(),
        key=lambda row: (int(row.get("sample_id") or 0), CONDITIONS.index(row["condition"])),
    )
    merged_path = out_root / "results.jsonl"
    tmp = merged_path.with_suffix(f".jsonl.tmp.{os.getpid()}")
    with tmp.open("w") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, merged_path)
    summary = _summary(merged)
    _write_json(out_root / "summary.json", summary)
    return summary


def _python() -> str:
    override = str(os.environ.get("AUTO_HARNESS_PYTHON") or "").strip()
    if override and Path(override).is_file():
        return override
    alaya = Path("python")
    return str(alaya) if alaya.is_file() else sys.executable


def _selected_conditions(value: str) -> Tuple[str, ...]:
    selected = tuple(x.strip() for x in str(value or "").split(",") if x.strip())
    if not selected:
        raise ValueError("--conditions must select at least one condition")
    unknown = [x for x in selected if x not in CONDITIONS]
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}; choices={CONDITIONS}")
    if len(set(selected)) != len(selected):
        raise ValueError(f"duplicate conditions: {selected}")
    return selected


def _forward_args(args: argparse.Namespace) -> List[str]:
    return [
        "--out-root", str(Path(args.out_root).resolve()),
        "--n-per-family", str(args.n_per_family),
        "--sample-seed", str(args.sample_seed),
        "--episode-seed", str(args.episode_seed),
        "--temperature", str(args.temperature),
        "--protocol", str(args.protocol),
        "--fixed-harness", str(Path(args.fixed_harness).resolve()),
        "--auto-k-harness", str(Path(args.auto_k_harness).resolve()),
        "--adaptive-harness", str(Path(args.adaptive_harness).resolve()),
        "--auto-k-root", str(Path(args.auto_k_root).resolve()),
        "--adaptive-root", str(Path(args.adaptive_root).resolve()),
        "--conditions", str(args.conditions),
        "--workers", str(args.workers),
        "--cuda-device", str(args.cuda_device),
    ]


def _worker_cuda_device(spec: str, rank: int) -> str:
    devices = [item.strip() for item in str(spec).split(",") if item.strip()]
    if not devices:
        raise ValueError("--cuda-device must name at least one CUDA device")
    return devices[int(rank) % len(devices)]


def supervise(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    py = _python()
    procs: List[Tuple[int, subprocess.Popen[Any], Any]] = []
    for rank in range(int(args.workers)):
        log_path = out_root / f"worker_{rank}.log"
        log_handle = log_path.open("a")
        cmd = [
            py, "-u", "-m", "curriculum.eval_fixed_knowledge_probe_generalization",
            *_forward_args(args), "--worker-rank", str(rank),
        ]
        env = os.environ.copy()
        worker_cuda = _worker_cuda_device(args.cuda_device, rank)
        env["CUDA_VISIBLE_DEVICES"] = worker_cuda
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        procs.append((rank, proc, log_handle))
        print(
            f"[supervisor] worker={rank} cuda={worker_cuda} "
            f"pid={proc.pid} log={log_path}",
            flush=True,
        )
    state = {
        "supervisor_pid": os.getpid(),
        "workers": [{"rank": rank, "pid": proc.pid} for rank, proc, _ in procs],
        "status": "running",
        "started_at": time.time(),
    }
    _write_json(out_root / "run_state.json", state)
    codes = []
    for rank, proc, log_handle in procs:
        code = proc.wait()
        log_handle.close()
        codes.append(code)
        print(f"[supervisor] worker={rank} exit={code}", flush=True)
        merge_results(out_root, int(args.workers))
    summary = merge_results(out_root, int(args.workers))
    state.update(
        {
            "status": "completed" if all(code == 0 for code in codes) else "failed",
            "worker_exit_codes": codes,
            "finished_at": time.time(),
            "summary": summary,
        }
    )
    _write_json(out_root / "run_state.json", state)
    return 0 if all(code == 0 for code in codes) else 1


def launch(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = build_sample_manifest(
        out_root,
        sample_seed=args.sample_seed,
        n_per_family=args.n_per_family,
        auto_k_root=Path(args.auto_k_root),
        adaptive_root=Path(args.adaptive_root),
    )
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    supervisor_log = (out_root / "supervisor.log").open("a")
    cmd = [
        _python(), "-u", "-m", "curriculum.eval_fixed_knowledge_probe_generalization",
        *_forward_args(args), "--supervise",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=str(_ROOT),
        env=env,
        stdout=supervisor_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    supervisor_log.close()
    _write_json(
        out_root / "launch.json",
        {
            "pid": proc.pid,
            "command": cmd,
            "launched_at": time.time(),
            "workers": int(args.workers),
            "cuda_device": str(args.cuda_device),
        },
    )
    print(
        json.dumps(
            {
                "status": "submitted",
                "pid": proc.pid,
                "out_root": str(out_root),
                "workers": int(args.workers),
                "n_samples": manifest["n_samples"],
                "n_episodes": manifest["n_samples"] * len(_selected_conditions(args.conditions)),
            },
            indent=2,
        )
    )
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--n-per-family", type=int, default=10)
    ap.add_argument("--sample-seed", type=int, default=20260922)
    ap.add_argument("--episode-seed", type=int, default=404)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--protocol", default="legacy_k1_t0")
    ap.add_argument("--fixed-harness", default=str(AUTO_K_ROOT / "prior_harness_base.json"))
    ap.add_argument("--auto-k-harness", default=str(AUTO_K_ROOT / "prior_harness_best.json"))
    ap.add_argument("--adaptive-harness", default=str(ADAPTIVE_ROOT / "prior_harness_best.json"))
    ap.add_argument("--auto-k-root", default=str(AUTO_K_ROOT))
    ap.add_argument("--adaptive-root", default=str(ADAPTIVE_ROOT))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--worker-rank", type=int, default=None)
    ap.add_argument("--cuda-device", default="0")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--supervise", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.launch or args.dry_run:
        return launch(args)
    if args.supervise:
        return supervise(args)
    if args.worker_rank is None:
        raise SystemExit("use --launch, --dry-run, or --worker-rank")
    return run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
