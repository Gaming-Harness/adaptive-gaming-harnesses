"""Replay a strict bare / ordinary-auto / probing-evolved Sandbox triple.

The three conditions use the same frozen VLA, task, episode seed, decoding
configuration, and freshly reset episodic memory.  ``persist_memory=False``
also prevents a visual replay from updating the persisted probing bandits.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Optional

from curriculum.harness_runtime import HarnessRuntime
from curriculum.harness_schema import HarnessSpec


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _configure_shared_vla(vla: Any, harness: HarnessSpec) -> None:
    if vla is None or not hasattr(vla, "cfg"):
        return
    cfg = vla.cfg
    cfg.instruction = str(harness.runtime.instruction)
    cfg.action_chunks_len = int(harness.runtime.action_chunk_len)
    cfg.temperature = float(getattr(harness.runtime, "vla_temperature", 0.0))
    cfg.do_sample = bool(getattr(harness.runtime, "vla_do_sample", False))
    cfg.history_window = int(getattr(harness.runtime, "vla_history_window", 10) or 10)
    cfg.protocol = str(getattr(harness.runtime, "vla_protocol", "legacy_k1_t0"))


def _make_bare(source: HarnessSpec) -> HarnessSpec:
    h = copy.deepcopy(source)
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
    return h


def _run_one(
    harness: HarnessSpec,
    output: Path,
    seed: int,
    *,
    vla: Optional[Any],
    probe_bandit: str = "",
    knowledge_bandit: str = "",
) -> tuple[dict, Any]:
    h = copy.deepcopy(harness)
    h.runtime.img_save_dir = str(output / "sandbox_frames")
    if probe_bandit and h.probe.enabled:
        h.probe.bandit_state_path = probe_bandit
    if knowledge_bandit and h.knowledge_probe.enabled:
        h.knowledge_probe.bandit_state_path = knowledge_bandit
    output.mkdir(parents=True, exist_ok=True)
    h.save(str(output / "harness.json"))
    _seed_everything(seed)
    _configure_shared_vla(vla, h)
    runtime = HarnessRuntime(h, seed=seed, vla=vla)
    metrics = runtime.evaluate_on_seeds([seed], persist_memory=False)
    with open(output / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return metrics, runtime.vla


def _short(metrics: dict) -> dict:
    episode = (metrics.get("episodes") or [{}])[0]
    return {
        "success": bool(episode.get("success")),
        "steps": episode.get("steps"),
        "reward": episode.get("reward"),
        "n_probe": episode.get("n_probe"),
        "n_recover": episode.get("n_recover"),
        "score": metrics.get("score"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--auto-search-dir", type=Path, required=True)
    ap.add_argument("--probing-search-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--probe-bandit", default="")
    ap.add_argument("--knowledge-bandit", default="")
    args = ap.parse_args()

    auto_seed = HarnessSpec.load(str(args.auto_search_dir / "harness_seed.json"))
    auto_best = HarnessSpec.load(str(args.auto_search_dir / "harness_best.json"))
    probing_best = HarnessSpec.load(str(args.probing_search_dir / "harness_best.json"))
    bare = _make_bare(auto_seed)

    shared_vla = None
    results = {}
    for condition, harness in (
        ("bare", bare),
        ("auto", auto_best),
        ("probing", probing_best),
    ):
        metrics, shared_vla = _run_one(
            harness,
            args.output / condition,
            args.seed,
            vla=shared_vla,
            probe_bandit=args.probe_bandit if condition == "probing" else "",
            knowledge_bandit=args.knowledge_bandit if condition == "probing" else "",
        )
        results[condition] = _short(metrics)
        print(f"[triple] {args.name} seed={args.seed} {condition} {results[condition]}", flush=True)

    strict = (
        not results["bare"]["success"]
        and not results["auto"]["success"]
        and results["probing"]["success"]
    )
    summary = {
        "name": args.name,
        "seed": args.seed,
        "auto_search_dir": str(args.auto_search_dir),
        "probing_search_dir": str(args.probing_search_dir),
        "frozen_vla": True,
        "independent_memory": True,
        "bandit_writeback": False,
        "strict_001": strict,
        "conditions": results,
    }
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
