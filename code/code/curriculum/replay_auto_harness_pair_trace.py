"""Replay the frozen netherrack H0/H* pair with auditable per-tick traces."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple

from curriculum.harness_runtime import HarnessRuntime
from curriculum.harness_schema import HarnessSpec


def run_one(
    harness_path: Path,
    output: Path,
    seed: int,
    *,
    vla: Optional[Any] = None,
    max_steps: Optional[int] = None,
) -> Tuple[dict, Any]:
    harness = HarnessSpec.load(str(harness_path))
    if max_steps is not None:
        harness.runtime.max_steps = int(max_steps)
    # The frozen weights can be shared across cases, but decoding/runtime
    # settings belong to the current harness and must not leak from the first
    # case loaded in the process.
    if vla is not None and hasattr(vla, "cfg"):
        cfg = vla.cfg
        cfg.instruction = str(harness.runtime.instruction)
        cfg.action_chunks_len = int(harness.runtime.action_chunk_len)
        cfg.temperature = float(getattr(harness.runtime, "vla_temperature", 0.0))
        cfg.do_sample = bool(getattr(harness.runtime, "vla_do_sample", False))
        cfg.history_window = int(getattr(harness.runtime, "vla_history_window", 10) or 10)
        cfg.protocol = str(getattr(harness.runtime, "vla_protocol", "minecraft_v1"))
    harness.runtime.img_save_dir = str(output / "sandbox_frames")
    output.mkdir(parents=True, exist_ok=True)
    harness.save(str(output / "harness.json"))
    runtime = HarnessRuntime(harness, seed=seed, vla=vla)
    result = runtime.evaluate_on_seeds([seed], persist_memory=False)
    with open(output / "metrics.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result, runtime.vla


def run_pair(
    search_dir: Path,
    output: Path,
    seed: int,
    condition: str,
    *,
    vla: Optional[Any] = None,
    max_steps: Optional[int] = None,
) -> Tuple[dict, Any]:
    results = {}
    if condition in ("both", "h0"):
        results["h0"], vla = run_one(
            search_dir / "harness_seed.json", output / "h0", seed,
            vla=vla, max_steps=max_steps
        )
    if condition in ("both", "hstar"):
        results["hstar"], vla = run_one(
            search_dir / "harness_best.json", output / "hstar", seed,
            vla=vla, max_steps=max_steps
        )
    summary = {"seed": seed, "search_dir": str(search_dir)}
    for name, metrics in results.items():
        summary[name] = {
            k: metrics.get(k) for k in ("success_rate", "score", "avg_reward")
        }
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary, vla


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--condition", choices=("both", "h0", "hstar"), default="both")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override each harness episode length (useful for short visual replays).",
    )
    parser.add_argument(
        "--max-actions-per-step",
        type=int,
        default=None,
        help="Cap decoded low-level actions per VLA decision for short visual replays.",
    )
    parser.add_argument(
        "--search-dir",
        type=Path,
        default=Path(
            "curriculum/outputs/auto_harness_hard_k1_t0_legacy_actionable/"
            "search_0002_mine_block_netherrack"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("curriculum/outputs/replay_netherrack_tick_trace"),
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="NAME=SEARCH_DIR",
        help=(
            "Run multiple search directories in one process while sharing the "
            "frozen VLA, e.g. --case orange=... --case poppy=... ."
        ),
    )
    args = parser.parse_args()
    os.environ["HARNESS_TRACE_TICKS"] = "1"
    if args.max_actions_per_step is not None:
        os.environ["HARNESS_MAX_ACTIONS_PER_STEP"] = str(args.max_actions_per_step)
    cases = []
    for raw in args.case:
        if "=" not in raw:
            parser.error(f"--case must be NAME=SEARCH_DIR, got: {raw}")
        name, path = raw.split("=", 1)
        if not name.strip() or not path.strip():
            parser.error(f"--case must be NAME=SEARCH_DIR, got: {raw}")
        cases.append((name.strip(), Path(path.strip())))
    if not cases:
        cases = [("", args.search_dir)]

    shared_vla = None
    combined = {}
    for name, search_dir in cases:
        output = args.output / name if name else args.output
        summary, shared_vla = run_pair(
            search_dir,
            output,
            args.seed,
            args.condition,
            vla=shared_vla,
            max_steps=args.max_steps,
        )
        combined[name or search_dir.name] = summary
    if len(cases) > 1:
        args.output.mkdir(parents=True, exist_ok=True)
        with open(args.output / "summary.json", "w") as f:
            json.dump({"seed": args.seed, "cases": combined}, f, indent=2)


if __name__ == "__main__":
    main()
