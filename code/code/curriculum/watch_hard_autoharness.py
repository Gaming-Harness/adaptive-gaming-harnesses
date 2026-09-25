#!/usr/bin/env python3
"""Watch bare hard_tasks.json; launch Auto-Harness LLM search on new fails.

Also can be given --tasks explicitly. Skips tasks already searched.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Optional, Set

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


def _write_cfg(task_file: str, out_cfg: str, logdir: str) -> None:
    task_path = os.path.join(TASK_DIR, task_file)
    body = f"""seed: 0
logdir: {logdir}
n_rounds: 3
n_eval_episodes: 2
eval_seeds: [101, 202]
holdout_seeds: [909]
accept_eps: 0.01
start: strong
backend: minestudio
vla_mode: hf
vla_model_path: {VLA}
vla_device: cuda
vla_dtype: bfloat16
action_chunk_len: 4
max_steps: 100
ticks_per_action: 4
checkpoint_every: 8
recover_max_retries: 2
probe_enabled: true
memory_enabled: true
verify_enabled: false
recover_enabled: true
prefer_memory_action: false
probe_budget: 8
probe_every_n: 4
probe_action_pool: [forward, attack, forward_attack, turn_left, look_up]
proposer_mode: llm
proposer_model: gemini-2.5-flash
prefer_components: [probe, recover]
task_config: {task_path}
img_save_dir: {logdir}/images
success_reward_thresh: 0.5
soft_rollback: true
"""
    os.makedirs(os.path.dirname(out_cfg) or ".", exist_ok=True)
    with open(out_cfg, "w") as f:
        f.write(body)


def _launch(task_file: str, out_root: str) -> Optional[int]:
    stem = task_file.replace(".json", "")
    logdir = os.path.join(out_root, f"search_{stem}")
    if os.path.isfile(os.path.join(logdir, "summary.json")):
        print(f"[watch] skip already done {task_file}", flush=True)
        return None
    cfg = os.path.join(out_root, f"cfg_{stem}.yaml")
    _write_cfg(task_file, cfg, logdir)
    os.makedirs(logdir, exist_ok=True)
    log = open(os.path.join(logdir, "run.log"), "w")
    proc = subprocess.Popen(
        [PY, "-u", "-m", "curriculum.auto_harness_search", "--config", cfg],
        cwd=_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    print(f"[watch] launched {task_file} pid={proc.pid} logdir={logdir}", flush=True)
    return proc.pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-json", default="curriculum/outputs/batch_openha_bare/hard_tasks.json")
    ap.add_argument("--out-root", default="curriculum/outputs/auto_harness_from_hard")
    ap.add_argument("--max-tasks", type=int, default=3)
    ap.add_argument("--poll-s", type=int, default=120)
    ap.add_argument("--tasks", nargs="*", default=None, help="explicit task json names")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    launched: Set[str] = set()
    while True:
        files = list(args.tasks or [])
        if not files and os.path.isfile(args.hard_json):
            with open(args.hard_json) as f:
                hard = json.load(f)
            files = [h["file"] for h in hard if h.get("file")]
        new = [f for f in files if f not in launched][: max(0, args.max_tasks - len(launched))]
        for f in new:
            _launch(f, args.out_root)
            launched.add(f)
        if args.once or len(launched) >= args.max_tasks:
            break
        time.sleep(int(args.poll_s))
    print(f"[watch] launched={sorted(launched)}", flush=True)


if __name__ == "__main__":
    main()