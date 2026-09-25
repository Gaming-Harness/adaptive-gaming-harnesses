#!/usr/bin/env python3
"""Sequential K=1 temp=0 Auto-Harness on oak / cobble / stone_slab (fair compare)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = "python"
CFGS = [
    ("oak", "curriculum/configs/auto_harness_oak_k1_t0.yaml", "curriculum/outputs/auto_harness_oak_k1_t0"),
    ("cobble", "curriculum/configs/auto_harness_cobble_k1_t0.yaml", "curriculum/outputs/auto_harness_cobble_k1_t0"),
    ("stone_slab", "curriculum/configs/auto_harness_stone_slab_k1_t0.yaml", "curriculum/outputs/auto_harness_stone_slab_k1_t0"),
]
OUT_LOG = "curriculum/outputs/k1_t0_compare3/run.log"


def _fresh(logdir: str) -> None:
    if os.path.isdir(logdir):
        # keep nothing — clean restart for fair compare
        for name in os.listdir(logdir):
            p = os.path.join(logdir, name)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    os.remove(p)
                except OSError:
                    pass
    os.makedirs(logdir, exist_ok=True)


def main() -> int:
    os.chdir(_ROOT)
    os.makedirs("curriculum/outputs/k1_t0_compare3", exist_ok=True)
    log = open(OUT_LOG, "w", buffering=1)

    def p(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")

    p("[compare3] kill-check done; start oak/cobble/stone_slab K=1 temp=0")
    rc_all = 0
    for name, cfg, logdir in CFGS:
        _fresh(logdir)
        p(f"[compare3] START {name} cfg={cfg}")
        t0 = time.time()
        rc = subprocess.call(
            [PY, "-u", "-m", "curriculum.auto_harness_search", "--config", cfg],
            cwd=_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        mins = (time.time() - t0) / 60.0
        summary = os.path.join(logdir, "summary.json")
        p(
            f"[compare3] END {name} rc={rc} "
            f"summary={'ok' if os.path.isfile(summary) else 'MISSING'} "
            f"{mins:.1f}min"
        )
        rc_all = rc_all or rc
    p(f"[compare3] ALL DONE rc={rc_all}")
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
