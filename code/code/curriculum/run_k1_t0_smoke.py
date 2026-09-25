#!/usr/bin/env python3
"""Wait for GPU headroom, then run K=1 temp=0 Auto-Harness smoke (cobble + stone_slab)."""
from __future__ import annotations

import os
import subprocess
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = "python"
CFGS = [
    "curriculum/configs/auto_harness_cobble_k1_t0.yaml",
    "curriculum/configs/auto_harness_stone_slab_k1_t0.yaml",
]
OUT_LOG = "curriculum/outputs/k1_t0_smoke/waiter.log"
MEM_FREE_MIB = 25000  # need ~one HF VLA slot


def _gpu_free_mib() -> int:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return max(int(x.strip()) for x in out.strip().splitlines() if x.strip())
    except Exception:
        return 0


def main() -> int:
    os.chdir(_ROOT)
    os.makedirs("curriculum/outputs/k1_t0_smoke", exist_ok=True)
    log = open(OUT_LOG, "a", buffering=1)
    def p(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")

    p(f"[k1_t0] waiter start need_free>={MEM_FREE_MIB}MiB")
    while True:
        free = _gpu_free_mib()
        p(f"[k1_t0] gpu_free={free}MiB")
        if free >= MEM_FREE_MIB:
            break
        time.sleep(120)

    rc_all = 0
    for cfg in CFGS:
        p(f"[k1_t0] START {cfg}")
        t0 = time.time()
        rc = subprocess.call(
            [PY, "-u", "-m", "curriculum.auto_harness_search", "--config", cfg],
            cwd=_ROOT,
        )
        p(f"[k1_t0] END {cfg} rc={rc} {((time.time()-t0)/60):.1f}min")
        rc_all = rc_all or rc
    p(f"[k1_t0] ALL DONE rc={rc_all}")
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
