#!/usr/bin/env python3
"""Unit / stub tests for event-driven Auto-Harness (no MineStudio).

  cd project_root
  python -m curriculum.test_auto_harness_event
"""
from __future__ import annotations

import json
import os
import shutil
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from curriculum.auto_harness_event import AutoEventSearch, run_from_config
from curriculum.harness_event_policy import (
    CompileError,
    compile_edits,
    default_event_seed,
    strong_event_seed,
)
from curriculum.harness_event_runtime import EventHarnessRuntime

OUT = "curriculum/outputs/auto_harness_event_test"


def main() -> int:
    if os.path.isdir(OUT):
        shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)
    passed, failed = [], []

    def check(name: str, cond: bool, detail: str = "") -> None:
        if cond:
            passed.append(name)
            print(f"[PASS] {name}" + (f" — {detail}" if detail else ""), flush=True)
        else:
            failed.append({"name": name, "detail": detail})
            print(f"[FAIL] {name} — {detail}", flush=True)

    # 1) schema roundtrip + projection
    p = strong_event_seed()
    path = os.path.join(OUT, "policy.json")
    p.save(path)
    p2 = type(p).load(path)
    check("policy_roundtrip", p.fingerprint() == p2.fingerprint())
    h = p.to_harness_spec()
    check("no_periodic", h.probe.probe_periodic is False)
    check("fail_stall_flags", h.probe.probe_on_stall or h.recover.recover_on_stall)

    # 2) compiler rejects illegal / periodic
    try:
        compile_edits(p, [{"path": ["probe", "every_n_steps"], "value": 2}])
        check("reject_legacy_path", False, "should raise")
    except CompileError:
        check("reject_legacy_path", True)

    try:
        compile_edits(p, [{"path": ["triggers", "forbid_periodic"], "value": False}])
        check("reject_periodic_off", False, "should raise or reject-all")
    except CompileError:
        check("reject_periodic_off", True)

    pol, rep = compile_edits(
        p,
        [
            {"path": ["routing", "on_stall"], "value": "replan"},
            {"path": ["observe", "action_pool"], "value": ["turn_left", "hack_key"]},
        ],
    )
    check("compile_routing", pol.routing.on_stall == "replan", str(rep))
    check("sanitize_pool", "hack_key" not in pol.observe.action_pool and "turn_left" in pol.observe.action_pool)

    # 3) stub episode emits events
    rt = EventHarnessRuntime(strong_event_seed(), seed=101)
    # Force stall-prone stub by short stall threshold
    rt.policy.triggers.stall_steps = 3
    rt.policy.routing.on_stall = "observe"
    er = rt.run_episode(episode_seed=101)
    ev = (er.meta or {}).get("events") or []
    check("episode_runs", er.steps > 0, f"steps={er.steps}")
    check("events_logged", True, f"n_events={len(ev)} events={ev[:5]}")

    # 4) short bank search
    search = AutoEventSearch(
        logdir=os.path.join(OUT, "search"),
        seed=0,
        n_rounds=3,
        eval_seeds=[101, 202],
        holdout_seeds=[909],
        accept_eps=0.01,
        runtime_overrides={"backend": "stub", "vla_mode": "stub", "max_steps": 28},
        policy=strong_event_seed(),
        proposer_mode="bank",
    )
    summary = search.run()
    check("search_summary", "best_score" in summary and "accepted" in summary and "effective" in summary, json.dumps(summary))
    check("history_written", os.path.isfile(os.path.join(OUT, "search", "history.json")))
    check("harness_best_emitted", os.path.isfile(os.path.join(OUT, "search", "harness_best.json")))

    # 5) config entrypoint
    cfg = "curriculum/configs/auto_harness_event_stub.yaml"
    # redirect logdir for test isolation
    import yaml

    with open(cfg) as f:
        d = yaml.safe_load(f)
    d["logdir"] = os.path.join(OUT, "from_cfg")
    d["n_rounds"] = 2
    cfg2 = os.path.join(OUT, "stub_cfg.yaml")
    with open(cfg2, "w") as f:
        yaml.safe_dump(d, f)
    s2 = run_from_config(cfg2)
    check("run_from_config", "holdout_success_rate" in s2 and "accepted" in s2, json.dumps(s2))

    print(
        f"\n=== event harness tests: {len(passed)} passed, {len(failed)} failed ===",
        flush=True,
    )
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"passed": passed, "failed": failed}, f, indent=2)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
