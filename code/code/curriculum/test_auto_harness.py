#!/usr/bin/env python3
"""Full local test suite for AutoHarness (no API / no MineStudio).

  python -m curriculum.test_auto_harness
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from curriculum.auto_harness_search import (
    MUTATION_BANK,
    AutoHarnessSearch,
    apply_mutation,
    degraded_seed,
    run_from_config,
    strong_seed,
)
from curriculum.harness_runtime import HarnessRuntime
from curriculum.harness_schema import HarnessSpec, default_seed_harness

OUT = "curriculum/outputs/auto_harness_full_test"


def main() -> int:
    if os.path.isdir(OUT):
        shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)

    results = {"passed": [], "failed": []}

    def check(name: str, cond: bool, detail: str = "") -> None:
        if cond:
            results["passed"].append(name)
            print(f"[PASS] {name}" + (f" — {detail}" if detail else ""), flush=True)
        else:
            results["failed"].append({"name": name, "detail": detail})
            print(f"[FAIL] {name} — {detail}", flush=True)

    # 1) schema
    h = default_seed_harness()
    p = os.path.join(OUT, "seed_roundtrip.json")
    h.save(p)
    h2 = HarnessSpec.load(p)
    check("schema_roundtrip", h.fingerprint() == h2.fingerprint())
    try:
        py = os.path.join(OUT, "seed_roundtrip.yaml")
        h.save(py)
        h3 = HarnessSpec.load(py)
        check("schema_yaml_roundtrip", h.fingerprint() == h3.fingerprint())
    except Exception as e:
        check("schema_yaml_roundtrip", False, str(e))

    # 2) baselines
    m_deg = HarnessRuntime(degraded_seed(), seed=0).evaluate(8, persist_memory=True)
    m_stg = HarnessRuntime(strong_seed(), seed=0).evaluate(8, persist_memory=True)
    check(
        "strong_beats_degraded",
        m_stg["success_rate"] >= m_deg["success_rate"] + 0.25
        or m_stg["score"] >= m_deg["score"] + 0.25,
        f"deg succ={m_deg['success_rate']:.2f} score={m_deg['score']:.3f} | "
        f"stg succ={m_stg['success_rate']:.2f} score={m_stg['score']:.3f}",
    )
    with open(os.path.join(OUT, "baseline_degraded.json"), "w") as f:
        json.dump(m_deg, f, indent=2)
    with open(os.path.join(OUT, "baseline_strong.json"), "w") as f:
        json.dump(m_stg, f, indent=2)

    # 3) ablations
    ablation = {}
    for comp, field, val, label in [
        ("recover", "enabled", False, "no_recover"),
        ("verify", "enabled", False, "no_verify"),
        ("probe", "enabled", False, "no_probe"),
        ("memory", "enabled", False, "no_memory"),
    ]:
        hh = strong_seed()
        getattr(hh, comp).__setattr__(field, val)
        if label == "no_memory":
            hh.memory.inject_into_prompt = False
        mm = HarnessRuntime(hh, seed=0).evaluate(8, persist_memory=True)
        ablation[label] = {
            "success_rate": mm["success_rate"],
            "score": mm["score"],
            "cascade_rate": mm["cascade_rate"],
        }
        print(
            f"  ablation {label}: succ={mm['success_rate']:.2f} "
            f"score={mm['score']:.3f} cascade={mm['cascade_rate']:.2f}",
            flush=True,
        )
    check(
        "ablation_no_recover_hurts",
        ablation["no_recover"]["success_rate"] <= m_stg["success_rate"] - 0.25
        or ablation["no_recover"]["score"] <= m_stg["score"] - 0.25,
        f"full={m_stg['success_rate']:.2f} vs no_recover={ablation['no_recover']['success_rate']:.2f}",
    )
    with open(os.path.join(OUT, "ablations.json"), "w") as f:
        json.dump(
            {
                "full": {
                    "success_rate": m_stg["success_rate"],
                    "score": m_stg["score"],
                },
                **ablation,
            },
            f,
            indent=2,
        )

    # 4) mutation plumbing
    mut = next(m for m in MUTATION_BANK if m["component"] == "recover" and m["op"] == "enable")
    hd = degraded_seed()
    hc, man = apply_mutation(hd, mut)
    check("mutation_changes_fp", hd.fingerprint() != hc.fingerprint())
    check(
        "manifest_fields",
        all(k in man.to_dict() for k in ("component", "op", "claim", "expect_delta")),
    )

    # 5) multi-seed search
    summaries = {}
    for seed in (0, 1, 2):
        logdir = os.path.join(OUT, f"search_seed{seed}")
        summary = AutoHarnessSearch(
            logdir=logdir,
            seed_harness=degraded_seed(),
            n_rounds=10,
            n_eval_episodes=6,
            seed=seed,
        ).run()
        summaries[str(seed)] = summary
        check(
            f"search_improves_seed{seed}",
            summary["best_score"] > summary["seed_score"] + 0.05,
            f"seed={summary['seed_score']:.3f} → best={summary['best_score']:.3f} "
            f"holdout_succ={summary['holdout_success_rate']:.2f} "
            f"acc={summary['accepted']} rev={summary['reverted']}",
        )

    # 6) config entry
    try:
        from omegaconf import OmegaConf

        c = OmegaConf.load("curriculum/configs/auto_harness.yaml")
        c.logdir = os.path.join(OUT, "from_config")
        c.n_rounds = 8
        c.n_eval_episodes = 5
        c.seed = 3
        tmp = os.path.join(OUT, "_runtime_cfg.yaml")
        OmegaConf.save(c, tmp)
        sum_cfg = run_from_config(tmp)
        check(
            "run_from_config",
            sum_cfg["best_score"] >= sum_cfg["seed_score"],
            f"seed={sum_cfg['seed_score']:.3f} best={sum_cfg['best_score']:.3f}",
        )
    except Exception as e:
        sum_cfg = {}
        check("run_from_config", False, str(e))

    # 7) CLI
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "curriculum.auto_harness_search",
            "--smoke",
            "--rounds",
            "8",
            "--seed",
            "4",
            "--logdir",
            os.path.join(OUT, "cli_smoke"),
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    check("cli_smoke_exit0", r.returncode == 0, (r.stderr or "")[-400:])
    check("cli_smoke_ok_msg", "smoke_auto_harness] OK" in r.stdout, r.stdout[-200:])

    # 8) run_curriculum
    r2 = subprocess.run(
        [sys.executable, "-m", "curriculum.run_curriculum", "smoke_auto_harness"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    check("run_curriculum_smoke", r2.returncode == 0, (r2.stderr or r2.stdout)[-300:])

    report = {
        "n_passed": len(results["passed"]),
        "n_failed": len(results["failed"]),
        "passed": results["passed"],
        "failed": results["failed"],
        "baselines": {
            "degraded": {
                k: m_deg[k]
                for k in ("score", "success_rate", "cascade_rate", "avg_reward")
            },
            "strong": {
                k: m_stg[k]
                for k in ("score", "success_rate", "cascade_rate", "avg_reward")
            },
        },
        "ablations": ablation,
        "searches": summaries,
        "from_config": sum_cfg,
    }
    with open(os.path.join(OUT, "TEST_REPORT.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n========== SUMMARY ==========", flush=True)
    print(f"PASS {report['n_passed']}  FAIL {report['n_failed']}", flush=True)
    print(f"degraded succ={m_deg['success_rate']:.2f}  strong succ={m_stg['success_rate']:.2f}", flush=True)
    if results["failed"]:
        for x in results["failed"]:
            print(" -", x, flush=True)
        return 1
    print(f"report → {OUT}/TEST_REPORT.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
