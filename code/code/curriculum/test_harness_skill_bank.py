#!/usr/bin/env python3
"""Unit tests for transferable skill bank (no GPU / no MineStudio)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from curriculum.harness_prior_merge import merge_harness_prior
from curriculum.harness_schema import HarnessSpec
from curriculum.harness_skill_bank import (
    abstract_text,
    bind_skill_recovery,
    extract_skill,
    hydrate_skill_bank,
    infer_skill_id,
    overlay_promoted_skills,
    parse_task_ref,
    upsert_skill,
)
from curriculum.harness_success_memory import format_success_memory_for_prompt


def main() -> int:
    failed = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        if cond:
            print(f"[PASS] {name}", flush=True)
        else:
            failed.append(name)
            print(f"[FAIL] {name} — {detail}", flush=True)

    ref = parse_task_ref("0020_mine_block_melon.json")
    check("parse_target", ref["target"] == "melon" and ref["task_type"] == "mine_block")
    heldout_ref = parse_task_ref("g000_mine_block_poppy_ws1309133838.json")
    check(
        "parse_prefixed_heldout_target",
        heldout_ref["task_type"] == "mine_block"
        and heldout_ref["target"] == "poppy ws1309133838",
        str(heldout_ref),
    )

    raw = (
        "If the target melon block is not centered, re-orient. "
        "<actions> mouseMove(0, 70) ; keyPress(a) ; mouseClick(left) </actions>"
    )
    abs_t = abstract_text(raw, target="melon")
    check("strip_actions", "<actions>" not in abs_t and "mouseMove" not in abs_t)
    check("strip_melon", "melon" not in abs_t.lower() and "{target}" in abs_t)

    sid = infer_skill_id(
        ["orientation", "target_loss"],
        diagnosis="looking away towards the sky",
        recovery=raw,
    )
    check("infer_reacquire", sid == "reacquire_and_center")
    check(
        "generic_skip",
        infer_skill_id([], recovery="Previous action failed. Return to the last successful checkpoint shown in memory, then retry the current subgoal with a different action.")
        is None,
    )

    tmp = tempfile.mkdtemp(prefix="skill_bank_")
    try:
        bank = os.path.join(tmp, "skill_bank.jsonl")
        s1 = extract_skill(
            task_file="0020_mine_block_melon.json",
            summary={"accepted": 1, "holdout_success_rate": 1.0},
            diagnosis="fails to maintain orientation towards the melon, looking away",
            proposal="re-orient and center the melon then mine",
            recovery=raw,
        )
        check("extract_skill", bool(s1) and s1["skill_id"] == "reacquire_and_center")
        u1 = upsert_skill(bank, s1)
        check("promote_on_holdout", bool(u1.get("promoted")) and u1.get("n_tasks") == 1)

        s2 = extract_skill(
            task_file="0037_mine_block_sea_pickle.json",
            summary={"accepted": 1, "holdout_success_rate": 0.0},
            diagnosis="looking away from the sea pickle",
            proposal="center the target then attack",
            recovery="center it",
            tags=["target_loss"],
        )
        u2 = upsert_skill(bank, s2)
        check("promote_second_task", u2.get("n_tasks") == 2 and bool(u2.get("promoted")))

        mem = os.path.join(tmp, "success_memory.jsonl")
        with open(mem, "w") as f:
            f.write(json.dumps({
                "task": "0020_mine_block_melon.json",
                "accepted": 1,
                "holdout_success_rate": 1.0,
                "lesson": {
                    "diagnosis": "Agent looks away from the melon block",
                    "proposal": "Modify the recovery_prompt to center the melon",
                    "recovery_prompt_snip": raw,
                },
            }) + "\n")
        n = hydrate_skill_bank(tmp)
        check("hydrate_runs", n >= 0)

        fmt = format_success_memory_for_prompt([{
            "task": "0020_mine_block_melon.json",
            "holdout_success_rate": 1.0,
            "skill": u1,
            "lesson": {"diagnosis": "melon looking away", "proposal": raw},
        }])
        check("prompt_no_melon_script", "mouseMove" not in fmt)
        check("prompt_has_skill", "reacquire_and_center" in fmt)

        h = HarnessSpec()
        h.runtime.task_config = "/tmp/0037_mine_block_sea_pickle.json"
        overlay_promoted_skills(h, tmp)
        bound = bind_skill_recovery(h, tags=["target_loss"])
        check("bind_fills_target", "sea pickle" in bound and "melon" not in bound.lower())

        incoming = HarnessSpec()
        incoming.recovery_prompt = raw
        incoming.runtime.instruction = "Mine a melon block."
        incoming.meta = {
            "merged_from_task": "0020_mine_block_melon.json",
            "last_diagnosis": "looking away from the melon",
            "last_proposal": "Modify the recovery_prompt to center the melon",
            "holdout_success_rate": 1.0,
            "holdout_win": True,
        }
        base = HarnessSpec()
        base.recovery_prompt = (
            "Previous action failed. Return to the last successful checkpoint "
            "shown in memory, then retry the current subgoal with a different action."
        )
        merged, report = merge_harness_prior(base, incoming)
        check(
            "merge_keeps_generic_recovery",
            "melon" not in (merged.recovery_prompt or "").lower()
            and "mouseMove" not in (merged.recovery_prompt or ""),
        )
        skills = (merged.meta or {}).get("prior_skills") or []
        check(
            "merge_stores_skill",
            any(s.get("skill_id") == "reacquire_and_center" for s in skills),
            str(report.get("applied")),
        )
        tips = " ".join((merged.meta or {}).get("prior_tips") or [])
        check("merge_tips_not_task_locked", "melon" not in tips.lower())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failed:
        print("FAILED", failed, flush=True)
        return 1
    print("ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
