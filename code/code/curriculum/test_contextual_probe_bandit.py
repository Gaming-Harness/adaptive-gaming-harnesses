from __future__ import annotations

import json
import random
from types import SimpleNamespace

from curriculum.contextual_probe_bandit import ContextualProbeBandit
from curriculum.harness_runtime import HarnessRuntime, StubMinecraftEnv, _kb_ms_from_name
from curriculum.harness_schema import HarnessSpec


def test_ucb_explores_then_prefers_useful_arm() -> None:
    bandit = ContextualProbeBandit(ucb_c=0.0, transfer_weight=0.5)
    rng = random.Random(3)
    arms = ["forward", "back", "attack"]
    seen = []
    for reward in (1.0, -1.0, 0.1):
        arm = bandit.select("mine:stall", arms, rng)
        seen.append(arm)
        bandit.update("mine:stall", arm, reward)
    assert set(seen) == set(arms)
    for arm in arms:
        bandit.update("mine:stall", arm, 2.0 if arm == "forward" else -1.0)
    assert bandit.select("mine:stall", arms, rng) == "forward"
    assert bandit.select("craft:stall", arms, rng) == "forward"


def test_bandit_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "probe_bandit.json"
    bandit = ContextualProbeBandit(ucb_c=0.7, transfer_weight=0.4)
    bandit.update("mine:fail", "look_down", 0.8)
    bandit.update("mine:fail", "look_down", 0.5, kind="delayed")
    bandit.save(str(path))
    restored = ContextualProbeBandit.load(str(path))
    assert restored.to_dict() == bandit.to_dict()
    assert json.loads(path.read_text())["version"] == 1


def test_shared_delta_merge_preserves_parallel_updates(tmp_path) -> None:
    path = tmp_path / "shared.json"
    baseline = ContextualProbeBandit().to_dict()
    first = ContextualProbeBandit.from_dict(baseline)
    second = ContextualProbeBandit.from_dict(baseline)
    first.update("mine:stall", "look_down", 1.0)
    second.update("craft:fail", "back", 0.5)
    first.merge_delta_save(str(path), baseline)
    merged = second.merge_delta_save(str(path), baseline)
    assert merged.global_arms["look_down"]["pulls"] == 1
    assert merged.global_arms["back"]["pulls"] == 1
    assert set(merged.contexts) == {"mine:stall", "craft:fail"}


def _adaptive_harness(state_path: str = "") -> HarnessSpec:
    h = HarnessSpec()
    h.runtime.backend = "stub"
    h.runtime.vla_mode = "stub"
    h.runtime.instruction = "Mine an oak log"
    h.runtime.max_steps = 4
    h.verify.enabled = False
    h.recover.enabled = False
    h.probe.enabled = True
    h.probe.selection_strategy = "ucb"
    h.probe.warmup_probes = 2
    h.probe.budget_per_episode = 2
    h.probe.probe_periodic = False
    h.probe.action_pool = ["forward"]
    h.probe.bandit_state_path = state_path
    return h


def test_runtime_records_immediate_and_delayed_probe_credit(tmp_path) -> None:
    state_path = tmp_path / "learned.json"
    runtime = HarnessRuntime(_adaptive_harness(str(state_path)), seed=5)
    result = runtime.run_episode(
        env=StubMinecraftEnv(target_progress=2.0, fail_every=0, seed=5),
        episode_seed=5,
    )
    assert result.success
    assert result.n_probe == 2
    events = result.meta["probe_events"]
    assert all(e["immediate_utility"] > 0 for e in events)
    assert all(e["downstream_credit"] > 0 for e in events)
    assert result.meta["probe_bandit"]["global_arms"]["forward"]["pulls"] == 2
    assert state_path.is_file()


def test_paired_eval_isolates_bandit_and_does_not_persist(tmp_path) -> None:
    state_path = tmp_path / "must_not_be_written.json"
    runtime = HarnessRuntime(_adaptive_harness(str(state_path)), seed=9)
    metrics = runtime.evaluate_on_seeds([9, 10], persist_memory=False)
    assert len(metrics["episodes"]) == 2
    assert runtime.probe_bandit.summary()["global_arms"] == {}
    assert not state_path.exists()


def test_schema_keeps_old_default_and_roundtrips_adaptive_fields() -> None:
    old = HarnessSpec.from_dict({"probe": {"enabled": True}})
    assert old.probe.selection_strategy == "coverage"
    h = _adaptive_harness()
    h2 = HarnessSpec.from_dict(h.to_dict())
    assert h2.probe.selection_strategy == "ucb"
    assert h2.probe.transfer_weight == h.probe.transfer_weight


def test_probe_context_prefers_structured_task_type() -> None:
    h = _adaptive_harness()
    h.runtime.instruction = "Obtain a poppy"
    h.meta = {"current_task_type": "mine_block"}
    runtime = HarnessRuntime(h, seed=3)
    assert runtime._probe_context("stall") == "mine:stall"


def test_probe_context_falls_back_to_task_filename() -> None:
    h = _adaptive_harness()
    h.runtime.instruction = "Defeat a cat"
    h.runtime.task_config = "/tmp/g022_kill_entity_cat_ws123.json"
    runtime = HarnessRuntime(h, seed=3)
    assert runtime._probe_context("fail") == "combat:fail"


def _knowledge_harness(state_path: str = "") -> HarnessSpec:
    h = HarnessSpec()
    h.runtime.backend = "stub"
    h.runtime.vla_mode = "stub"
    h.runtime.instruction = "Mine a melon block"
    h.runtime.max_steps = 5
    h.verify.enabled = False
    h.probe.enabled = False
    h.recover.enabled = True
    h.recover.max_retries = 2
    h.knowledge_probe.enabled = True
    h.knowledge_probe.selection_strategy = "ucb"
    h.knowledge_probe.bandit_state_path = state_path
    h.meta = {
        "current_task_type": "mine_block",
        "current_target": "melon",
        "knowledge_candidates": [
            {
                "skill_id": "reacquire_and_center",
                "triggers": ["orientation", "target_loss"],
                "template": "Center {target}, then retry.",
                "promoted": False,
                "n_tasks": 1,
            },
            {
                "skill_id": "backup_and_turn",
                "triggers": ["stall_heavy"],
                "template": "Back up and turn, then retry {target}.",
                "promoted": True,
                "n_tasks": 2,
            },
        ],
    }
    return h


def test_knowledge_selector_uses_failure_context() -> None:
    from curriculum.knowledge_prober import select_knowledge

    h = _knowledge_harness()
    bandit = ContextualProbeBandit(ucb_c=0.0)
    context = "mine_block:orientation+target_loss"
    bandit.update(context, "reacquire_and_center", 1.0)
    bandit.update(context, "backup_and_turn", -1.0)
    selected = select_knowledge(
        h,
        tags=["orientation", "target_loss"],
        bandit=bandit,
        rng=random.Random(1),
    )
    assert selected is not None
    assert selected["skill_id"] == "reacquire_and_center"
    assert "melon" in selected["text"]
    assert selected["context"] == "mine_block:orientation+target_loss"


def test_runtime_records_knowledge_credit_and_persists(tmp_path) -> None:
    state_path = tmp_path / "knowledge_bandit.json"
    runtime = HarnessRuntime(_knowledge_harness(str(state_path)), seed=7)
    result = runtime.run_episode(
        env=StubMinecraftEnv(target_progress=2.0, fail_every=2, seed=7),
        episode_seed=7,
    )
    assert result.success
    events = result.meta["knowledge_events"]
    assert events
    selected_id = events[0]["knowledge_id"]
    assert selected_id in {"reacquire_and_center", "backup_and_turn"}
    assert set(events[0]["candidate_ids"]) == {"reacquire_and_center", "backup_and_turn"}
    assert events[0]["downstream_credit"] > 0
    assert state_path.is_file()
    saved = json.loads(state_path.read_text())
    assert saved["global_arms"][selected_id]["pulls"] >= 1


def test_paired_eval_isolates_knowledge_bandit(tmp_path) -> None:
    state_path = tmp_path / "must_not_write_knowledge.json"
    runtime = HarnessRuntime(_knowledge_harness(str(state_path)), seed=11)
    metrics = runtime.evaluate_on_seeds([11, 12], persist_memory=False)
    assert len(metrics["episodes"]) == 2
    assert runtime.knowledge_bandit.summary()["global_arms"] == {}
    assert not state_path.exists()


def test_schema_roundtrips_knowledge_probe() -> None:
    old = HarnessSpec.from_dict({})
    assert not old.knowledge_probe.enabled
    h = _knowledge_harness("/tmp/k.json")
    restored = HarnessSpec.from_dict(h.to_dict())
    assert restored.knowledge_probe.enabled
    assert restored.knowledge_probe.bandit_state_path == "/tmp/k.json"
class _CapturingVLA:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(mode="hf")
        self.instructions = []

    def act_image(self, _image, *, instruction, **_kwargs):
        self.instructions.append(instruction)
        keyboard, mouse = _kb_ms_from_name("forward")
        return {
            "keyboard": keyboard,
            "mouse": mouse,
            "actions": None,
            "text": "forward",
        }


def test_k_to_p_switch_controls_recovery_injection() -> None:
    injected = _knowledge_harness()
    injected.runtime.vla_protocol = "legacy_k1_t0"
    injected.knowledge_probe.inject_into_recovery = True
    vla_on = _CapturingVLA()
    runtime_on = HarnessRuntime(injected, seed=3, vla=vla_on)
    runtime_on._recover_tags = ["orientation", "target_loss"]
    _, _, _, meta_on = runtime_on._propose_action(
        {"image": None}, memory_hits=[], recovering=True
    )
    assert meta_on["knowledge_injected"] is True
    assert "melon" in vla_on.instructions[-1]

    blocked = _knowledge_harness()
    blocked.runtime.vla_protocol = "legacy_k1_t0"
    blocked.knowledge_probe.inject_into_recovery = False
    vla_off = _CapturingVLA()
    runtime_off = HarnessRuntime(blocked, seed=3, vla=vla_off)
    runtime_off._recover_tags = ["orientation", "target_loss"]
    _, _, _, meta_off = runtime_off._propose_action(
        {"image": None}, memory_hits=[], recovering=True
    )
    assert meta_off["knowledge_injected"] is False
    assert vla_off.instructions[-1].startswith(blocked.recovery_prompt + "\n")
    assert not vla_off.instructions[-1].startswith("Center melon")


def test_p_to_k_switch_blocks_outcome_updates(tmp_path) -> None:
    state_path = tmp_path / "knowledge_must_stay_frozen.json"
    harness = _knowledge_harness(str(state_path))
    harness.knowledge_probe.update_from_outcome = False
    runtime = HarnessRuntime(harness, seed=7)
    result = runtime.run_episode(
        env=StubMinecraftEnv(target_progress=2.0, fail_every=2, seed=7),
        episode_seed=7,
    )
    assert result.meta["knowledge_events"]
    assert runtime.knowledge_bandit.summary()["global_arms"] == {}
    assert not state_path.exists()
