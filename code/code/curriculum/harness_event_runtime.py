"""Event-driven harness runtime (HarnessWAM-lite) for frozen VLA.

Dual timescale:
  - fast: VLA / open-loop observe actions
  - slow: only on events (warmup / stall / fail) choose
        continue | observe | replan | recover | terminate

Emits an event log for the Auto proposer. Does not modify the original
``harness_runtime.HarnessRuntime`` episode path used by production Auto.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from curriculum.harness_event_policy import EventPolicy
from curriculum.harness_runtime import (
    EpisodeResult,
    HarnessRuntime,
    _kb_ms_from_name,
    _latent_from_obs,
)
from curriculum.harness_schema import HarnessSpec


@dataclass
class EventRecord:
    step: int
    event: str
    decision: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BeliefState:
    notes: List[str] = field(default_factory=list)
    fail_count: int = 0
    stall_events: int = 0
    last_decision: str = "continue"

    def add(self, note: str, max_notes: int = 8) -> None:
        note = (note or "").strip()
        if not note:
            return
        self.notes.append(note)
        if len(self.notes) > max_notes:
            self.notes = self.notes[-max_notes:]

    def prompt_suffix(self) -> str:
        if not self.notes:
            return ""
        return "Scene notes:\n- " + "\n- ".join(self.notes[-5:])


class EventHarnessRuntime(HarnessRuntime):
    """HarnessRuntime with event-driven deliberation overlay."""

    def __init__(
        self,
        policy: EventPolicy,
        *,
        seed: int = 0,
        harness_overrides: Optional[Dict[str, Any]] = None,
    ):
        h = policy.to_harness_spec()
        if harness_overrides:
            # Apply runtime fields from yaml (backend, vla, task_config, ...)
            rt = h.runtime
            for k, v in harness_overrides.items():
                if hasattr(rt, k):
                    setattr(rt, k, v)
        super().__init__(h, seed=seed)
        self.policy = policy
        self.event_log: List[EventRecord] = []
        self.belief = BeliefState()

    def _run_episode_loop(self, env: Any, h: HarnessSpec, seed: int) -> EpisodeResult:
        pol = self.policy
        self.rng = random.Random(int(seed) ^ 0xE7E7E7E7)
        self.event_log = []
        if not (pol.belief.enabled and pol.belief.keep_across_recover):
            self.belief = BeliefState()
        else:
            # keep notes, reset counters
            self.belief.fail_count = 0
            self.belief.stall_events = 0

        obs = env.reset()
        ep_reward = 0.0
        n_probe = n_block = n_recover = n_write = n_hit = 0
        n_replan = 0
        fails_in_row = 0
        cascade = False
        retries = 0
        replans = 0
        done = False
        step = 0
        max_steps = int(h.runtime.max_steps)
        backend = str(h.runtime.backend)
        last_prog = float(
            (obs.get("progress") if isinstance(obs, dict) else None)
            or getattr(env, "progress", 0.0)
            or 0.0
        )
        stall_count = 0
        warmup_left = int(pol.triggers.warmup_observes or 0)
        force_name: Optional[str] = None
        recovering = False
        replan_active = False

        while step < max_steps and not done:
            prog_now = float(
                (obs.get("progress") if isinstance(obs, dict) else None)
                or getattr(env, "progress", 0.0)
                or 0.0
            )
            if prog_now <= last_prog + 1e-6:
                stall_count += 1
            else:
                stall_count = 0
                last_prog = prog_now
                replan_active = False

            # --- slow loop: emit at most one event ---
            event: Optional[str] = None
            if warmup_left > 0 and pol.observe.enabled and n_probe < pol.observe.budget:
                event = "warmup"
            elif (
                pol.triggers.on_fail
                and fails_in_row > 0
                and retries < pol.recover.max_retries
            ):
                event = "fail"
            elif (
                pol.triggers.on_stall
                and stall_count >= max(1, int(pol.triggers.stall_steps))
                and n_probe + n_recover + n_replan < (
                    pol.observe.budget + pol.recover.max_retries + pol.replan.max_replans
                )
            ):
                event = "stall"

            decision = "continue"
            force_name = None
            recovering = False
            if event is not None:
                decision = pol.resolve_decision(event)
                detail = ""
                if decision == "observe" and n_probe < pol.observe.budget:
                    force_name = self._event_observe_action(step)
                    n_probe += 1
                    self.stats["probe"] += 1
                    detail = f"observe:{force_name}"
                    if event == "warmup":
                        warmup_left = max(0, warmup_left - 1)
                    if event == "stall":
                        stall_count = 0
                        self.belief.stall_events += 1
                    self.belief.add(f"observe@{step}:{force_name}", pol.belief.max_notes)
                elif decision == "replan" and replans < pol.replan.max_replans:
                    replans += 1
                    n_replan += 1
                    replan_active = True
                    detail = f"replan#{replans}"
                    if event == "stall":
                        stall_count = 0
                        self.belief.stall_events += 1
                    self.belief.add(f"replan@{step}:{pol.replan.hint[:60]}", pol.belief.max_notes)
                elif decision == "recover" and retries < pol.recover.max_retries:
                    n_recover += 1
                    retries += 1
                    self.stats["recover"] += 1
                    recovering = True
                    if pol.recover.use_checkpoint and hasattr(env, "rollback"):
                        obs = env.rollback()
                    pool = list(pol.recover.force_actions) or ["back"]
                    force_name = self.rng.choice(pool)
                    detail = f"recover:{force_name}"
                    stall_count = 0
                    self.belief.fail_count += 1
                    if not pol.belief.keep_across_recover:
                        self.belief.notes.clear()
                    self.belief.add(f"recover@{step}:{force_name}", pol.belief.max_notes)
                elif decision == "terminate":
                    self.event_log.append(
                        EventRecord(step=step, event=event, decision=decision, detail="stop")
                    )
                    break
                else:
                    decision = "continue"
                    detail = "gated_continue"
                    if event == "warmup":
                        warmup_left = max(0, warmup_left - 1)

                self.event_log.append(
                    EventRecord(step=step, event=event, decision=decision, detail=detail)
                )
                self.belief.last_decision = decision

            z = _latent_from_obs(obs)
            kb0, ms0 = _kb_ms_from_name("forward")
            hits = self._retrieve(z, kb0, ms0)
            if hits:
                n_hit += 1

            # Inject belief + replan into temporary prompts
            old_task = h.task_prompt
            old_rec = h.recovery_prompt
            old_instr = h.runtime.instruction
            try:
                suffix = ""
                if pol.belief.enabled and pol.belief.inject_into_prompt:
                    suffix = self.belief.prompt_suffix()
                if replan_active:
                    h.task_prompt = (
                        pol.prompts.replan + "\n" + pol.replan.hint + "\n" + old_task
                    )
                    if suffix:
                        h.task_prompt = h.task_prompt + "\n" + suffix
                elif suffix:
                    h.task_prompt = old_task + "\n" + suffix
                if recovering:
                    h.recovery_prompt = pol.prompts.recover

                name, kb, ms, meta = self._propose_action(
                    obs,
                    memory_hits=hits,
                    force_name=force_name,
                    recovering=recovering,
                )
            finally:
                h.task_prompt = old_task
                h.recovery_prompt = old_rec
                h.runtime.instruction = old_instr

            allow, abstain, alt = self._verify_gate(z, kb, ms, name)
            if not allow and alt:
                n_block += 1
                self.stats["verify_block"] += 1
                name = alt
                kb, ms = _kb_ms_from_name(alt)
                meta = {**(meta or {}), "actions": None, "source": "verify_block"}

            if (
                pol.recover.enabled
                and step > 0
                and step % max(1, h.runtime.checkpoint_every) == 0
                and not getattr(env, "last_fail", False)
            ):
                env.mark_checkpoint()

            actions = (meta or {}).get("actions")
            if (
                actions
                and hasattr(env, "step_action_chunk")
                and str(h.runtime.backend).lower() in ("minestudio", "sandbox", "openha")
            ):
                obs2, reward, done, info = env.step_action_chunk(actions, primary_name=name)
            else:
                obs2, reward, done, info = env.step_action_name(name)
            z2 = _latent_from_obs(obs2)
            ep_reward += reward
            fail = bool(info.get("fail", False))
            if isinstance(obs2, dict) and (obs2.get("success") or obs2.get("terminated")):
                done = True
            if info.get("success") or info.get("terminated"):
                done = True
            if getattr(env, "_task_success", False) or getattr(env, "_done", False):
                done = True
            if done and (info.get("success") or getattr(env, "_task_success", False)):
                fail = False

            wrote = self._write_memory(
                z,
                kb,
                ms,
                z2,
                action=name,
                success=not fail,
                source="probe" if force_name and decision == "observe" else "exec",
                step=step,
                note=decision if event else ("fail" if fail else "ok"),
            )
            if wrote:
                n_write += 1
                self.stats["memory_write"] += 1

            if fail and not done:
                fails_in_row += 1
                if fails_in_row >= 3:
                    cascade = True
                self.belief.add(f"fail@{step}:action={name}", pol.belief.max_notes)
            else:
                if not fail:
                    fails_in_row = 0
                    retries = 0

            obs = obs2
            step += 1

        prog = float(getattr(env, "progress", 0.0))
        tgt = float(getattr(env, "target", 1.0))
        success = bool(getattr(env, "_task_success", False))
        if isinstance(obs, dict) and obs.get("success"):
            success = True
        if not success and tgt > 0 and prog >= tgt * 0.99:
            success = True
        self.stats["episodes"] += 1
        if success:
            self.stats["successes"] += 1
        self.memory.promote_mastered(min_verify=2, min_conf=h.memory.min_confidence)

        return EpisodeResult(
            success=success,
            steps=step,
            reward=float(ep_reward),
            n_probe=n_probe,
            n_verify_block=n_block,
            n_recover=n_recover,
            n_memory_write=n_write,
            n_memory_hit=n_hit,
            cascade_fail=cascade,
            meta={
                "progress": prog,
                "target": tgt,
                "backend": backend,
                "seed": seed,
                "event_policy": True,
                "n_replan": n_replan,
                "events": [e.to_dict() for e in self.event_log],
                "belief_notes": list(self.belief.notes),
                "diversity": "event_observe_replan_recover",
                "world_seed_source": "task_config",
            },
        )

    def _event_observe_action(self, step: int) -> str:
        pool = list(self.policy.observe.action_pool) or ["forward"]
        if self.policy.observe.prefer_uncertain:
            cov = self.memory.coverage_report(action_names=pool)
            unknown = [n for n, v in cov["per_action"].items() if v.get("unknown", 0) >= 1.0]
            if unknown:
                return self.rng.choice(unknown)
        return self.rng.choice(pool)

    def evaluate(
        self,
        n_episodes: int = 8,
        *,
        base_seed: Optional[int] = None,
        persist_memory: bool = True,
        episode_seeds: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        out = super().evaluate(
            n_episodes,
            base_seed=base_seed,
            persist_memory=persist_memory,
            episode_seeds=episode_seeds,
        )
        # Aggregate event stats from last evaluate batch via results
        ev_counts: Dict[str, int] = {}
        for r in out.get("episodes") or []:
            meta = (r or {}).get("meta") or {}
            for e in meta.get("events") or []:
                key = f"{e.get('event')}→{e.get('decision')}"
                ev_counts[key] = ev_counts.get(key, 0) + 1
        out["event_counts"] = ev_counts
        out["policy_fingerprint"] = self.policy.fingerprint()
        return out
