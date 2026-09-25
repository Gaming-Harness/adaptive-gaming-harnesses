"""HarnessWAM-inspired event policy for Auto-Harness (parallel experiment).

Search object is an *event-driven* policy, not raw threshold knobs:

  event ∈ {warmup, stall, fail} → decision ∈ {continue, observe, replan, recover}

A deterministic compiler projects open edits into a legal primitive set.
``to_harness_spec()`` maps into the existing HarnessSpec so MineStudio eval
can reuse HarnessRuntime when needed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from curriculum.harness_schema import HarnessSpec, default_seed_harness


DECISIONS = ("continue", "observe", "replan", "recover", "terminate")
EVENTS = ("warmup", "stall", "fail", "milestone")

ALLOWED_ACTIONS = (
    "forward",
    "back",
    "left",
    "right",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
    "attack",
    "forward_attack",
    "jump",
    "noop",
)

# Paths the proposer may touch (compiler rejects everything else).
EDITABLE_PATHS = {
    ("triggers", "stall_steps"),
    ("triggers", "on_fail"),
    ("triggers", "on_stall"),
    ("triggers", "warmup_observes"),
    ("triggers", "forbid_periodic"),
    ("observe", "enabled"),
    ("observe", "budget"),
    ("observe", "action_pool"),
    ("observe", "prefer_uncertain"),
    ("replan", "enabled"),
    ("replan", "max_replans"),
    ("replan", "hint"),
    ("recover", "enabled"),
    ("recover", "max_retries"),
    ("recover", "force_actions"),
    ("recover", "use_checkpoint"),
    ("belief", "enabled"),
    ("belief", "keep_across_recover"),
    ("belief", "inject_into_prompt"),
    ("belief", "max_notes"),
    ("routing", "on_stall"),
    ("routing", "on_fail"),
    ("routing", "on_warmup"),
    ("prompts", "task"),
    ("prompts", "replan"),
    ("prompts", "recover"),
}


@dataclass
class EventTriggers:
    stall_steps: int = 8
    on_fail: bool = True
    on_stall: bool = True
    warmup_observes: int = 0
    # Hard rule from HarnessWAM dual-timescale: no periodic poke on healthy rollouts.
    forbid_periodic: bool = True


@dataclass
class ObserveSpec:
    """Observe ≈ short open-loop probe (evidence acquisition)."""

    enabled: bool = True
    budget: int = 6
    prefer_uncertain: bool = True
    action_pool: List[str] = field(
        default_factory=lambda: [
            "forward",
            "attack",
            "forward_attack",
            "turn_left",
            "turn_right",
            "look_up",
        ]
    )


@dataclass
class ReplanSpec:
    """Replan ≈ revise unexecuted strategy via prompt suffix (history-invariant)."""

    enabled: bool = True
    max_replans: int = 2
    hint: str = (
        "Progress stalled. Change viewpoint, then try a different approach "
        "toward the goal."
    )


@dataclass
class RecoverEventSpec:
    """Recover ≈ restore embodiment soft-state; keep belief notes."""

    enabled: bool = True
    max_retries: int = 2
    use_checkpoint: bool = True
    force_actions: List[str] = field(
        default_factory=lambda: ["back", "turn_left", "noop"]
    )


@dataclass
class BeliefSpec:
    """Lightweight scene/task notes (not a full task graph)."""

    enabled: bool = True
    keep_across_recover: bool = True
    inject_into_prompt: bool = True
    max_notes: int = 8


@dataclass
class RoutingSpec:
    """Map event → preferred decision (compiler validates enum)."""

    on_stall: str = "observe"
    on_fail: str = "recover"
    on_warmup: str = "observe"


@dataclass
class PromptSpec:
    task: str = (
        "You are a Minecraft agent. Break the goal into subgoals, "
        "prefer safe forward progress, and avoid repeating failed actions."
    )
    replan: str = (
        "Previous plan stalled. Keep acquired knowledge; revise only the "
        "remaining approach."
    )
    recover: str = (
        "Local execution failed. Soft-recover embodiment, then retry with a "
        "different local action while keeping scene notes."
    )


@dataclass
class EventPolicy:
    """Searchable event-driven harness (HarnessWAM-lite)."""

    name: str = "event_seed"
    version: int = 0
    triggers: EventTriggers = field(default_factory=EventTriggers)
    observe: ObserveSpec = field(default_factory=ObserveSpec)
    replan: ReplanSpec = field(default_factory=ReplanSpec)
    recover: RecoverEventSpec = field(default_factory=RecoverEventSpec)
    belief: BeliefSpec = field(default_factory=BeliefSpec)
    routing: RoutingSpec = field(default_factory=RoutingSpec)
    prompts: PromptSpec = field(default_factory=PromptSpec)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = self.to_dict()
        if path.endswith((".yaml", ".yml")) and yaml is not None:
            with open(path, "w") as f:
                yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
        else:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EventPolicy":
        d = copy.deepcopy(d or {})

        def _sub(dc, key):
            raw = dict(d.get(key) or {})
            return dc(**{k: v for k, v in raw.items() if k in dc.__dataclass_fields__})

        return cls(
            name=str(d.get("name", "event_seed")),
            version=int(d.get("version", 0)),
            triggers=_sub(EventTriggers, "triggers"),
            observe=_sub(ObserveSpec, "observe"),
            replan=_sub(ReplanSpec, "replan"),
            recover=_sub(RecoverEventSpec, "recover"),
            belief=_sub(BeliefSpec, "belief"),
            routing=_sub(RoutingSpec, "routing"),
            prompts=_sub(PromptSpec, "prompts"),
            meta=dict(d.get("meta") or {}),
        )

    @classmethod
    def load(cls, path: str) -> "EventPolicy":
        with open(path) as f:
            if path.endswith((".yaml", ".yml")) and yaml is not None:
                d = yaml.safe_load(f) or {}
            else:
                d = json.load(f)
        return cls.from_dict(d)

    def clone(self, *, bump_version: bool = True) -> "EventPolicy":
        p = EventPolicy.from_dict(self.to_dict())
        if bump_version:
            p.version = int(self.version) + 1
        return p

    def resolve_decision(self, event: str) -> str:
        r = self.routing
        if event == "stall":
            d = str(r.on_stall)
        elif event == "fail":
            d = str(r.on_fail)
        elif event == "warmup":
            d = str(r.on_warmup)
        else:
            d = "continue"
        if d not in DECISIONS:
            return "continue"
        # Capability gates
        if d == "observe" and not self.observe.enabled:
            return "continue"
        if d == "replan" and not self.replan.enabled:
            return "observe" if self.observe.enabled else "continue"
        if d == "recover" and not self.recover.enabled:
            return "replan" if self.replan.enabled else "continue"
        return d

    def to_harness_spec(self) -> HarnessSpec:
        """Project into legacy HarnessSpec (fail/stall-only; no periodic)."""
        h = default_seed_harness()
        h.name = self.name
        h.version = self.version
        h.task_prompt = self.prompts.task
        h.recovery_prompt = self.prompts.recover
        h.decompose_prompt = self.prompts.replan

        h.probe.enabled = bool(self.observe.enabled)
        h.probe.budget_per_episode = int(self.observe.budget)
        h.probe.action_pool = list(self.observe.action_pool)
        h.probe.prefer_uncertain = bool(self.observe.prefer_uncertain)
        h.probe.probe_periodic = False  # always off under event policy
        h.probe.probe_on_stall = bool(self.triggers.on_stall) and self.resolve_decision("stall") == "observe"
        h.probe.probe_on_fail = bool(self.triggers.on_fail) and self.resolve_decision("fail") == "observe"
        h.probe.stall_steps = int(self.triggers.stall_steps)
        h.probe.warmup_probes = int(self.triggers.warmup_observes)
        h.probe.every_n_steps = 9999

        h.recover.enabled = bool(self.recover.enabled)
        h.recover.max_retries = int(self.recover.max_retries)
        h.recover.use_memory_checkpoint = bool(self.recover.use_checkpoint)
        h.recover.replan_with_prompt = bool(self.replan.enabled)
        h.recover.recover_on_stall = bool(self.triggers.on_stall) and self.resolve_decision("stall") in (
            "recover",
            "replan",
        )
        h.recover.stall_steps = int(self.triggers.stall_steps)

        h.memory.enabled = bool(self.belief.enabled)
        h.memory.inject_into_prompt = bool(self.belief.inject_into_prompt)
        h.memory.write_failure = True
        h.memory.write_probe = True
        h.verify.enabled = False

        h.meta = {
            **dict(self.meta or {}),
            "event_policy": True,
            "event_fingerprint": self.fingerprint(),
            "routing": asdict(self.routing),
        }
        return h


def default_event_seed() -> EventPolicy:
    """Fail/stall-only seed: observe on stall, recover on fail."""
    return EventPolicy(name="event_seed", version=0)


def strong_event_seed() -> EventPolicy:
    """Aligned with Auto queue cold-start: fail/stall-only, no periodic."""
    p = default_event_seed()
    p.name = "event_strong"
    p.triggers.stall_steps = 8
    p.triggers.on_fail = True
    p.triggers.on_stall = True
    p.triggers.warmup_observes = 0
    p.triggers.forbid_periodic = True
    p.observe.enabled = True
    p.observe.budget = 8
    p.observe.action_pool = [
        "forward",
        "attack",
        "forward_attack",
        "turn_left",
        "look_up",
        "look_down",
    ]
    p.routing.on_stall = "observe"
    p.routing.on_fail = "recover"
    p.routing.on_warmup = "observe"
    p.replan.enabled = True
    p.replan.max_replans = 2
    p.recover.enabled = True
    p.recover.max_retries = 2
    p.recover.use_checkpoint = True
    return p


# --------------------------------------------------------------------------- #
# Compiler: open edits → legal EventPolicy
# --------------------------------------------------------------------------- #

class CompileError(ValueError):
    pass


def _set_path(obj: Any, path: Sequence[str], value: Any) -> None:
    cur = obj
    for k in path[:-1]:
        cur = getattr(cur, k)
    setattr(cur, path[-1], value)


def _get_path(obj: Any, path: Sequence[str]) -> Any:
    cur = obj
    for k in path:
        cur = getattr(cur, k)
    return cur


def sanitize_action_pool(pool: Any) -> List[str]:
    out: List[str] = []
    for a in list(pool or []):
        s = str(a).strip().lower()
        if s in ALLOWED_ACTIONS and s not in out:
            out.append(s)
    return out or ["forward", "turn_left", "noop"]


def compile_edits(
    base: EventPolicy,
    edits: Sequence[Dict[str, Any]],
) -> Tuple[EventPolicy, Dict[str, Any]]:
    """Deterministic executable-space projection for event-policy patches.

    Returns (policy, report). Raises CompileError on illegal proposals.
    """
    pol = base.clone(bump_version=True)
    applied: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for raw in edits or []:
        if not isinstance(raw, dict):
            rejected.append({"edit": raw, "reason": "not_dict"})
            continue
        path = raw.get("path")
        if isinstance(path, str):
            path = [p for p in path.replace("/", ".").split(".") if p]
        if not isinstance(path, (list, tuple)) or not path:
            rejected.append({"edit": raw, "reason": "bad_path"})
            continue
        path_t = tuple(str(x) for x in path)
        if path_t not in EDITABLE_PATHS:
            rejected.append({"edit": raw, "reason": f"path_not_editable:{path_t}"})
            continue
        value = raw.get("value")
        # Type / enum gates
        if path_t[0] == "routing" and str(value) not in DECISIONS:
            rejected.append({"edit": raw, "reason": f"bad_decision:{value}"})
            continue
        if path_t == ("observe", "action_pool") or path_t == ("recover", "force_actions"):
            value = sanitize_action_pool(value)
        if path_t == ("triggers", "stall_steps"):
            value = max(2, min(40, int(value)))
        if path_t == ("observe", "budget"):
            value = max(0, min(32, int(value)))
        if path_t == ("replan", "max_replans") or path_t == ("recover", "max_retries"):
            value = max(0, min(8, int(value)))
        if path_t == ("triggers", "warmup_observes"):
            value = max(0, min(8, int(value)))
        if path_t == ("triggers", "forbid_periodic") and value is False:
            # Allow only if explicitly searching periodic — still flag
            rejected.append({"edit": raw, "reason": "periodic_forbidden_by_default"})
            continue
        try:
            _set_path(pol, path_t, value)
            applied.append({"path": list(path_t), "value": value})
        except Exception as e:
            rejected.append({"edit": raw, "reason": f"set_failed:{e}"})

    # Hard invariants after projection
    pol.triggers.forbid_periodic = True
    pol.observe.action_pool = sanitize_action_pool(pol.observe.action_pool)
    pol.recover.force_actions = sanitize_action_pool(pol.recover.force_actions)
    for attr in ("on_stall", "on_fail", "on_warmup"):
        d = getattr(pol.routing, attr)
        if d not in DECISIONS:
            setattr(pol.routing, attr, "continue")

    if not applied:
        raise CompileError(f"no_legal_edits rejected={rejected}")

    report = {
        "ok": True,
        "applied": applied,
        "rejected": rejected,
        "fingerprint": pol.fingerprint(),
    }
    return pol, report


MUTATION_BANK: List[Dict[str, Any]] = [
    {"edits": [{"path": ["triggers", "stall_steps"], "value": 5}], "claim": "earlier_stall"},
    {"edits": [{"path": ["triggers", "stall_steps"], "value": 10}], "claim": "later_stall"},
    {"edits": [{"path": ["routing", "on_stall"], "value": "replan"}], "claim": "stall_replan"},
    {"edits": [{"path": ["routing", "on_stall"], "value": "recover"}], "claim": "stall_recover"},
    {"edits": [{"path": ["routing", "on_fail"], "value": "replan"}], "claim": "fail_replan"},
    {"edits": [{"path": ["routing", "on_fail"], "value": "observe"}], "claim": "fail_observe"},
    {"edits": [{"path": ["observe", "budget"], "value": 10}], "claim": "more_observe"},
    {
        "edits": [
            {"path": ["replan", "hint"], "value": "Turn to find the target block, then attack."},
            {"path": ["routing", "on_stall"], "value": "replan"},
        ],
        "claim": "replan_hint_mine",
    },
    {
        "edits": [
            {"path": ["recover", "force_actions"], "value": ["back", "turn_right", "look_up"]},
            {"path": ["recover", "max_retries"], "value": 3},
        ],
        "claim": "stronger_recover",
    },
    {
        "edits": [
            {"path": ["observe", "action_pool"], "value": ["turn_left", "turn_right", "look_up", "attack"]},
            {"path": ["routing", "on_stall"], "value": "observe"},
        ],
        "claim": "observe_view_then_attack",
    },
]


def apply_bank_mutation(base: EventPolicy, mut: Dict[str, Any]) -> Tuple[EventPolicy, Dict[str, Any]]:
    return compile_edits(base, list(mut.get("edits") or []))
