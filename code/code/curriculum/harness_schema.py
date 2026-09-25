"""Editable VLA harness — file-level components for training-free search.

Harness = everything around a *frozen* VLA:
  prompt / probe schedule / memory write-read / verify / recover

No gradients. Search mutates these files; sandbox scores; bad edits revert.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


EDITABLE_KEYS = (
    "task_prompt",
    "decompose_prompt",
    "recovery_prompt",
    "probe",
    "knowledge_probe",
    "memory",
    "verify",
    "recover",
    "runtime",
)


@dataclass
class ProbeSpec:
    """When / how to actively probe for policy-facing experience."""

    enabled: bool = True
    every_n_steps: int = 4
    budget_per_episode: int = 3
    prefer_uncertain: bool = True
    action_pool: List[str] = field(
        default_factory=lambda: [
            "forward", "back", "left", "right",
            "turn_left", "turn_right", "attack", "forward_attack", "noop",
        ]
    )
    write_on_pass: bool = True
    min_confidence: float = 0.45
    # Stall-triggered probing (editable by Auto-Harness)
    probe_on_stall: bool = False
    stall_steps: int = 8
    # Force a short open-loop explore burst at episode start
    warmup_probes: int = 0
    # Fail-triggered: only intervene when episode is already failing / recovering
    probe_on_fail: bool = False
    # If False, disable every_n_steps schedule (use stall/fail only)
    probe_periodic: bool = True
    # "coverage" preserves the old behavior; "ucb" learns probe utility.
    selection_strategy: str = "coverage"  # random | coverage | ucb
    ucb_c: float = 1.2
    transfer_weight: float = 0.25
    progress_reward_weight: float = 1.0
    env_reward_weight: float = 0.25
    novelty_reward_weight: float = 0.15
    probe_cost: float = 0.05
    failure_penalty: float = 0.5
    downstream_success_reward: float = 1.0
    downstream_discount: float = 0.8
    # Optional JSON state; empty keeps learning in memory only.
    bandit_state_path: str = ""


@dataclass
class KnowledgeProbeSpec:
    """Select and validate transferable knowledge under sandbox feedback."""

    enabled: bool = False
    # K -> P (outcome-mediated): selected knowledge changes the recovery
    # instruction, and therefore the state/failure context seen by later probes.
    inject_into_recovery: bool = True
    # P -> K (outcome-mediated): sandbox utility and delayed success credit
    # update the knowledge-selection bandit and its persistent cross-task state.
    update_from_outcome: bool = True
    selection_strategy: str = "ucb"
    ucb_c: float = 1.2
    transfer_weight: float = 0.25
    progress_reward_weight: float = 1.0
    env_reward_weight: float = 0.25
    probe_cost: float = 0.02
    failure_penalty: float = 0.5
    downstream_success_reward: float = 1.0
    downstream_discount: float = 0.8
    bandit_state_path: str = ""


@dataclass
class MemorySpec:
    """What to store for *later policy use* (not WM MSE)."""

    enabled: bool = True
    merge_thresh: float = 0.85
    max_items: int = 256
    min_confidence: float = 0.45
    retrieve_topk: int = 3
    inject_into_prompt: bool = True
    write_success: bool = True
    write_failure: bool = True
    write_probe: bool = True
    decay_unused: bool = True
    revoke_on_fail_streak: int = 2


@dataclass
class VerifySpec:
    """Pre-execution gate (HELM-style SV, rule/memory/abstain — no training)."""

    enabled: bool = True
    abstain_thresh: float = 0.72
    block_on_abstain: bool = True
    prefer_memory_action: bool = True
    memory_sim_thresh: float = 0.55


@dataclass
class RecoverSpec:
    """Failure → rollback / replan using memory checkpoints."""

    enabled: bool = True
    max_retries: int = 3
    rollback_steps: int = 1
    use_memory_checkpoint: bool = True
    replan_with_prompt: bool = True
    # Soft recover when reward/progress stalls (no hard fail flag)
    recover_on_stall: bool = False
    stall_steps: int = 12


@dataclass
class RuntimeSpec:
    max_steps: int = 32
    action_chunk_len: int = 4
    checkpoint_every: int = 8
    backend: str = "stub"  # stub | minestudio
    vla_mode: str = "stub"  # stub | hf | gemini | gemini-flash
    vla_model_path: str = ""  # hf ckpt path OR gemini model id
    vla_device: str = "cuda"
    vla_dtype: str = "bfloat16"
    gemini_api_key: str = ""  # LLM API key; else LLM_API_APP_ID / LLM_API_APP_ID
    gemini_temperature: float = 0.7
    # HF Qwen-VLA sampling. Eval / paired J: prefer temperature=0, do_sample=False.
    vla_temperature: float = 0.0
    vla_do_sample: bool = False
    # OpenHA / collaborator_b multi_turn history window (past user+assistant turns).
    vla_history_window: int = 10
    # Explicit protocol keeps legacy k1_t0 and official minecraft_v1 comparable.
    vla_protocol: str = "minecraft_v1"  # minecraft_v1 | legacy_k1_t0

    instruction: str = "Explore and progress the Minecraft task."
    # MineStudio / OpenHA
    ticks_per_action: int = 4
    task_config: str = ""  # optional JSON/path passed to sandbox.reset
    img_save_dir: str = ""
    success_reward_thresh: float = 0.5
    fail_on_no_image: bool = True
    soft_rollback: bool = True  # MineStudio cannot true-reset mid-ep; soft recover only


@dataclass
class HarnessSpec:
    """Full editable harness (seed or evolved)."""

    name: str = "seed_harness"
    version: int = 0
    task_prompt: str = (
        "You are a Minecraft agent. Break the goal into subgoals, "
        "prefer safe forward progress, and avoid repeating failed actions."
    )
    decompose_prompt: str = (
        "Decompose the instruction into 3-6 ordered subgoals. "
        "Return one subgoal per line."
    )
    recovery_prompt: str = (
        "Previous action failed. Return to the last successful checkpoint "
        "shown in memory, then retry the current subgoal with a different action."
    )
    probe: ProbeSpec = field(default_factory=ProbeSpec)
    knowledge_probe: KnowledgeProbeSpec = field(default_factory=KnowledgeProbeSpec)
    memory: MemorySpec = field(default_factory=MemorySpec)
    verify: VerifySpec = field(default_factory=VerifySpec)
    recover: RecoverSpec = field(default_factory=RecoverSpec)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
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
    def from_dict(cls, d: Dict[str, Any]) -> "HarnessSpec":
        d = copy.deepcopy(d or {})
        return cls(
            name=str(d.get("name", "seed_harness")),
            version=int(d.get("version", 0)),
            task_prompt=str(d.get("task_prompt", cls().task_prompt)),
            decompose_prompt=str(d.get("decompose_prompt", cls().decompose_prompt)),
            recovery_prompt=str(d.get("recovery_prompt", cls().recovery_prompt)),
            probe=ProbeSpec(**{
                k: v for k, v in dict(d.get("probe") or {}).items()
                if k in ProbeSpec.__dataclass_fields__
            }),
            knowledge_probe=KnowledgeProbeSpec(**{
                k: v for k, v in dict(d.get("knowledge_probe") or {}).items()
                if k in KnowledgeProbeSpec.__dataclass_fields__
            }),
            memory=MemorySpec(**{
                k: v for k, v in dict(d.get("memory") or {}).items()
                if k in MemorySpec.__dataclass_fields__
            }),
            verify=VerifySpec(**{
                k: v for k, v in dict(d.get("verify") or {}).items()
                if k in VerifySpec.__dataclass_fields__
            }),
            recover=RecoverSpec(**{
                k: v for k, v in dict(d.get("recover") or {}).items()
                if k in RecoverSpec.__dataclass_fields__
            }),
            runtime=RuntimeSpec(**{
                k: v for k, v in dict(d.get("runtime") or {}).items()
                if k in RuntimeSpec.__dataclass_fields__
            }),
            meta=dict(d.get("meta") or {}),
        )

    @classmethod
    def load(cls, path: str) -> "HarnessSpec":
        with open(path) as f:
            if path.endswith((".yaml", ".yml")) and yaml is not None:
                d = yaml.safe_load(f) or {}
            else:
                d = json.load(f)
        return cls.from_dict(d)

    def clone(self, *, bump_version: bool = True) -> "HarnessSpec":
        h = HarnessSpec.from_dict(self.to_dict())
        if bump_version:
            h.version = int(self.version) + 1
        return h


def default_seed_harness() -> HarnessSpec:
    return HarnessSpec(name="seed_harness", version=0)
