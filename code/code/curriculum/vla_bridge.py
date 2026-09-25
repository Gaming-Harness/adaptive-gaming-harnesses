"""Optional bridge: plug an external Qwen / ARES VLA into Stage-2.

Example (pseudo)::

    from curriculum.vla_bridge import build_external_policy
    from curriculum.frozen_wm import FrozenWorldModel, Stage1Bundle

    pi = build_external_policy(my_qwen_fn)
    wm = FrozenWorldModel(backend="full", bundle=Stage1Bundle.load("..."))

    # inside your PPO loop:
    #   real:  sandbox.step(a)
    #   imag:  wm.imagine_from_sample(sample)   # frozen teacher
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from curriculum.policy import ExternalVLAPolicy, PolicyBase


def build_external_policy(fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> PolicyBase:
    """fn(obs) must return keyboard[4], mouse[2], optional log_prob/value."""
    return ExternalVLAPolicy(fn)


def recommended_mix(step: int, max_steps: int, start_real: float = 0.8, end_real: float = 0.5) -> float:
    """Annealed real_ratio for imagination curriculum."""
    t = min(step, max_steps) / max(1, max_steps)
    return start_real + (end_real - start_real) * t
