"""
WM ↔ VLA curriculum closed loop for project_root.

Stages
------
  1. Bundle & freeze a usable world-model imagination substrate
  2. Train VLA / policy with Frozen WM (real + imagined rollouts)
  3. Policy-aware WM adaptation with anchor loss (no catastrophic forgetting)
  4. Alternating closed loop (Dreamer / MuZero style)

Do NOT joint-update F_φ and π_θ from scratch — that is the failure mode this
package is designed to avoid.
"""

from .replay_buffer import MixedReplayBuffer
from .policy import PolicyBase, LatentMLPPolicy

__all__ = [
    "MixedReplayBuffer",
    "PolicyBase",
    "LatentMLPPolicy",
]
