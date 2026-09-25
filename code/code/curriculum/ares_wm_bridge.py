#!/usr/bin/env python3
"""ARES GRPO ↔ curriculum WM handoff.

Dump format (written by ARES agent when ARES_WM_DUMP_DIR is set):
  {dump_dir}/{episode_id}/meta.json
  {dump_dir}/{episode_id}/frame_XXXX.png
  {dump_dir}/{episode_id}/actions.jsonl   # one MineStudio action text per turn

Consumer:
  python -m curriculum.coevolve_vla --config ... --update-from-dir $ARES_WM_DUMP_DIR --wm-steps 20
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional


def dump_ares_episode(
    dump_root: str,
    *,
    image_paths: List[str],
    action_texts: List[str],
    rewards: List[float],
    episode_score: float,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Write one GRPO episode for WM adaptation. Returns episode dir."""
    os.makedirs(dump_root, exist_ok=True)
    ep_id = f"ep_{int(time.time() * 1000)}_{os.getpid()}"
    ep_dir = os.path.join(dump_root, ep_id)
    os.makedirs(ep_dir, exist_ok=True)

    saved_frames: List[str] = []
    for i, src in enumerate(image_paths):
        if not src or not os.path.isfile(src):
            continue
        dst = os.path.join(ep_dir, f"frame_{i:04d}.png")
        try:
            shutil.copy2(src, dst)
            saved_frames.append(dst)
        except OSError:
            continue

    with open(os.path.join(ep_dir, "actions.jsonl"), "w") as f:
        for t, text in enumerate(action_texts):
            r = float(rewards[t]) if t < len(rewards) else 0.0
            f.write(json.dumps({"turn": t, "action_text": text, "reward": r}, ensure_ascii=False) + "\n")

    payload = {
        "episode_id": ep_id,
        "n_frames": len(saved_frames),
        "n_actions": len(action_texts),
        "episode_score": float(episode_score),
        "frames": [os.path.basename(p) for p in saved_frames],
        "meta": meta or {},
    }
    with open(os.path.join(ep_dir, "meta.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return ep_dir


def list_episode_dirs(dump_root: str) -> List[str]:
    if not dump_root or not os.path.isdir(dump_root):
        return []
    eps = []
    for name in sorted(os.listdir(dump_root)):
        d = os.path.join(dump_root, name)
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "meta.json")):
            eps.append(d)
    return eps
