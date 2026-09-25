"""Sandbox-grounded experience + intervention for WM-only self-evolution.

Backends
--------
1. minestudio  — curriculum.sandbox_bridge.MineStudioSandbox (online interact)
2. llm_api      — minecraftbench MineRLSandboxEnv with seed reset (true counterfactuals)
3. stub        — no network; scripted interventions on offline .pt (unit tests)

WM-only: actions are scripted / random presets (no VLA).
"""
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from curriculum.action_codec import mg2_to_sandbox
from curriculum.replay_buffer import Transition
from curriculum.wm_reward_bridge import cheap_latent_from_frames

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# --------------------------------------------------------------------------- #
# Scripted action library (pretrained_wm space)
# --------------------------------------------------------------------------- #

def preset_mg2_action(name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return single-tick (kb[4], mouse[2]) in pretrained_wm space."""
    from curriculum.action_codec import CAM_VALUE

    kb = torch.zeros(4)
    ms = torch.zeros(2)
    if name == "forward":
        kb[0] = 1.0
    elif name == "back":
        kb[1] = 1.0
    elif name == "left":
        kb[2] = 1.0
    elif name == "right":
        kb[3] = 1.0
    elif name == "forward_left":
        kb[0] = 1.0
        kb[2] = 1.0
    elif name == "forward_right":
        kb[0] = 1.0
        kb[3] = 1.0
    elif name == "turn_left":
        ms[1] = -0.5 * CAM_VALUE
    elif name == "turn_right":
        ms[1] = 0.5 * CAM_VALUE
    elif name == "look_up":
        ms[0] = -0.4 * CAM_VALUE
    elif name == "look_down":
        ms[0] = 0.4 * CAM_VALUE
    elif name == "noop":
        pass
    elif name == "attack":
        pass  # attack bit set by MineStudioHarnessEnv / callers
    elif name == "forward_attack":
        kb[0] = 1.0
    else:
        raise ValueError(f"unknown preset action: {name}")
    return kb, ms


ACTION_NAMES = [
    "forward", "back", "left", "right",
    "forward_left", "forward_right",
    "turn_left", "turn_right", "look_up", "look_down", "noop",
    "attack", "forward_attack",
]


def expand_action_window(
    kb: torch.Tensor,
    ms: torch.Tensor,
    nfpb: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single tick → pretrained_wm action window covering one latent block transition."""
    T_a = max(1, 4 * (nfpb - 1) + 1)
    if kb.ndim == 1:
        kb_w = kb.unsqueeze(0).expand(T_a, -1).contiguous()
        ms_w = ms.unsqueeze(0).expand(T_a, -1).contiguous()
    else:
        kb_w, ms_w = kb[:T_a], ms[:T_a]
    return kb_w, ms_w


def sample_scripted_action(
    prefer: Optional[str] = None,
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    if prefer and prefer in ACTION_NAMES:
        name = prefer
    else:
        name = random.choice(ACTION_NAMES)
    kb, ms = preset_mg2_action(name)
    return name, kb, ms


def _plan_turn(
    action_plan: Optional[Sequence[Dict[str, Any]]],
    turn: int,
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """Return (action_a, intervene_b, meta) for this turn from scheduler plan."""
    if not action_plan:
        return None, None, {}
    if turn < len(action_plan):
        item = action_plan[turn]
    else:
        # wrap / reuse high-priority actions rather than falling back to random
        item = action_plan[turn % len(action_plan)]
    return (
        item.get("action"),
        item.get("intervene_b"),
        {k: v for k, v in item.items() if k not in ("action", "intervene_b")},
    )


# --------------------------------------------------------------------------- #
# Latent encoder
# --------------------------------------------------------------------------- #

class LatentEncoder:
    """Encode POV frames → [16,f,H,W]. cheap by default; optional Wanx VAE."""

    def __init__(
        self,
        mode: str = "cheap",
        pretrained_model_path: str = "pretrained_wm",
        device: str = "cpu",
        dtype: str = "float16",
        nfpb: int = 3,
    ):
        self.mode = mode
        self.nfpb = int(nfpb)
        self._vae = None
        if mode == "vae":
            from curriculum.coevolve_vla import FrameVAEEncoder
            self._vae = FrameVAEEncoder(pretrained_model_path, device=device, dtype=dtype)

    def encode_block(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        frames = list(frames)
        if self._vae is not None:
            arr = np.stack(frames, 0)
            return self._vae.encode_frames(arr, F=self.nfpb)["latent"]
        return cheap_latent_from_frames(frames, nfpb=self.nfpb)


# --------------------------------------------------------------------------- #
# Collectors
# --------------------------------------------------------------------------- #

@dataclass
class InterventionPair:
    """Same prefix state, two actions, two sandbox outcomes."""
    latent_t: torch.Tensor
    kb_a: torch.Tensor
    ms_a: torch.Tensor
    latent_a: torch.Tensor
    kb_b: torch.Tensor
    ms_b: torch.Tensor
    latent_b: torch.Tensor
    name_a: str = ""
    name_b: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class StubSandboxCollector:
    """No network: treat offline transitions as Env; synthesize interventions."""

    def __init__(self, encoder: LatentEncoder, nfpb: int = 3):
        self.encoder = encoder
        self.nfpb = nfpb

    def collect_episode(
        self,
        buf_expert: List[Transition],
        max_turns: int = 8,
        action_plan: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not buf_expert:
            return {"n_trans": 0, "transitions": [], "interventions": []}
        transitions: List[Transition] = []
        interventions: List[InterventionPair] = []
        for i in range(min(max_turns, len(buf_expert))):
            t = random.choice(buf_expert)
            prefer_a, prefer_b, plan_meta = _plan_turn(action_plan, i)
            name, kb, ms = sample_scripted_action(prefer=prefer_a)
            kb_w, ms_w = expand_action_window(kb, ms, self.nfpb)
            transitions.append(Transition(
                latent_t=t.latent_t,
                keyboard=kb_w,
                mouse=ms_w,
                latent_tp1=t.latent_tp1,
                reward=0.0,
                source="policy",
                meta={
                    "sandbox": "stub",
                    "action": name,
                    "turn": i,
                    "planned": bool(prefer_a),
                    **plan_meta,
                },
            ))
            name_b, kb_b, ms_b = sample_scripted_action(prefer=prefer_b)
            while name_b == name:
                name_b, kb_b, ms_b = sample_scripted_action()
            kb_bw, ms_bw = expand_action_window(kb_b, ms_b, self.nfpb)
            alt = t.latent_tp1 + 0.05 * torch.randn_like(t.latent_tp1)
            interventions.append(InterventionPair(
                latent_t=t.latent_t,
                kb_a=kb_w, ms_a=ms_w, latent_a=t.latent_tp1,
                kb_b=kb_bw, ms_b=ms_bw, latent_b=alt,
                name_a=name, name_b=name_b,
                meta={"sandbox": "stub", "planned": bool(prefer_a or prefer_b)},
            ))
        return {
            "n_trans": len(transitions),
            "transitions": transitions,
            "interventions": interventions,
            "ep_reward": 0.0,
            "planned_turns": float(sum(1 for i in range(min(max_turns, len(buf_expert))) if _plan_turn(action_plan, i)[0])),
        }

    def close(self):
        pass


class MineStudioCollector:
    """Online MineStudio interaction with scripted actions + sequential intervene."""

    def __init__(
        self,
        encoder: LatentEncoder,
        nfpb: int = 3,
        img_save_dir: str = "",
        task_list: Optional[List[str]] = None,
        ticks_per_action: int = 4,
    ):
        from curriculum.sandbox_bridge import MineStudioSandbox, SandboxConfig

        self.encoder = encoder
        self.nfpb = nfpb
        self.ticks_per_action = int(ticks_per_action)
        self.task_list = task_list or []
        self._ep = 0
        sc = SandboxConfig.from_env()
        self.sandbox = MineStudioSandbox(
            sc, img_save_dir=img_save_dir or os.path.join(_ROOT, "curriculum", "outputs", "sandbox_images")
        )

    def _encode(self, frames: List[np.ndarray]) -> torch.Tensor:
        return self.encoder.encode_block(frames)

    def _step_action(self, kb: torch.Tensor, ms: torch.Tensor):
        from curriculum.sandbox_bridge import extract_pil, pov_to_uint8

        act = mg2_to_sandbox(kb, ms)
        frames = []
        last_obs = None
        for _ in range(self.ticks_per_action):
            last_obs = self.sandbox.step(act)
            img = extract_pil(last_obs)
            if img is not None:
                frames.append(pov_to_uint8(img))
        return last_obs, frames

    def collect_episode(
        self,
        max_turns: int = 12,
        action_plan: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        from curriculum.sandbox_bridge import extract_pil, pov_to_uint8

        task = None
        if self.task_list:
            task = self.task_list[self._ep % len(self.task_list)]
        self._ep += 1
        obs = self.sandbox.reset(task_config_file_path=task)
        img = extract_pil(obs)
        if img is None:
            raise RuntimeError("MineStudio reset returned no image")

        frames_buf: List[np.ndarray] = [pov_to_uint8(img)]
        transitions: List[Transition] = []
        interventions: List[InterventionPair] = []
        ep_reward = 0.0
        planned_turns = 0

        for turn in range(max_turns):
            while len(frames_buf) < self.nfpb:
                obs_pad, frs = self._step_action(*preset_mg2_action("noop"))
                frames_buf.extend(frs or frames_buf[-1:])

            z_t = self._encode(frames_buf[-self.nfpb :])
            prefer_a, prefer_b, plan_meta = _plan_turn(action_plan, turn)
            if prefer_a:
                planned_turns += 1
            name_a, kb_a, ms_a = sample_scripted_action(prefer=prefer_a)
            kb_aw, ms_aw = expand_action_window(kb_a, ms_a, self.nfpb)

            obs_a, frs_a = self._step_action(kb_a, ms_a)
            if not frs_a:
                break
            frames_buf.extend(frs_a)
            z_a = self._encode(frames_buf[-self.nfpb :])
            r = 0.0
            if isinstance(obs_a, dict):
                try:
                    r = float(obs_a.get("reward", 0.0) or 0.0)
                except (TypeError, ValueError):
                    r = 0.0
            ep_reward = r

            transitions.append(Transition(
                latent_t=z_t,
                keyboard=kb_aw,
                mouse=ms_aw,
                latent_tp1=z_a,
                reward=r,
                source="policy",
                meta={
                    "sandbox": "minestudio",
                    "action": name_a,
                    "turn": turn,
                    "planned": bool(prefer_a),
                    **plan_meta,
                },
            ))

            z_t2 = z_a
            name_b, kb_b, ms_b = sample_scripted_action(prefer=prefer_b)
            while name_b == name_a:
                name_b, kb_b, ms_b = sample_scripted_action()
            kb_bw, ms_bw = expand_action_window(kb_b, ms_b, self.nfpb)
            obs_b, frs_b = self._step_action(kb_b, ms_b)
            if frs_b:
                frames_buf.extend(frs_b)
                z_b = self._encode(frames_buf[-self.nfpb :])
                interventions.append(InterventionPair(
                    latent_t=z_t2,
                    kb_a=kb_aw, ms_a=ms_aw, latent_a=z_a,
                    kb_b=kb_bw, ms_b=ms_bw, latent_b=z_b,
                    name_a=name_a, name_b=name_b,
                    meta={
                        "sandbox": "minestudio",
                        "mode": "sequential",
                        "planned": bool(prefer_a or prefer_b),
                    },
                ))
                transitions.append(Transition(
                    latent_t=z_t2,
                    keyboard=kb_bw,
                    mouse=ms_bw,
                    latent_tp1=z_b,
                    reward=0.0,
                    source="policy",
                    meta={
                        "sandbox": "minestudio",
                        "action": name_b,
                        "turn": turn,
                        "branch": "B",
                        "planned": bool(prefer_b),
                    },
                ))

            if isinstance(obs_a, dict) and obs_a.get("terminated"):
                break

        return {
            "n_trans": len(transitions),
            "transitions": transitions,
            "interventions": interventions,
            "ep_reward": ep_reward,
            "planned_turns": float(planned_turns),
        }

    def close(self):
        self.sandbox.close()


class LLMAPISandboxCollector:
    """Seed-resettable MineRL sandbox → true counterfactual interventions.

    Uses train.data.sandbox_collect.make_env (LLM_SANDBOX_* credentials).
    """

    def __init__(
        self,
        encoder: LatentEncoder,
        nfpb: int = 3,
        env_id: str = "MineRLBasaltFindCave-v0",
        render_size: Tuple[int, int] = (640, 360),
        seed_start: int = 0,
        ticks_per_action: int = 4,
    ):
        self.encoder = encoder
        self.nfpb = nfpb
        self.env_id = env_id
        self.render_size = list(render_size)
        self.seed = int(seed_start)
        self.ticks_per_action = int(ticks_per_action)
        self._env = None

    def _make_env(self, seed: int):
        # Reuse sandbox_collect helper
        sys.path.insert(0, _ROOT)
        from train.data.sandbox_collect import make_env
        return make_env(self.env_id, seed, self.render_size)

    def _pov(self, obs) -> np.ndarray:
        from train.data.sandbox_collect import _pov_to_uint8
        if isinstance(obs, dict):
            return _pov_to_uint8(obs.get("pov") or obs.get("rgb"))
        return np.asarray(obs, dtype=np.uint8)

    def _step_n(self, env, kb, ms, n: int) -> List[np.ndarray]:
        from train.data.sandbox_collect import mg2_action_to_sandbox
        frames = []
        for _ in range(n):
            action = mg2_action_to_sandbox(kb, ms)
            obs, _, terminated, truncated, _ = env.step(action)
            frames.append(self._pov(obs))
            if terminated or truncated:
                break
        return frames

    def _reset_frames(self, env, seed: int) -> List[np.ndarray]:
        from train.data.sandbox_collect import _reset_takes_seed
        if _reset_takes_seed(env):
            obs, _ = env.reset(seed=int(seed))
        else:
            obs, _ = env.reset()
        return [self._pov(obs)]

    def collect_episode(
        self,
        max_turns: int = 8,
        prefix_turns: int = 2,
        action_plan: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        seed = self.seed
        self.seed += 1
        env = self._make_env(seed)
        transitions: List[Transition] = []
        interventions: List[InterventionPair] = []
        planned_turns = 0
        try:
            frames = self._reset_frames(env, seed)
            # warm-up prefix
            history: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for _ in range(prefix_turns):
                name, kb, ms = sample_scripted_action()
                frs = self._step_n(env, kb, ms, self.ticks_per_action)
                if not frs:
                    break
                frames.extend(frs)
                history.append((kb, ms))

            for turn in range(max_turns):
                while len(frames) < self.nfpb:
                    frs = self._step_n(env, *preset_mg2_action("noop"), self.ticks_per_action)
                    frames.extend(frs or frames[-1:])
                z_t = self.encoder.encode_block(frames[-self.nfpb :])

                prefer_a, prefer_b, plan_meta = _plan_turn(action_plan, turn)
                if prefer_a:
                    planned_turns += 1
                name_a, kb_a, ms_a = sample_scripted_action(prefer=prefer_a)
                name_b, kb_b, ms_b = sample_scripted_action(prefer=prefer_b)
                while name_b == name_a:
                    name_b, kb_b, ms_b = sample_scripted_action()

                # --- branch A from current env ---
                frs_a = self._step_n(env, kb_a, ms_a, self.ticks_per_action)
                if not frs_a:
                    break
                frames_a = frames + frs_a
                z_a = self.encoder.encode_block(frames_a[-self.nfpb :])
                kb_aw, ms_aw = expand_action_window(kb_a, ms_a, self.nfpb)
                transitions.append(Transition(
                    latent_t=z_t, keyboard=kb_aw, mouse=ms_aw, latent_tp1=z_a,
                    reward=0.0, source="policy",
                    meta={
                        "sandbox": "llm_api",
                        "action": name_a,
                        "seed": seed,
                        "turn": turn,
                        "planned": bool(prefer_a),
                        **plan_meta,
                    },
                ))

                # --- true counterfactual B: reset seed, replay prefix+history, take B ---
                env_b = self._make_env(seed)
                try:
                    frames_b = self._reset_frames(env_b, seed)
                    for kb_h, ms_h in history:
                        frames_b.extend(self._step_n(env_b, kb_h, ms_h, self.ticks_per_action))
                    frs_b = self._step_n(env_b, kb_b, ms_b, self.ticks_per_action)
                    if frs_b:
                        frames_b.extend(frs_b)
                        z_b = self.encoder.encode_block(frames_b[-self.nfpb :])
                        kb_bw, ms_bw = expand_action_window(kb_b, ms_b, self.nfpb)
                        interventions.append(InterventionPair(
                            latent_t=z_t,
                            kb_a=kb_aw, ms_a=ms_aw, latent_a=z_a,
                            kb_b=kb_bw, ms_b=ms_bw, latent_b=z_b,
                            name_a=name_a, name_b=name_b,
                            meta={
                                "sandbox": "llm_api",
                                "seed": seed,
                                "mode": "fork",
                                "planned": bool(prefer_a or prefer_b),
                            },
                        ))
                finally:
                    try:
                        env_b.close()
                    except Exception:
                        pass

                frames = frames_a
                history.append((kb_a, ms_a))
        finally:
            try:
                env.close()
            except Exception:
                pass

        return {
            "n_trans": len(transitions),
            "transitions": transitions,
            "interventions": interventions,
            "ep_reward": 0.0,
            "seed": seed,
            "planned_turns": float(planned_turns),
        }

    def close(self):
        pass


def build_collector(cfg, logdir: str, nfpb: int) -> Any:
    """Factory from OmegaConf / dict-like cfg."""
    backend = str(cfg.get("sandbox_backend", "stub"))
    enc_mode = str(cfg.get("latent_encoder", "cheap"))
    encoder = LatentEncoder(
        mode=enc_mode,
        pretrained_model_path=str(cfg.get("pretrained_model_path", "pretrained_wm")),
        device=str(cfg.get("device", "cuda") if cfg.get("device") != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )),
        dtype=str(cfg.get("vae_dtype", "float16")),
        nfpb=nfpb,
    )
    if backend == "stub":
        return StubSandboxCollector(encoder, nfpb=nfpb)
    if backend == "minestudio":
        task_dir = cfg.get("task_dir", None)
        tasks = []
        if task_dir and os.path.isdir(str(task_dir)):
            tasks = sorted(
                os.path.join(str(task_dir), f)
                for f in os.listdir(str(task_dir))
                if f.endswith((".json", ".yaml", ".yml"))
            )
        return MineStudioCollector(
            encoder,
            nfpb=nfpb,
            img_save_dir=os.path.join(logdir, "sandbox_images"),
            task_list=tasks,
            ticks_per_action=int(cfg.get("ticks_per_action", 4)),
        )
    if backend == "llm_api":
        return LLMAPISandboxCollector(
            encoder,
            nfpb=nfpb,
            env_id=str(cfg.get("llm_api_env_id", "MineRLBasaltFindCave-v0")),
            render_size=tuple(cfg.get("render_size", [640, 360])),
            seed_start=int(cfg.get("seed_start", 0)),
            ticks_per_action=int(cfg.get("ticks_per_action", 4)),
        )
    raise ValueError(f"unknown sandbox_backend={backend}")
