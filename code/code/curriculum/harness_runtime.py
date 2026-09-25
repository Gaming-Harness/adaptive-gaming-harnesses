"""Frozen-VLA harness runtime: retrieve → propose → verify → execute → memory.

Training-free: VLA weights never update. Memory grows from probe/sandbox truth
for *later policy use* (prompt injection / action bias / recover).
"""
from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from curriculum.harness_schema import HarnessSpec
from curriculum.contextual_probe_bandit import ContextualProbeBandit
from curriculum.knowledge_memory import WorldKnowledgeMemory, action_embed, pool_latent
from curriculum.sandbox_experience import ACTION_NAMES, preset_mg2_action


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    reward: float
    n_probe: int = 0
    n_verify_block: int = 0
    n_recover: int = 0
    n_memory_write: int = 0
    n_memory_hit: int = 0
    cascade_fail: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "steps": self.steps,
            "reward": self.reward,
            "n_probe": self.n_probe,
            "n_verify_block": self.n_verify_block,
            "n_recover": self.n_recover,
            "n_memory_write": self.n_memory_write,
            "n_memory_hit": self.n_memory_hit,
            "cascade_fail": self.cascade_fail,
            "meta": self.meta,
        }


class StubMinecraftEnv:
    """Lightweight long-horizon stub with injected failures.

    Goal: reach `target_progress` via mostly-forward actions.
    Failures: every `fail_every` steps a bad action (or random) zeros progress
    unless recover restores the last checkpoint.
    """

    def __init__(
        self,
        *,
        target_progress: float = 12.0,
        fail_every: int = 4,
        seed: int = 0,
        obs_size: Tuple[int, int] = (64, 64),
        fail_penalty: float = 1.5,
    ):
        self.target = float(target_progress)
        self.fail_every = int(fail_every)
        self.fail_penalty = float(fail_penalty)
        self.rng = random.Random(seed)
        self.np = np.random.RandomState(seed)
        self.obs_size = obs_size
        self.progress = 0.0
        self.t = 0
        self.checkpoint = 0.0
        self.last_fail = False
        self._done = False

    def reset(self) -> Dict[str, Any]:
        self.progress = 0.0
        self.t = 0
        self.checkpoint = 0.0
        self.last_fail = False
        self._done = False
        return self._obs()

    def _obs(self) -> Dict[str, Any]:
        # encode progress in image brightness for stub VLA / latent
        g = int(np.clip(self.progress / max(1.0, self.target), 0, 1) * 200 + 20)
        arr = np.full((self.obs_size[1], self.obs_size[0], 3), g, dtype=np.uint8)
        # noise stripe so frames aren't identical
        arr[:, :8, 0] = (self.t * 17) % 255
        img = Image.fromarray(arr)
        return {
            "image": img,
            "progress": self.progress,
            "target": self.target,
            "t": self.t,
            "last_fail": self.last_fail,
        }

    def mark_checkpoint(self) -> None:
        self.checkpoint = float(self.progress)

    def rollback(self) -> Dict[str, Any]:
        self.progress = float(self.checkpoint)
        self.last_fail = False
        return self._obs()

    def step_action_name(self, name: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.t += 1
        injected = self.fail_every > 0 and (self.t % self.fail_every == 0)
        delta = 0.0
        fail = False
        if name in ("forward", "forward_left", "forward_right"):
            delta = 1.0
        elif name == "forward_attack":
            delta = 1.2
        elif name == "attack":
            delta = 0.6
        elif name in ("left", "right"):
            delta = 0.2
        elif name in ("turn_left", "turn_right", "look_up", "look_down"):
            delta = 0.0
        elif name == "back":
            delta = -0.8
        else:
            delta = 0.0

        # Injected fault: corrupt unless we just rolled back (caller recovers)
        if injected:
            fail = True
            delta = -self.fail_penalty
            self.last_fail = True
        elif name in ("back", "noop") and self.rng.random() < 0.4:
            fail = True
            delta = -self.fail_penalty * 0.5
            self.last_fail = True
        else:
            self.last_fail = False

        self.progress = max(0.0, self.progress + delta)
        reward = float(delta)
        done = self.progress >= self.target
        self._done = done
        info = {"fail": fail, "injected": injected, "action": name, "delta": delta}
        return self._obs(), reward, done, info


class MineStudioHarnessEnv:
    """Live MineStudio sandbox adapter with the same surface as StubMinecraftEnv.

    True mid-episode state rollback is unavailable on the OpenHA gateway.
    ``rollback()`` is a *soft* recover: clear fail flag and optionally issue a
    reverse/noop tick so the harness replan path can continue.

    ``seed`` diversifies harness RNG / task_id — it does **not** override
    OpenHA ``task_config["seed"]`` (world seed). Wrong world seeds invalidate
    ``/tp`` and yield black screens.
    """

    def __init__(
        self,
        *,
        ticks_per_action: int = 4,
        task_config: str = "",
        img_save_dir: str = "",
        success_reward_thresh: float = 0.5,
        soft_rollback: bool = True,
        seed: int = 0,
    ):
        from curriculum.action_codec import mg2_to_sandbox
        from curriculum.sandbox_bridge import (
            MineStudioSandbox,
            SandboxConfig,
            extract_pil,
            pov_to_uint8,
        )

        self._mg2_to_sandbox = mg2_to_sandbox
        self._extract_pil = extract_pil
        self._pov_to_uint8 = pov_to_uint8
        self.ticks_per_action = int(ticks_per_action)
        self.task_config = task_config or None
        self.success_reward_thresh = float(success_reward_thresh)
        self.soft_rollback = bool(soft_rollback)
        self.seed = int(seed)
        self.progress = 0.0
        self.target = 1.0  # binary success proxy for scoring compatibility
        self.last_fail = False
        self._done = False
        self._terminated = False
        self._task_success = False
        self._last_reward = 0.0
        self._peak_reward = 0.0
        self._ckpt_reward = 0.0
        self._last_action = "noop"
        self._frame: Optional[np.ndarray] = None
        self._ckpt_frame: Optional[np.ndarray] = None
        self._last_obs: Optional[Dict[str, Any]] = None
        save = img_save_dir or os.path.join(
            os.path.dirname(__file__), "outputs", "harness_sandbox_images"
        )
        self._trace_enabled = os.environ.get("HARNESS_TRACE_TICKS", "").lower() in (
            "1", "true", "yes",
        )
        self._trace_index = 0
        self._trace_dir = os.path.join(save, f"tick_trace_seed_{self.seed}")
        if self._trace_enabled:
            os.makedirs(self._trace_dir, exist_ok=True)
        self.sandbox = MineStudioSandbox(
            SandboxConfig.from_env(),
            img_save_dir=save,
            task_id=(0, seed % 1000),
        )

    def _obs_from_raw(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        img = self._extract_pil(raw)
        if img is None:
            # fallback gray frame so latent path does not crash
            img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        else:
            self._frame = self._pov_to_uint8(img, size=(128, 128))
        try:
            r = float(raw.get("reward", 0.0) or 0.0)
        except (TypeError, ValueError):
            r = 0.0
        info = raw.get("info_dict") if isinstance(raw.get("info_dict"), dict) else {}
        # OpenHA MineStudioEnv uses game_state for terminated, not "terminated"/"done".
        terminated = bool(
            raw.get("terminated")
            or raw.get("done")
            or raw.get("game_state")
            or info.get("terminated")
            or info.get("done")
            or False
        )
        if r >= self.success_reward_thresh:
            self._task_success = True
        success = bool(
            self._task_success
            or raw.get("success")
            or raw.get("task_success")
            or info.get("success")
            or info.get("task_success")
            or (terminated and r >= self.success_reward_thresh)
        )
        self._last_reward = r
        self._peak_reward = max(self._peak_reward, r)
        self.progress = float(self._peak_reward if success else r)
        if success:
            self.progress = max(self.progress, self.target)
        self._terminated = bool(terminated or success)
        self._done = bool(success or terminated)
        obs = {
            "image": img,
            "progress": self.progress,
            "target": self.target,
            "t": int(raw.get("turn_count") or 0),
            "last_fail": self.last_fail,
            "reward": r,
            "terminated": self._terminated,
            "success": success,
            "raw": raw,
        }
        self._last_obs = obs
        return obs

    def reset(self) -> Dict[str, Any]:
        self.progress = 0.0
        self.last_fail = False
        self._done = False
        self._terminated = False
        self._task_success = False
        self._last_reward = 0.0
        self._peak_reward = 0.0
        self._ckpt_reward = 0.0
        self._last_action = "noop"
        self._last_obs = None
        raw = self.sandbox.reset(task_config_file_path=self.task_config)
        obs = self._obs_from_raw(raw if isinstance(raw, dict) else {})
        self.mark_checkpoint()
        return obs

    def mark_checkpoint(self) -> None:
        self._ckpt_reward = float(self._peak_reward)
        self._ckpt_frame = None if self._frame is None else self._frame.copy()

    def rollback(self) -> Dict[str, Any]:
        """Soft recover — cannot rewind MineStudio world state."""
        self.last_fail = False
        if not self.soft_rollback:
            return {
                "image": Image.fromarray(self._frame) if self._frame is not None
                else Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)),
                "progress": self.progress,
                "target": self.target,
                "t": 0,
                "last_fail": False,
                "reward": self._last_reward,
                "terminated": self._terminated,
                "success": False,
            }
        # heuristic undo: if last move was forward, try back once
        undo = "back" if self._last_action in ("forward", "forward_left", "forward_right") else "noop"
        obs2, _, _, _ = self.step_action_name(undo)
        obs2["last_fail"] = False
        self.last_fail = False
        return obs2

    def step_action_name(self, name: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        kb, ms = preset_mg2_action(name)
        act = self._mg2_to_sandbox(kb, ms)
        if "attack" in str(name):
            act["attack"] = 1
        return self._step_actions([act] * max(1, self.ticks_per_action), primary_name=name)

    def step_action_chunk(
        self,
        actions: List[Dict[str, Any]],
        *,
        primary_name: str = "chunk",
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute a VLA MineStudio action chunk (typically len=4)."""
        if not actions:
            return self.step_action_name("noop")
        return self._step_actions(list(actions), primary_name=primary_name)

    def _step_actions(
        self,
        actions: List[Dict[str, Any]],
        *,
        primary_name: str,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._done and self._last_obs is not None:
            return self._last_obs, 0.0, True, {
                "fail": False,
                "action": primary_name,
                "delta": 0.0,
                "sandbox": "minestudio",
                "terminated": True,
                "success": bool(self._task_success),
                "soft_rollback": self.soft_rollback,
                "n_ticks": 0,
            }
        prev_r = self._last_reward
        last_raw = None
        n_ran = 0
        for act in actions:
            last_raw = self.sandbox.step(act)
            n_ran += 1
            obs = self._obs_from_raw(last_raw if isinstance(last_raw, dict) else {})
            if self._trace_enabled:
                trace_i = self._trace_index
                self._trace_index += 1
                image = obs.get("image") if isinstance(obs, dict) else None
                image_path = os.path.join(self._trace_dir, f"tick_{trace_i:06d}.png")
                if isinstance(image, Image.Image):
                    image.save(image_path)
                record = {
                    "tick": trace_i,
                    "primary_name": primary_name,
                    "action": act,
                    "attack": int(act.get("attack", 0) or 0) if isinstance(act, dict) else None,
                    "reward": float(obs.get("reward", 0.0) or 0.0),
                    "success": bool(obs.get("success", False)),
                    "terminated": bool(obs.get("terminated", False)),
                    "image": os.path.basename(image_path) if os.path.isfile(image_path) else None,
                }
                with open(os.path.join(self._trace_dir, "trace.jsonl"), "a") as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            if self._done:
                break
        else:
            obs = self._obs_from_raw(last_raw if isinstance(last_raw, dict) else {})
        r = float(obs.get("reward", 0.0) or 0.0)
        delta = r - prev_r
        success = bool(self._task_success or obs.get("success"))
        terminated = bool(self._terminated or obs.get("terminated"))
        fail = False
        if terminated and not success:
            fail = True
        if delta < -0.05 and not success:
            fail = True
        self.last_fail = fail
        self._last_action = primary_name
        if success:
            reward = max(float(self._peak_reward), float(self.success_reward_thresh), float(r))
        else:
            reward = float(delta if abs(delta) > 1e-8 else (0.05 if "forward" in primary_name else 0.0))
        done = bool(success or terminated)
        if success:
            self.progress = max(self.progress, self.target)
            print(
                f"[HarnessRuntime] sandbox success name={primary_name} "
                f"reward={r} ticks={n_ran}/{len(actions)} stop episode",
                flush=True,
            )
        info = {
            "fail": fail,
            "action": primary_name,
            "delta": delta,
            "sandbox": "minestudio",
            "terminated": terminated,
            "success": success,
            "soft_rollback": self.soft_rollback,
            "n_ticks": n_ran,
        }
        return obs, reward, done, info

    def rng_fail_noop(self) -> bool:
        return False

    def close(self) -> None:
        try:
            self.sandbox.close()
        except Exception:
            pass


def _instruction_from_task(task_config: str, fallback: str) -> str:
    if not task_config or not os.path.isfile(task_config):
        return fallback
    try:
        import json
        with open(task_config) as f:
            d = json.load(f)
        for k in ("instruction", "task", "task_instruction", "text", "goal"):
            if d.get(k):
                return str(d[k])
        # OpenHA nested
        if isinstance(d.get("task"), dict) and d["task"].get("instruction"):
            return str(d["task"]["instruction"])
    except Exception:
        pass
    return fallback


def make_env(harness: HarnessSpec, *, seed: int = 0) -> Any:
    """Factory: stub (default) or live MineStudio."""
    backend = str(harness.runtime.backend or "stub").lower()
    if backend in ("stub", "local"):
        return StubMinecraftEnv(
            target_progress=12.0,
            fail_every=4,
            seed=seed,
            fail_penalty=3.0,
        )
    if backend in ("minestudio", "sandbox", "openha"):
        return MineStudioHarnessEnv(
            ticks_per_action=int(harness.runtime.ticks_per_action),
            task_config=str(harness.runtime.task_config or ""),
            img_save_dir=str(harness.runtime.img_save_dir or ""),
            success_reward_thresh=float(harness.runtime.success_reward_thresh),
            soft_rollback=bool(harness.runtime.soft_rollback),
            seed=seed,
        )
    raise ValueError(f"unknown harness backend: {backend}")


def _latent_from_obs(obs: Dict[str, Any], dim: int = 32) -> torch.Tensor:
    """Cheap deterministic latent from stub image / progress (no WM train)."""
    img = obs["image"]
    arr = np.asarray(img, dtype=np.float32) / 255.0
    flat = arr.reshape(-1)
    # fold into dim
    z = np.zeros(dim, dtype=np.float32)
    for i, v in enumerate(flat[: dim * 8]):
        z[i % dim] += v
    z = z / max(1.0, flat[: dim * 8].size / dim)
    z[0] = float(obs.get("progress", 0.0))
    z[1] = float(obs.get("target", 1.0))
    z[2] = float(obs.get("t", 0))
    # fake [C,f,H,W]-like for pool_latent: use [C,1,1,1]
    return torch.tensor(z, dtype=torch.float32).view(dim, 1, 1, 1)


def _kb_ms_from_name(name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    kb, ms = preset_mg2_action(name)
    return kb.unsqueeze(0), ms.unsqueeze(0)


def _name_from_kb_ms(kb: torch.Tensor, ms: torch.Tensor) -> str:
    k = kb.reshape(-1) if kb.ndim else kb
    m = ms.reshape(-1) if ms.ndim else ms
    if k.numel() >= 4:
        vals = [
            ("forward", float(k[0])),
            ("back", float(k[1])),
            ("left", float(k[2])),
            ("right", float(k[3])),
        ]
        best = max(vals, key=lambda x: x[1])
        if best[1] > 0.5:
            if best[0] == "forward" and float(k[2]) > 0.5:
                return "forward_left"
            if best[0] == "forward" and float(k[3]) > 0.5:
                return "forward_right"
            return best[0]
    if m.numel() >= 2:
        if float(m[1]) < -1.0:
            return "turn_left"
        if float(m[1]) > 1.0:
            return "turn_right"
        if float(m[0]) < -1.0:
            return "look_up"
        if float(m[0]) > 1.0:
            return "look_down"
    return "noop"


class HarnessRuntime:
    """Execute one or many episodes under a frozen VLA + editable harness."""

    def __init__(
        self,
        harness: HarnessSpec,
        *,
        memory: Optional[WorldKnowledgeMemory] = None,
        vla: Any = None,
        seed: int = 0,
    ):
        self.harness = harness
        self.seed = int(seed)
        self.rng = random.Random(seed)
        mem_cfg = harness.memory
        self.memory = memory or WorldKnowledgeMemory(
            merge_thresh=mem_cfg.merge_thresh,
            max_items=mem_cfg.max_items,
            min_confidence=mem_cfg.min_confidence,
        )
        self.vla = vla
        self._ensure_vla()
        p = harness.probe
        bandit_path = str(getattr(p, "bandit_state_path", "") or "")
        self.probe_bandit = ContextualProbeBandit.load(
            bandit_path,
            ucb_c=float(getattr(p, "ucb_c", 1.2)),
            transfer_weight=float(getattr(p, "transfer_weight", 0.25)),
        )
        self._initial_probe_bandit = copy.deepcopy(self.probe_bandit.to_dict())
        kp = harness.knowledge_probe
        knowledge_path = str(getattr(kp, "bandit_state_path", "") or "")
        self.knowledge_bandit = ContextualProbeBandit.load(
            knowledge_path,
            ucb_c=float(getattr(kp, "ucb_c", 1.2)),
            transfer_weight=float(getattr(kp, "transfer_weight", 0.25)),
        )
        self._initial_knowledge_bandit = copy.deepcopy(self.knowledge_bandit.to_dict())
        self._suppress_bandit_save = False
        self.stats: Dict[str, float] = {
            "episodes": 0,
            "successes": 0,
            "probe": 0,
            "verify_block": 0,
            "recover": 0,
            "memory_write": 0,
            "memory_hit": 0,
        }

    def _ensure_vla(self) -> None:
        if self.vla is not None:
            return

        mode = str(self.harness.runtime.vla_mode or "stub").lower()
        # stub backend forces stub VLA unless explicitly overridden
        if self.harness.runtime.backend == "stub" and mode not in ("hf", "gemini", "gemini-flash", "gemini_flash"):
            mode = "stub"

        current = str(self.harness.runtime.instruction or "").strip()
        generic = current in ("", "Explore and progress the Minecraft task.")
        task_instr = _instruction_from_task(
            str(self.harness.runtime.task_config or ""),
            current or "Explore and progress the Minecraft task.",
        )
        # Task JSON is the default goal. Keep an already-customized instruction
        # (Auto patch / replay) so _ensure_vla does not wipe it.
        instr = task_instr if generic else current
        self.harness.runtime.instruction = instr

        if mode in ("gemini", "gemini-flash", "gemini_flash"):
            from curriculum.gemini_vla import DEFAULT_GEMINI_MODEL, GeminiVLAConfig, GeminiVLAPolicy

            model = str(self.harness.runtime.vla_model_path or "").strip() or DEFAULT_GEMINI_MODEL
            if model.endswith((".pt", ".bin", ".safetensors")) or "/checkpoint-" in model:
                # user accidentally left hf path; fall back to default flash id
                model = DEFAULT_GEMINI_MODEL
            self.vla = GeminiVLAPolicy(
                GeminiVLAConfig(
                    model=model,
                    api_key=str(getattr(self.harness.runtime, "gemini_api_key", "") or ""),
                    mode="api",
                    instruction=instr,
                    action_chunks_len=self.harness.runtime.action_chunk_len,
                    temperature=float(getattr(self.harness.runtime, "gemini_temperature", 0.7)),
                )
            )
            print(f"[HarnessRuntime] VLA mode=gemini model={model} instr={instr[:60]!r}", flush=True)
            return

        from curriculum.qwen_vla import DEFAULT_VLA_CKPT, QwenVLAConfig, QwenVLAPolicy

        if self.harness.runtime.backend == "stub" and mode != "hf":
            mode = "stub"
        model_path = str(self.harness.runtime.vla_model_path or "").strip() or DEFAULT_VLA_CKPT
        self.vla = QwenVLAPolicy(
            QwenVLAConfig(
                model_path=model_path,
                mode=mode if mode in ("hf", "stub") else "stub",
                device=str(self.harness.runtime.vla_device or "cuda"),
                dtype=str(self.harness.runtime.vla_dtype or "bfloat16"),
                instruction=instr,
                action_chunks_len=self.harness.runtime.action_chunk_len,
                temperature=float(getattr(self.harness.runtime, "vla_temperature", 0.0)),
                do_sample=bool(getattr(self.harness.runtime, "vla_do_sample", False)),
                history_window=int(getattr(self.harness.runtime, "vla_history_window", 10) or 10),
                protocol=str(getattr(self.harness.runtime, "vla_protocol", "minecraft_v1")),
            )
        )
        print(
            f"[HarnessRuntime] VLA mode={mode} path={model_path} "
            f"temp={getattr(self.harness.runtime, 'vla_temperature', 0.0)} "
            f"do_sample={getattr(self.harness.runtime, 'vla_do_sample', False)} "
            f"history_window={getattr(self.harness.runtime, 'vla_history_window', 10)} "
            f"protocol={getattr(self.harness.runtime, 'vla_protocol', 'minecraft_v1')} "
            f"instr={instr[:60]!r}",
            flush=True,
        )

    def _format_memory_context(self, hits: List[Any]) -> str:
        if not hits:
            return ""
        lines = [self.harness.task_prompt, "Verified experience (use these):"]
        for it, sim in hits:
            act = str((it.meta or {}).get("action", "?"))
            status = it.status
            lines.append(
                f"- action={act} status={status} conf={it.confidence:.2f} "
                f"sim={sim:.2f} note={(it.meta or {}).get('note', '')}"
            )
        return "\n".join(lines)

    def _propose_action(
        self,
        obs: Dict[str, Any],
        *,
        memory_hits: List[Any],
        force_name: Optional[str] = None,
        recovering: bool = False,
        step_num: int = 0,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if force_name:
            kb, ms = _kb_ms_from_name(force_name)
            return force_name, kb, ms, {"source": "force", "name": force_name}

        # Memory-biased stub / soft prefer
        prefer = None
        if (
            self.harness.verify.prefer_memory_action
            and memory_hits
            and memory_hits[0][1] >= self.harness.verify.memory_sim_thresh
        ):
            prefer = str((memory_hits[0][0].meta or {}).get("action") or "") or None

        instr = self.harness.runtime.instruction
        protocol = str(
            getattr(self.harness.runtime, "vla_protocol", "minecraft_v1")
            or "minecraft_v1"
        ).lower()
        legacy = protocol in ("legacy", "legacy_k1_t0", "k1_t0", "single_turn")
        extras: List[str] = []
        rec_txt = ""
        knowledge_meta: Dict[str, Any] = {}
        if recovering:
            tags = list(getattr(self, "_recover_tags", None) or [])
            try:
                if bool(getattr(self.harness.knowledge_probe, "enabled", False)):
                    from curriculum.knowledge_prober import select_knowledge

                    selected = select_knowledge(
                        self.harness, tags=tags, bandit=self.knowledge_bandit, rng=self.rng
                    )
                    if selected:
                        inject_selected = bool(
                            getattr(
                                self.harness.knowledge_probe,
                                "inject_into_recovery",
                                True,
                            )
                        )
                        if inject_selected:
                            rec_txt = str(selected["text"])
                        knowledge_meta = {
                            "knowledge_id": selected["skill_id"],
                            "knowledge_context": selected["context"],
                            "knowledge_candidates": selected["candidate_ids"],
                            "knowledge_promoted": selected["promoted"],
                            "knowledge_n_tasks": selected["n_tasks"],
                            "knowledge_injected": inject_selected,
                        }
                else:
                    from curriculum.harness_skill_bank import bind_skill_recovery

                    rec_txt = bind_skill_recovery(self.harness, tags=tags)
            except Exception:
                rec_txt = ""
                knowledge_meta = {}
            if not rec_txt:
                rec_txt = str(self.harness.recovery_prompt or "").strip()
            if rec_txt:
                if legacy:
                    instr = rec_txt + "\n" + instr
                else:
                    extras.append(rec_txt)
        mem_txt = self._format_memory_context(memory_hits) if self.harness.memory.inject_into_prompt else ""
        if mem_txt:
            extras.append(mem_txt)
        extra_user = "\n".join(extras) if extras else None

        if getattr(self.vla.cfg, "mode", "stub") in ("stub",):
            # Deterministic / stub path (Qwen stub or Gemini stub)
            name = prefer or "forward"
            if recovering:
                name = prefer or "forward"
            if prefer is None and self.rng.random() < 0.15:
                name = self.rng.choice(["forward", "left", "right", "turn_left"])
            kb, ms = _kb_ms_from_name(name)
            return name, kb, ms, {
                "source": "stub", "prefer": prefer, "name": name, "actions": None,
                **knowledge_meta,
            }

        # Legacy ignores chat_history; minecraft_v1 consumes the last N turns.
        win = int(getattr(self.harness.runtime, "vla_history_window", 10) or 10)
        out = self.vla.act_image(
            obs["image"],
            instruction=instr,
            chat_history=None if legacy else chat_history,
            step_num=int(step_num),
            history_window=0 if legacy else win,
            extra_user_text=extra_user,
        )
        name = _name_from_kb_ms(out["keyboard"], out["mouse"])
        actions = out.get("actions")
        if prefer and self.harness.verify.prefer_memory_action:
            if memory_hits and memory_hits[0][1] >= 0.75:
                name = prefer
                kb, ms = _kb_ms_from_name(name)
                return name, kb, ms, {
                    "source": "memory_override",
                    "vla_text": out.get("text", ""),
                    "actions": None,
                    **knowledge_meta,
                }
        src = "gemini" if "Gemini" in type(self.vla).__name__ else "vla"
        return name, out["keyboard"], out["mouse"], {
            "source": src,
            "text": out.get("text", ""),
            "actions": actions,
            **knowledge_meta,
        }

    def _retrieve(self, z: torch.Tensor, kb: torch.Tensor, ms: torch.Tensor) -> List[Any]:
        if not self.harness.memory.enabled:
            return []
        s = pool_latent(z).detach().cpu().reshape(-1)
        a = action_embed(kb, ms).detach().cpu().reshape(-1)
        return self.memory.nearest(s, a, topk=self.harness.memory.retrieve_topk)

    def _should_probe(
        self,
        step: int,
        n_probe: int,
        stall_count: int = 0,
        *,
        failing: bool = False,
    ) -> bool:
        p = self.harness.probe
        if not p.enabled:
            return False
        if n_probe >= p.budget_per_episode:
            return False
        # Prefer failure / stall triggers — do not randomly poke a healthy rollout
        if bool(getattr(p, "probe_on_fail", False)) and failing:
            return True
        if (
            bool(getattr(p, "probe_on_stall", False))
            and stall_count >= max(1, int(getattr(p, "stall_steps", 8) or 8))
        ):
            return True
        if not bool(getattr(p, "probe_periodic", True)):
            return False
        return step > 0 and (step % max(1, p.every_n_steps) == 0)

    def _probe_context(self, trigger: str) -> str:
        meta = dict(getattr(self.harness, "meta", None) or {})
        task_type = str(meta.get("current_task_type") or "").lower()
        task_path = str(getattr(self.harness.runtime, "task_config", "") or "").lower()
        if not task_type:
            for known in ("mine_block", "kill_entity", "craft_item"):
                if known in task_path:
                    task_type = known
                    break

        family = {
            "mine_block": "mine",
            "kill_entity": "combat",
            "craft_item": "craft",
        }.get(task_type)
        if family is None:
            text = str(self.harness.runtime.instruction or "").lower()
            if any(x in text for x in ("craft", "制作", "合成")):
                family = "craft"
            elif any(x in text for x in ("mine", "dig", "矿", "挖")):
                family = "mine"
            elif any(x in text for x in ("kill", "attack", "combat", "defeat", "击杀", "战斗")):
                family = "combat"
            else:
                family = "generic"
        return f"{family}:{trigger}"

    def _probe_action(self, step: int, *, context: str = "generic:periodic") -> str:
        pool = list(self.harness.probe.action_pool) or list(ACTION_NAMES)
        strategy = str(
            getattr(self.harness.probe, "selection_strategy", "coverage") or "coverage"
        ).lower()
        if strategy == "ucb":
            return self.probe_bandit.select(context, pool, self.rng)
        if strategy == "coverage" and self.harness.probe.prefer_uncertain:
            # prefer families with low coverage
            cov = self.memory.coverage_report(action_names=pool)
            unknown = [n for n, v in cov["per_action"].items() if v.get("unknown", 0) >= 1.0]
            if unknown:
                return self.rng.choice(unknown)
        return self.rng.choice(pool)

    def _probe_utility(
        self,
        *,
        progress_before: float,
        progress_after: float,
        env_reward: float,
        novelty: float,
        failed: bool,
    ) -> float:
        p = self.harness.probe
        return float(
            float(getattr(p, "progress_reward_weight", 1.0)) * (progress_after - progress_before)
            + float(getattr(p, "env_reward_weight", 0.25)) * float(env_reward)
            + float(getattr(p, "novelty_reward_weight", 0.15)) * float(novelty)
            - float(getattr(p, "probe_cost", 0.05))
            - (float(getattr(p, "failure_penalty", 0.5)) if failed else 0.0)
        )

    def _verify_gate(
        self,
        z: torch.Tensor,
        kb: torch.Tensor,
        ms: torch.Tensor,
        name: str,
    ) -> Tuple[bool, float, Optional[str]]:
        """Return (allow, abstain, alt_action)."""
        v = self.harness.verify
        if not v.enabled:
            return True, 0.0, None
        s = pool_latent(z).detach().cpu().reshape(-1)
        a = action_embed(kb, ms).detach().cpu().reshape(-1)
        abstain = self.memory.abstain_score(s, a)
        # also block known-bad (revoked near) actions
        nn = self.memory.nearest(s, a, topk=1)
        if nn and nn[0][0].status == "revoked" and nn[0][1] >= v.memory_sim_thresh:
            return False, 1.0, "forward"
        if abstain >= v.abstain_thresh and v.block_on_abstain:
            # try memory-preferred safe action
            alt = "forward"
            for it, sim in self.memory.nearest(s, a, topk=5):
                if it.status == "mastered" and sim >= v.memory_sim_thresh:
                    alt = str((it.meta or {}).get("action") or "forward")
                    break
            return False, abstain, alt
        return True, abstain, None

    def _write_memory(
        self,
        z: torch.Tensor,
        kb: torch.Tensor,
        ms: torch.Tensor,
        z2: torch.Tensor,
        *,
        action: str,
        success: bool,
        source: str,
        step: int,
        note: str = "",
    ) -> bool:
        m = self.harness.memory
        if not m.enabled:
            return False
        if source == "probe" and not m.write_probe:
            return False
        if success and not m.write_success:
            return False
        if (not success) and not m.write_failure:
            return False

        # Synthetic probe_scores: success → high; failure → low causal
        if success:
            scores = {
                "action_causality": 0.8,
                "counterfactual": 0.7,
                "state_transfer": 0.6,
                "long_horizon": 0.55,
            }
            conf = max(m.min_confidence, 0.6)
        else:
            scores = {
                "action_causality": 0.2,
                "counterfactual": 0.2,
                "state_transfer": 0.3,
                "long_horizon": 0.3,
            }
            conf = m.min_confidence  # may still write failure if gate soft

        # For failures, force a write via temporary lower gate by tagging meta;
        # WorldKnowledgeMemory.propose may reject — then store a revoked-leaning item
        # by temporarily lowering min_confidence.
        old_min = self.memory.min_confidence
        if not success and m.write_failure:
            self.memory.min_confidence = min(old_min, 0.15)
            scores = {
                "action_causality": 0.5,
                "counterfactual": 0.5,
                "state_transfer": 0.5,
                "long_horizon": 0.5,
            }
            conf = 0.5
        try:
            item = self.memory.propose(
                z, kb, ms, z2,
                probe_scores=scores,
                confidence=conf,
                meta={
                    "action": action,
                    "success": success,
                    "source": source,
                    "step": step,
                    "note": note,
                },
            )
        finally:
            self.memory.min_confidence = old_min

        if item is None:
            return False
        if not success:
            # mark as caution: revoke so verify can block repeats
            self.memory.revoke_item(item.item_id, probe_scores=scores, note=note or "exec_fail")
        elif item.verify_count >= 2:
            self.memory.mark_mastered(item.item_id, step=step)
        return True

    def run_vla_skill(
        self,
        env: Any,
        instruction: str,
        *,
        max_chunks: int = 4,
        obs: Optional[Dict[str, Any]] = None,
        use_memory: bool = True,
        recovering: bool = False,
    ) -> Dict[str, Any]:
        """Treat frozen Qwen/Gemini VLA as a multi-chunk *skill* (Harness-VLA style).

        Planner / outer harness calls this with a subgoal; VLA closed-loops on
        MineStudio for ``max_chunks`` action chunks. Does **not** invent low-level
        keys itself — that stays inside ``vla.act_image``.
        """
        if obs is None:
            obs = env.reset() if hasattr(env, "reset") else {}
        instr = (instruction or "").strip() or str(self.harness.runtime.instruction)
        old_instr = str(self.harness.runtime.instruction)
        self.harness.runtime.instruction = instr
        # Keep VLA policy instruction in sync when present
        if hasattr(self.vla, "cfg") and hasattr(self.vla.cfg, "instruction"):
            self.vla.cfg.instruction = instr

        chunks: List[Dict[str, Any]] = []
        ep_reward = 0.0
        done = False
        success = False
        last_info: Dict[str, Any] = {}
        skill_history: List[Dict[str, Any]] = []
        try:
            for i in range(max(1, int(max_chunks))):
                z = _latent_from_obs(obs)
                kb0, ms0 = _kb_ms_from_name("forward")
                hits = self._retrieve(z, kb0, ms0) if use_memory else []
                name, kb, ms, meta = self._propose_action(
                    obs,
                    memory_hits=hits,
                    force_name=None,
                    recovering=recovering,
                    step_num=i,
                    chat_history=skill_history,
                )
                allow, _abstain, alt = self._verify_gate(z, kb, ms, name)
                if not allow and alt:
                    name = alt
                    kb, ms = _kb_ms_from_name(alt)
                    meta = {**(meta or {}), "actions": None, "source": "verify_block"}

                actions = (meta or {}).get("actions")
                vla_text = str((meta or {}).get("text") or (meta or {}).get("vla_text") or "")
                if (
                    actions
                    and hasattr(env, "step_action_chunk")
                    and str(self.harness.runtime.backend).lower()
                    in ("minestudio", "sandbox", "openha")
                ):
                    obs, reward, done, info = env.step_action_chunk(
                        actions, primary_name=name
                    )
                else:
                    obs, reward, done, info = env.step_action_name(name)

                if vla_text and (meta or {}).get("source") in ("vla", "gemini"):
                    post_img = None
                    if isinstance(obs, dict) and obs.get("image") is not None:
                        post_img = obs["image"]
                        try:
                            post_img = post_img.copy()
                        except Exception:
                            pass
                    if post_img is not None:
                        skill_history.append({
                            "step_num": int(i),
                            "assistant_content": vla_text,
                            "image": post_img,
                        })

                ep_reward += float(reward)
                last_info = info if isinstance(info, dict) else {}
                fail = bool(last_info.get("fail", False))
                z2 = _latent_from_obs(obs)
                wrote = self._write_memory(
                    z, kb, ms, z2,
                    action=name,
                    success=not fail,
                    source="skill",
                    step=i,
                    note=f"skill:{instr[:40]}",
                )
                chunks.append(
                    {
                        "i": i,
                        "name": name,
                        "source": (meta or {}).get("source"),
                        "reward": float(reward),
                        "fail": fail,
                        "done": bool(done),
                        "memory_write": bool(wrote),
                        "text": (meta or {}).get("text", "")[:200],
                    }
                )
                if isinstance(obs, dict) and obs.get("success"):
                    success = True
                    done = True
                if done or fail:
                    break
        finally:
            self.harness.runtime.instruction = old_instr
            if hasattr(self.vla, "cfg") and hasattr(self.vla.cfg, "instruction"):
                self.vla.cfg.instruction = old_instr

        if isinstance(obs, dict) and obs.get("success"):
            success = True
        return {
            "success": bool(success),
            "done": bool(done),
            "reward": float(ep_reward),
            "chunks_used": len(chunks),
            "chunks": chunks,
            "obs": obs,
            "last_info": last_info,
            "instruction": instr,
        }

    def run_episode(
        self,
        *,
        env: Any = None,
        episode_seed: Optional[int] = None,
    ) -> EpisodeResult:
        h = self.harness
        seed = self.seed if episode_seed is None else int(episode_seed)
        owns_env = env is None
        if env is None:
            env = make_env(h, seed=seed)
        try:
            return self._run_episode_loop(env, h, seed)
        finally:
            if owns_env and hasattr(env, "close"):
                try:
                    env.close()
                except Exception:
                    pass

    def _run_episode_loop(self, env: Any, h: HarnessSpec, seed: int) -> EpisodeResult:
        # Per-episode harness RNG: OpenHA world seed stays in task_config;
        # episode_seed only diversifies probe choice / stall recover timing.
        self.rng = random.Random(int(seed) ^ 0xA5A5A5A5)
        obs = env.reset()
        ep_reward = 0.0
        n_probe = n_block = n_recover = n_write = n_hit = 0
        fails_in_row = 0
        cascade = False
        retries = 0
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
        warmup_left = int(getattr(h.probe, "warmup_probes", 0) or 0)
        probe_events: List[Dict[str, Any]] = []
        knowledge_events: List[Dict[str, Any]] = []
        # OpenHA multi_turn: past (user image + assistant text) records.
        # image_path semantics mirror openha_eval: POST-action screenshot.
        chat_history: List[Dict[str, Any]] = []

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

            z = _latent_from_obs(obs)
            kb0, ms0 = _kb_ms_from_name("forward")
            hits = self._retrieve(z, kb0, ms0)
            if hits:
                n_hit += 1

            recovering = fails_in_row > 0 and h.recover.enabled
            if (
                h.recover.enabled
                and bool(getattr(h.recover, "recover_on_stall", False))
                and stall_count >= max(1, int(getattr(h.recover, "stall_steps", 12) or 12))
                and retries < h.recover.max_retries
            ):
                recovering = True
            self._recover_tags = []
            if recovering:
                if stall_count >= max(1, int(getattr(h.recover, "stall_steps", 12) or 12)):
                    self._recover_tags.append("stall_heavy")
                if fails_in_row > 0:
                    self._recover_tags.extend(
                        ["zero_progress", "target_loss", "orientation"]
                    )

            probing = False
            probe_trigger = ""
            if warmup_left > 0 and h.probe.enabled and n_probe < h.probe.budget_per_episode:
                probing = True
                probe_trigger = "warmup"
                warmup_left -= 1
            elif self._should_probe(
                step,
                n_probe,
                stall_count=stall_count,
                failing=(fails_in_row > 0 or recovering),
            ):
                probing = True
                if bool(getattr(h.probe, "probe_on_fail", False)) and (fails_in_row > 0 or recovering):
                    probe_trigger = "fail"
                elif (
                    bool(getattr(h.probe, "probe_on_stall", False))
                    and stall_count >= max(1, int(getattr(h.probe, "stall_steps", 8) or 8))
                ):
                    probe_trigger = "stall"
                else:
                    probe_trigger = "periodic"

            force = None
            probe_context = ""
            if probing:
                probe_context = self._probe_context(probe_trigger)
                force = self._probe_action(step, context=probe_context)
                n_probe += 1
                self.stats["probe"] += 1

            name, kb, ms, meta = self._propose_action(
                obs,
                memory_hits=hits,
                force_name=force,
                recovering=recovering,
                step_num=step,
                chat_history=chat_history,
            )

            allow, abstain, alt = self._verify_gate(z, kb, ms, name)
            if not allow and alt:
                n_block += 1
                self.stats["verify_block"] += 1
                name = alt
                kb, ms = _kb_ms_from_name(alt)
                meta = {**(meta or {}), "actions": None, "source": "verify_block"}

            if (
                h.recover.enabled
                and step > 0
                and step % max(1, h.runtime.checkpoint_every) == 0
                and not getattr(env, "last_fail", False)
            ):
                env.mark_checkpoint()

            # Prefer full VLA chunk on MineStudio; else named action
            actions = (meta or {}).get("actions")
            # Optional replay-only safety cap. A decoded VLA chunk can expand to
            # hundreds of low-level sandbox ticks even when action_chunk_len is
            # small; keeping this opt-in preserves evaluation semantics.
            replay_action_cap = int(os.environ.get("HARNESS_MAX_ACTIONS_PER_STEP", "0") or 0)
            if actions and replay_action_cap > 0:
                actions = list(actions)[:replay_action_cap]
            probe_novelty = 0.0
            if probing:
                probe_novelty = self.memory.novelty(
                    pool_latent(z).detach().cpu().reshape(-1),
                    action_embed(kb, ms).detach().cpu().reshape(-1),
                )
            vla_text = str((meta or {}).get("text") or (meta or {}).get("vla_text") or "")
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
            # success must not be treated as a recover-able fail
            if done and (info.get("success") or getattr(env, "_task_success", False)):
                fail = False

            if probing:
                progress_after = float(
                    (obs2.get("progress") if isinstance(obs2, dict) else None)
                    or getattr(env, "progress", 0.0)
                    or 0.0
                )
                utility = self._probe_utility(
                    progress_before=prog_now,
                    progress_after=progress_after,
                    env_reward=float(reward),
                    novelty=probe_novelty,
                    failed=fail,
                )
                if str(getattr(h.probe, "selection_strategy", "coverage")).lower() == "ucb":
                    self.probe_bandit.update(probe_context, name, utility, kind="immediate")
                probe_events.append({
                    "step": int(step),
                    "trigger": probe_trigger,
                    "context": probe_context,
                    "action": name,
                    "progress_delta": float(progress_after - prog_now),
                    "env_reward": float(reward),
                    "novelty": float(probe_novelty),
                    "failed": bool(fail),
                    "immediate_utility": float(utility),
                })

            knowledge_id = str((meta or {}).get("knowledge_id") or "")
            if knowledge_id:
                progress_after = float(
                    (obs2.get("progress") if isinstance(obs2, dict) else None)
                    or getattr(env, "progress", 0.0)
                    or 0.0
                )
                kp = h.knowledge_probe
                utility = (
                    float(getattr(kp, "progress_reward_weight", 1.0))
                    * (progress_after - prog_now)
                    + float(getattr(kp, "env_reward_weight", 0.25)) * float(reward)
                    - float(getattr(kp, "probe_cost", 0.02))
                    - (
                        float(getattr(kp, "failure_penalty", 0.5))
                        if fail else 0.0
                    )
                )
                context = str((meta or {}).get("knowledge_context") or "unknown:recover")
                if (
                    str(getattr(kp, "selection_strategy", "ucb")).lower() == "ucb"
                    and bool(getattr(kp, "update_from_outcome", True))
                ):
                    self.knowledge_bandit.update(
                        context, knowledge_id, utility, kind="immediate"
                    )
                knowledge_events.append({
                    "step": int(step),
                    "context": context,
                    "knowledge_id": knowledge_id,
                    "candidate_ids": list((meta or {}).get("knowledge_candidates") or []),
                    "promoted": bool((meta or {}).get("knowledge_promoted")),
                    "n_tasks": int((meta or {}).get("knowledge_n_tasks") or 0),
                    "injected": bool((meta or {}).get("knowledge_injected", True)),
                    "progress_delta": float(progress_after - prog_now),
                    "env_reward": float(reward),
                    "failed": bool(fail),
                    "immediate_utility": float(utility),
                })

            # Append OpenHA-style multi_turn history after a real VLA call.
            if vla_text and (meta or {}).get("source") in ("vla", "gemini"):
                post_img = None
                if isinstance(obs2, dict) and obs2.get("image") is not None:
                    post_img = obs2["image"]
                    try:
                        post_img = post_img.copy()
                    except Exception:
                        pass
                elif isinstance(obs, dict) and obs.get("image") is not None:
                    post_img = obs["image"]
                    try:
                        post_img = post_img.copy()
                    except Exception:
                        pass
                if post_img is not None:
                    chat_history.append({
                        "step_num": int(step),
                        "assistant_content": vla_text,
                        "image": post_img,
                    })
                    # Bound memory: keep a bit more than window
                    win = int(getattr(h.runtime, "vla_history_window", 10) or 10)
                    if len(chat_history) > max(win * 2, win + 2):
                        chat_history = chat_history[-(win * 2) :]

            wrote = self._write_memory(
                z, kb, ms, z2,
                action=name,
                success=not fail,
                source="probe" if probing else "exec",
                step=step,
                note="probe" if probing else ("fail" if fail else "ok"),
            )
            if wrote:
                n_write += 1
                self.stats["memory_write"] += 1

            stall_recover = (
                h.recover.enabled
                and bool(getattr(h.recover, "recover_on_stall", False))
                and stall_count >= max(1, int(getattr(h.recover, "stall_steps", 12) or 12))
                and not done
                and retries < h.recover.max_retries
            )
            if (fail or stall_recover) and not done:
                fails_in_row += 1
                if fails_in_row >= 3:
                    cascade = True
                if h.recover.enabled and retries < h.recover.max_retries:
                    n_recover += 1
                    retries += 1
                    self.stats["recover"] += 1
                    if h.recover.use_memory_checkpoint:
                        obs = env.rollback()
                    stall_count = 0
                    step += 1
                    continue
            else:
                fails_in_row = 0
                retries = 0
                if h.recover.enabled and h.recover.use_memory_checkpoint and not done:
                    if step % max(1, h.runtime.checkpoint_every) == 0 or name == "forward":
                        env.mark_checkpoint()

            obs = obs2
            step += 1

        # success: MineStudio latches _task_success; stub uses progress.
        prog = float(getattr(env, "progress", 0.0))
        tgt = float(getattr(env, "target", 1.0))
        success = bool(getattr(env, "_task_success", False))
        if isinstance(obs, dict) and obs.get("success"):
            success = True
        if not success and tgt > 0 and prog >= tgt * 0.99:
            success = True

        adaptive_probe = str(getattr(h.probe, "selection_strategy", "coverage")).lower() == "ucb"
        if adaptive_probe:
            terminal_reward = float(getattr(h.probe, "downstream_success_reward", 1.0))
            discount = float(getattr(h.probe, "downstream_discount", 0.8))
            for distance, event in enumerate(reversed(probe_events)):
                credit = terminal_reward * (discount ** distance) if success else 0.0
                event["downstream_credit"] = float(credit)
                if credit:
                    self.probe_bandit.update(
                        event["context"], event["action"], credit, kind="delayed"
                    )
            state_path = str(getattr(h.probe, "bandit_state_path", "") or "")
            if state_path and not self._suppress_bandit_save:
                self.probe_bandit = self.probe_bandit.merge_delta_save(
                    state_path, self._initial_probe_bandit
                )
                self._initial_probe_bandit = copy.deepcopy(self.probe_bandit.to_dict())

        adaptive_knowledge = (
            bool(getattr(h.knowledge_probe, "enabled", False))
            and str(getattr(h.knowledge_probe, "selection_strategy", "ucb")).lower() == "ucb"
        )
        update_knowledge = bool(
            adaptive_knowledge
            and bool(getattr(h.knowledge_probe, "update_from_outcome", True))
        )
        if update_knowledge:
            terminal_reward = float(
                getattr(h.knowledge_probe, "downstream_success_reward", 1.0)
            )
            discount = float(getattr(h.knowledge_probe, "downstream_discount", 0.8))
            for distance, event in enumerate(reversed(knowledge_events)):
                credit = terminal_reward * (discount ** distance) if success else 0.0
                event["downstream_credit"] = float(credit)
                if credit:
                    self.knowledge_bandit.update(
                        event["context"], event["knowledge_id"], credit, kind="delayed"
                    )
            state_path = str(getattr(h.knowledge_probe, "bandit_state_path", "") or "")
            if state_path and not self._suppress_bandit_save:
                self.knowledge_bandit = self.knowledge_bandit.merge_delta_save(
                    state_path, self._initial_knowledge_bandit
                )
                self._initial_knowledge_bandit = copy.deepcopy(
                    self.knowledge_bandit.to_dict()
                )

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
                # OpenHA world seed is task_config["seed"] (often 2025), not this.
                "diversity": "harness_rng+warmup+stall",
                "world_seed_source": "task_config",
                "probe_events": probe_events,
                "probe_bandit": self.probe_bandit.summary() if adaptive_probe else {},
                "knowledge_events": knowledge_events,
                "knowledge_bandit": (
                    self.knowledge_bandit.summary() if adaptive_knowledge else {}
                ),
            },
        )

    def evaluate(
        self,
        n_episodes: int = 8,
        *,
        base_seed: Optional[int] = None,
        persist_memory: bool = True,
        episode_seeds: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Sandbox-scored eval.

        If ``episode_seeds`` is set, those env seeds are used (paired J(H)).
        ``persist_memory=False`` resets memory before each seed so
        R(H; s_i) are independent — required for paired ACCEPT/REVERT.
        """
        if episode_seeds:
            seeds = [int(s) for s in episode_seeds]
        else:
            base = self.seed if base_seed is None else int(base_seed)
            seeds = [base + i * 17 for i in range(int(n_episodes))]
        results: List[EpisodeResult] = []
        bandit_baseline = copy.deepcopy(self._initial_probe_bandit)
        knowledge_bandit_baseline = copy.deepcopy(self._initial_knowledge_bandit)
        old_suppress_save = self._suppress_bandit_save
        if not persist_memory:
            self._suppress_bandit_save = True
        try:
            for s in seeds:
                if not persist_memory:
                    m = self.harness.memory
                    self.memory = WorldKnowledgeMemory(
                        merge_thresh=m.merge_thresh,
                        max_items=m.max_items,
                        min_confidence=m.min_confidence,
                    )
                    # Paired J(H) requires independent policy state as well as
                    # independent knowledge memory for every evaluation seed.
                    self.probe_bandit = ContextualProbeBandit.from_dict(bandit_baseline)
                    self.knowledge_bandit = ContextualProbeBandit.from_dict(
                        knowledge_bandit_baseline
                    )
                er = self.run_episode(episode_seed=int(s))
                results.append(er)
        finally:
            self._suppress_bandit_save = old_suppress_save
            if not persist_memory:
                self.probe_bandit = ContextualProbeBandit.from_dict(bandit_baseline)
                self.knowledge_bandit = ContextualProbeBandit.from_dict(
                    knowledge_bandit_baseline
                )

        succ = sum(1 for r in results if r.success) / max(1, len(results))
        avg_r = sum(r.reward for r in results) / max(1, len(results))
        cascade = sum(1 for r in results if r.cascade_fail) / max(1, len(results))
        avg_steps = sum(r.steps for r in results) / max(1, len(results))
        score = float(succ) * 1.0 + 0.05 * float(avg_r) / 10.0 - 0.2 * float(cascade)
        return {
            "score": score,
            "success_rate": float(succ),
            "avg_reward": float(avg_r),
            "cascade_rate": float(cascade),
            "avg_steps": float(avg_steps),
            "n_episodes": len(results),
            "eval_seeds": list(seeds),
            "memory_items": len(self.memory),
            "memory_stats": copy.deepcopy(self.memory.stats),
            "runtime_stats": copy.deepcopy(self.stats),
            "episodes": [r.to_dict() for r in results],
            "harness_fp": self.harness.fingerprint(),
        }

    def evaluate_on_seeds(
        self,
        seeds: Sequence[int],
        *,
        persist_memory: bool = False,
    ) -> Dict[str, Any]:
        """Paired-eval helper: independent R(H; s_i) on a fixed seed list."""
        return self.evaluate(
            n_episodes=len(list(seeds)),
            persist_memory=persist_memory,
            episode_seeds=seeds,
        )
