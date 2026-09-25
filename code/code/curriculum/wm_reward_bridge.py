#!/usr/bin/env python3
"""L2/L3 co-evolve: score ARES episodes/turns with WM dynamics consistency.

Episode: final_score = env_score + λ * agg(turn_unit)
Turn:    score_t     = env_score + λ * unit_t   (for grpo_wm_turn)

unit = clip((mse_ref - mse) / τ, -1, 1)
Uncertainty = action-noise ensemble variance over WM preds (gate for calibration).

Heavy VAE is lazy-loaded once per process. On failure → bonus 0 (env-only).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("wm_reward_bridge")

_LOCK = threading.Lock()
_SCORER: Optional["WMConsistencyScorer"] = None
_SCHEDULE_CACHE: Dict[str, Any] = {"mtime": -1.0, "data": {}}


def _effective_lambdas(lam: float) -> Tuple[float, float]:
    """MSE λ and transfer λ after schedule.json / env override."""
    sched = _load_schedule()
    if sched and "lambda" in sched:
        try:
            lam = float(sched["lambda"])
        except (TypeError, ValueError):
            pass
    lam_t = 0.0
    env_t = os.environ.get("ARES_WM_TRANSFER_LAMBDA", "").strip()
    if env_t:
        try:
            lam_t = float(env_t)
        except (TypeError, ValueError):
            lam_t = 0.0
    if sched and "transfer_lambda" in sched:
        try:
            lam_t = float(sched.get("transfer_lambda"))
        except (TypeError, ValueError):
            pass
    return float(lam), float(lam_t)


def _load_schedule() -> Dict[str, Any]:
    """Outer-loop schedule written by sidecar (λ / β / imag_gamma / imag_horizon)."""
    path = os.environ.get("ARES_WM_SCHEDULE_PATH", "").strip()
    if not path:
        path = os.path.join(_default_ckpt_dir(), "coevolve_schedule.json")
    try:
        m = os.path.getmtime(path)
    except OSError:
        return {}
    if m == _SCHEDULE_CACHE.get("mtime") and _SCHEDULE_CACHE.get("data") is not None:
        return dict(_SCHEDULE_CACHE["data"] or {})
    try:
        data = json.load(open(path))
    except Exception:
        data = {}
    _SCHEDULE_CACHE["mtime"] = m
    _SCHEDULE_CACHE["data"] = data
    return dict(data or {})



def _mg2_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_ckpt_dir() -> str:
    return os.path.join(_mg2_root(), "curriculum", "outputs", "coevolve_vla")


def cheap_latent_from_frames(frames: List[np.ndarray], nfpb: int = 3):
    """Shared RGB→[16,f,44,80] encoder for L2 when Wanx VAE is too heavy.

    Must match between ARES scoring and WM sidecar ingest (ARES_WM_CHEAP_LATENT=1).

    IMPORTANT: This is NOT Qwen3-VL's vision backbone. L2 consistency is
    self-consistency of ToyDynamics in this shared cheap space (scorer ↔ sidecar),
    not cosine-alignment with VLA visual features.
    """
    import torch
    from PIL import Image

    if not frames:
        return torch.zeros(16, nfpb, 44, 80)
    use = list(frames[:nfpb])
    while len(use) < nfpb:
        use.append(use[-1])
    lat = []
    for fr in use:
        im = Image.fromarray(fr.astype(np.uint8)).resize((80, 44))
        x = np.asarray(im).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # 3,H,W
        pad = np.zeros((13, 44, 80), dtype=np.float32)
        lat.append(np.concatenate([x, pad], axis=0))
    return torch.from_numpy(np.stack(lat, 1))  # 16,f,H,W


class WMConsistencyScorer:
    """ToyDynamics + ActionProjection (+ optional Wanx VAE) episode scorer."""

    def __init__(
        self,
        ckpt_dir: str,
        *,
        device: str = "cpu",
        use_vae: bool = True,
        pretrained_model_path: Optional[str] = None,
        nfpb: int = 3,
        max_pairs: int = 4,
        mse_ref: float = 0.035,
        tau: float = 0.01,
    ):
        import torch
        from curriculum.action_codec import chunk_to_mg2
        from curriculum.frozen_wm import ToyDynamics

        self.torch = torch
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.device = torch.device(device)
        self.use_vae = bool(use_vae)
        self.nfpb = int(nfpb)
        self.max_pairs = int(max_pairs)
        # Center on typical cheap-latent MSE (~0.03–0.04), NOT 0.4 (that saturates tanh).
        self.mse_ref = float(mse_ref)
        self.tau = float(tau)
        self.pretrained_model_path = pretrained_model_path or os.path.join(
            _mg2_root(), "pretrained_wm"
        )

        self.wm = ToyDynamics().to(self.device).eval()
        # L2 uses hard codec only (text→MineStudio→kb/mouse); no learnable ActionProjection.
        self.proj = None
        self._chunk_to_mg2 = chunk_to_mg2
        self.vae_enc = None
        self._wm_mtime = -1.0
        self._proj_mtime = -1.0
        self._reload_ckpts()
        if self.use_vae:
            self._init_vae()

    def _acts_to_mg2(self, acts):
        """Direct hard conversion: MineStudio ticks → mean-pooled (kb[1,4], mouse[1,2])."""
        import torch
        kb, ms = self._chunk_to_mg2(acts)
        kb = kb.mean(0, keepdim=True).to(self.device)
        ms = ms.mean(0, keepdim=True).to(self.device)
        return kb, ms

    def _path(self, name: str) -> str:
        return os.path.join(self.ckpt_dir, name)

    def _mtime(self, path: str) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return -1.0

    def _reload_ckpts(self) -> None:
        import torch

        wm_p = self._path("wm_student_latest.pt")
        wm_m = self._mtime(wm_p)
        if wm_m >= 0 and wm_m != self._wm_mtime:
            ck = torch.load(wm_p, map_location="cpu", weights_only=False)
            state = ck.get("toy", ck)
            self.wm.load_state_dict(state, strict=False)
            self.wm.to(self.device).eval()
            self._wm_mtime = wm_m
            logger.info("loaded WM ckpt %s (hard-codec actions, no ActionProjection)", wm_p)

    def _init_vae(self) -> None:
        try:
            # Reuse coevolve encoder (Wanx VAE).
            from curriculum.coevolve_vla import FrameVAEEncoder

            self.vae_enc = FrameVAEEncoder(
                self.pretrained_model_path,
                device=str(self.device),
                dtype="float16" if self.device.type == "cuda" else "float32",
            )
            logger.info("WM reward VAE ready on %s", self.device)
        except Exception as e:
            logger.warning("VAE init failed (%s); using cheap latent fallback", e)
            self.vae_enc = None

    def _encode_block(self, frames: List[np.ndarray]):
        import torch

        arr = np.stack(frames, 0)
        if self.vae_enc is not None:
            return self.vae_enc.encode_frames(arr, F=self.nfpb)["latent"]
        return cheap_latent_from_frames(frames, nfpb=self.nfpb)

    def _unit_from_mse(self, mse: float) -> float:
        mse_ref = float(os.environ.get("ARES_WM_MSE_REF", self.mse_ref))
        tau = float(os.environ.get("ARES_WM_MSE_TAU", self.tau))
        return float(np.clip((mse_ref - mse) / max(tau, 1e-8), -1.0, 1.0))

    def _load_frames(self, image_paths: Sequence[str]) -> List[np.ndarray]:
        from PIL import Image

        frames: List[np.ndarray] = []
        for p in image_paths:
            if not p or not os.path.isfile(p):
                continue
            try:
                frames.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))
            except Exception:
                continue
        return frames

    def score_turns(
        self,
        image_paths: Sequence[str],
        action_texts: Sequence[str],
        *,
        all_turns: bool = True,
        uncert_ensemble: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Per-turn WM consistency + action-noise uncertainty.

        Returns dict with parallel lists (indexed by turn id, missing → None):
          turn_mse, turn_unit, turn_uncert, and episode aggregates.
        """
        import torch
        from curriculum.action_codec import parse_actions_text

        self._reload_ckpts()
        frames = self._load_frames(image_paths)
        n_act = len(action_texts)
        n_turns = min(n_act, max(0, len(frames) - 1))
        empty = {
            "wm_mse": 0.0,
            "wm_bonus_unit": 0.0,
            "wm_uncert": 0.0,
            "n_pairs": 0.0,
            "turn_mse": [],
            "turn_unit": [],
            "turn_uncert": [],
            "mse_ref": float(os.environ.get("ARES_WM_MSE_REF", self.mse_ref)),
            "tau": float(os.environ.get("ARES_WM_MSE_TAU", self.tau)),
        }
        if n_turns <= 0 or len(frames) < self.nfpb + 1:
            return empty

        turns = list(range(n_turns))
        if (not all_turns) and len(turns) > self.max_pairs:
            idx = np.linspace(0, len(turns) - 1, self.max_pairs)
            scored = {turns[int(i)] for i in idx}
        else:
            scored = set(turns)

        k_ens = uncert_ensemble
        if k_ens is None:
            k_ens = int(os.environ.get("ARES_WM_UNCERT_ENSEMBLE", "3"))
        k_ens = max(1, int(k_ens))
        noise_std = float(os.environ.get("ARES_WM_UNCERT_NOISE", "0.05"))

        turn_mse: List[Optional[float]] = [None] * n_turns
        turn_unit: List[Optional[float]] = [None] * n_turns
        turn_uncert: List[Optional[float]] = [None] * n_turns
        turn_imag_unit: List[Optional[float]] = [None] * n_turns
        mses_agg: List[float] = []
        uncerts_agg: List[float] = []

        # Multi-step imagination consistency on real future frames (VLM-safe:
        # still scores the real action tokens, no fake images in GRPO).
        imag_h = max(1, int(os.environ.get("ARES_WM_IMAG_HORIZON", "1")))
        imag_gamma = float(os.environ.get("ARES_WM_IMAG_GAMMA", "0.0"))
        # Allow schedule file override.
        sched = _load_schedule()
        if sched:
            imag_h = max(1, int(sched.get("imag_horizon", imag_h)))
            imag_gamma = float(sched.get("imag_gamma", imag_gamma))

        with torch.no_grad():
            for t in turns:
                if t not in scored:
                    continue
                end = min(t + self.nfpb + 1, len(frames))
                start = end - (self.nfpb + 1)
                if start < 0:
                    continue
                window = frames[start:end]
                acts, _ = parse_actions_text(action_texts[t] if t < n_act else "")
                if acts:
                    kb, ms = self._acts_to_mg2(acts)
                else:
                    kb = torch.zeros(1, 4, device=self.device)
                    ms = torch.zeros(1, 2, device=self.device)
                z_t = self._encode_block(window[:-1]).unsqueeze(0).to(self.device).float()
                z_tp1 = self._encode_block(window[1:]).unsqueeze(0).to(self.device).float()
                pred = self.wm(z_t, kb, ms)
                mse = float(torch.mean((pred - z_tp1) ** 2).item())
                unit_1 = self._unit_from_mse(mse)

                # H-step imagination vs real future latents.
                imag_unit = unit_1
                if imag_h > 1 and imag_gamma > 0:
                    z_hat = z_t
                    imag_mses = []
                    for h in range(imag_h):
                        th = t + h
                        if th >= n_act:
                            break
                        acts_h, _ = parse_actions_text(action_texts[th])
                        if acts_h:
                            kb_h, ms_h = self._acts_to_mg2(acts_h)
                        else:
                            kb_h = torch.zeros(1, 4, device=self.device)
                            ms_h = torch.zeros(1, 2, device=self.device)
                        z_hat = self.wm(z_hat, kb_h, ms_h)
                        # Align real window ending at t+h+1
                        end_r = min(t + h + self.nfpb + 1, len(frames))
                        start_r = end_r - self.nfpb
                        if start_r < 0 or end_r > len(frames):
                            break
                        z_real = (
                            self._encode_block(frames[start_r:end_r])
                            .unsqueeze(0)
                            .to(self.device)
                            .float()
                        )
                        imag_mses.append(float(torch.mean((z_hat - z_real) ** 2).item()))
                    if imag_mses:
                        imag_unit = self._unit_from_mse(float(sum(imag_mses) / len(imag_mses)))

                unit = (1.0 - imag_gamma) * unit_1 + imag_gamma * imag_unit

                # Action-noise ensemble → predictive variance (uncertainty gate).
                uncert = 0.0
                if k_ens > 1 and noise_std > 0:
                    preds = [pred]
                    for _ in range(k_ens - 1):
                        kb_n = kb + noise_std * torch.randn_like(kb)
                        ms_n = ms + noise_std * torch.randn_like(ms)
                        preds.append(self.wm(z_t, kb_n, ms_n))
                    stack = torch.stack(preds, 0)
                    uncert = float(stack.var(dim=0).mean().item())

                turn_mse[t] = mse
                turn_unit[t] = unit
                turn_uncert[t] = uncert
                turn_imag_unit[t] = imag_unit
                mses_agg.append(mse)
                uncerts_agg.append(uncert)

        if not mses_agg:
            return empty
        mse = float(sum(mses_agg) / len(mses_agg))
        return {
            "wm_mse": mse,
            "wm_bonus_unit": self._unit_from_mse(mse),
            "wm_uncert": float(sum(uncerts_agg) / max(1, len(uncerts_agg))),
            "n_pairs": float(len(mses_agg)),
            "turn_mse": turn_mse,
            "turn_unit": turn_unit,
            "turn_uncert": turn_uncert,
            "turn_imag_unit": turn_imag_unit,
            "imag_horizon": float(imag_h),
            "imag_gamma": float(imag_gamma),
            "mse_ref": float(os.environ.get("ARES_WM_MSE_REF", self.mse_ref)),
            "tau": float(os.environ.get("ARES_WM_MSE_TAU", self.tau)),
        }


    def score_episode(
        self,
        image_paths: Sequence[str],
        action_texts: Sequence[str],
    ) -> Dict[str, float]:
        """Return {wm_mse, wm_bonus_unit, n_pairs} where bonus_unit ∈ (-1,1)."""
        full = self.score_turns(image_paths, action_texts, all_turns=False, uncert_ensemble=1)
        return {
            "wm_mse": float(full["wm_mse"]),
            "wm_bonus_unit": float(full["wm_bonus_unit"]),
            "n_pairs": float(full["n_pairs"]),
            "mse_ref": float(full["mse_ref"]),
            "tau": float(full["tau"]),
            "wm_uncert": float(full.get("wm_uncert", 0.0)),
        }


def get_scorer(
    ckpt_dir: Optional[str] = None,
    *,
    device: Optional[str] = None,
    use_vae: Optional[bool] = None,
    max_pairs: Optional[int] = None,
) -> Optional[WMConsistencyScorer]:
    global _SCORER
    ckpt_dir = ckpt_dir or os.environ.get("ARES_WM_CKPT_DIR", "").strip() or _default_ckpt_dir()
    if not os.path.isdir(ckpt_dir):
        logger.warning("WM ckpt dir missing: %s", ckpt_dir)
        return None
    if not os.path.isfile(os.path.join(ckpt_dir, "wm_student_latest.pt")):
        logger.warning("wm_student_latest.pt missing under %s", ckpt_dir)
        return None

    dev = device or os.environ.get("ARES_WM_SCORE_DEVICE", "cpu")
    if use_vae is None:
        use_vae = os.environ.get("ARES_WM_SCORE_USE_VAE", "1").strip() not in ("0", "false", "False")
    if max_pairs is None:
        max_pairs = int(os.environ.get("ARES_WM_SCORE_MAX_PAIRS", "4"))
    mse_ref = float(os.environ.get("ARES_WM_MSE_REF", "0.035"))
    tau = float(os.environ.get("ARES_WM_MSE_TAU", "0.01"))

    with _LOCK:
        if _SCORER is None or _SCORER.ckpt_dir != os.path.abspath(ckpt_dir):
            try:
                _SCORER = WMConsistencyScorer(
                    ckpt_dir,
                    device=dev,
                    use_vae=use_vae,
                    max_pairs=max_pairs,
                    mse_ref=mse_ref,
                    tau=tau,
                )
            except Exception as e:
                logger.warning("WM scorer init failed: %s", e)
                _SCORER = None
        elif _SCORER is not None:
            _SCORER.mse_ref = mse_ref
            _SCORER.tau = tau
            _SCORER.max_pairs = max_pairs
        return _SCORER


def _attach_transfer_essence(
    stats: Dict[str, Any],
    image_paths: Sequence[str],
    action_texts: Sequence[str],
    scorer: "WMConsistencyScorer",
) -> Dict[str, Any]:
    """Add transfer-vs-connection essence into GRPO bonus.

    final_bonus = λ_mse * connection_unit + λ_transfer * essence
    Auto sidecar writes λ via coevolve_schedule.json.
    """
    sched = _load_schedule()
    lam_t = os.environ.get("ARES_WM_TRANSFER_LAMBDA", "").strip()
    if sched and "transfer_lambda" in sched:
        try:
            lam_transfer = float(sched.get("transfer_lambda"))
        except (TypeError, ValueError):
            lam_transfer = float(lam_t) if lam_t else 0.0
    else:
        lam_transfer = float(lam_t) if lam_t else 0.0
    beta = float(os.environ.get("ARES_WM_CONNECTION_BETA", sched.get("connection_beta", 0.7) if sched else 0.7))
    stats = dict(stats)
    stats.setdefault("connection", float(stats.get("wm_bonus_unit", 0.0)))
    stats.setdefault("transfer", 0.0)
    stats.setdefault("essence", 0.0)
    stats["transfer_lambda"] = float(lam_transfer)
    if lam_transfer == 0.0:
        return _attach_probe_reward(stats, image_paths, action_texts, scorer)
    try:
        from curriculum.policy_transfer_probe import score_policy_episode
    except Exception:
        try:
            from policy_transfer_probe import score_policy_episode  # type: ignore
        except Exception as e:
            stats["transfer_error"] = str(e)
            return _attach_probe_reward(stats, image_paths, action_texts, scorer)
    try:
        tstats = score_policy_episode(
            scorer.wm,
            image_paths,
            action_texts,
            device=scorer.device,
            connection_unit=float(stats.get("wm_bonus_unit", 0.0)),
            beta=beta,
        )
    except Exception as e:
        logger.warning("transfer probe failed: %s", e)
        stats["transfer_error"] = str(e)
        return _attach_probe_reward(stats, image_paths, action_texts, scorer)
    stats.update(tstats)
    extra = float(lam_transfer) * float(tstats.get("essence", 0.0))
    # Memory slot alignment → GRPO (λ_slot · align(τ, M))
    lam_slot = 0.0
    if sched and "slot_lambda" in sched:
        try:
            lam_slot = float(sched.get("slot_lambda"))
        except (TypeError, ValueError):
            lam_slot = 0.0
    env_slot = os.environ.get("ARES_WM_SLOT_LAMBDA", "").strip()
    if env_slot:
        try:
            lam_slot = float(env_slot)
        except (TypeError, ValueError):
            pass
    stats["slot_lambda"] = float(lam_slot)
    stats["slot_align"] = 0.0
    if lam_slot != 0.0:
        try:
            from curriculum.knowledge_memory import WorldKnowledgeMemory
            from curriculum.knowledge_consolidation import memory_slot_alignment
            from curriculum.policy_transfer_probe import episode_to_transitions
            mem_path = ""
            if sched:
                mem_path = str(sched.get("memory_path") or "")
            if not mem_path:
                mem_path = os.path.join(_default_ckpt_dir(), "memory_latest.json")
            if os.path.isfile(mem_path):
                mem = WorldKnowledgeMemory.load(mem_path)
                trans = episode_to_transitions(image_paths, action_texts)
                al = memory_slot_alignment(mem, trans)
                stats["slot_align"] = float(al.get("slot_align", 0.0))
                stats["slot_n_hit"] = float(al.get("n_hit", 0.0))
                extra = extra + float(lam_slot) * float(stats["slot_align"])
                stats["wm_slot_bonus"] = float(lam_slot) * float(stats["slot_align"])
        except Exception as e:
            stats["slot_error"] = str(e)
    stats["wm_transfer_bonus"] = float(lam_transfer) * float(tstats.get("essence", 0.0))
    stats["wm_bonus"] = float(stats.get("wm_bonus", 0.0)) + extra
    # If Auto says this is a connection shortcut, optionally shrink MSE credit
    shrink = float(sched.get("mse_shrink", 1.0)) if sched else 1.0
    if tstats.get("shortcut", 0.0) > 0.25:
        shrink = min(shrink, float(os.environ.get("ARES_WM_MSE_SHRINK_ON_SHORTCUT", "0.5")))
    mse_part = float(stats.get("wm_lambda", 0.0)) * float(stats.get("wm_bonus_unit", 0.0))
    stats["wm_bonus"] = mse_part * shrink + extra
    stats["mse_shrink"] = shrink
    stats = _attach_probe_reward(stats, image_paths, action_texts, scorer)
    return stats


def _probe_mode_on(sched: Optional[Dict[str, Any]] = None) -> Tuple[bool, float]:
    sched = sched if sched is not None else _load_schedule()
    on = os.environ.get("ARES_WM_PROBE_MODE", "0").strip() not in ("0", "", "false", "False")
    if sched and str(sched.get("probe_mode", "")).lower() in ("1", "true", "on", "yes"):
        on = True
    alpha = 0.55
    if sched and sched.get("probe_alpha") is not None:
        try:
            alpha = float(sched.get("probe_alpha"))
        except (TypeError, ValueError):
            pass
    env_a = os.environ.get("ARES_WM_PROBE_ALPHA", "").strip()
    if env_a:
        try:
            alpha = float(env_a)
        except (TypeError, ValueError):
            pass
    return bool(on), float(np.clip(alpha, 0.0, 1.0))


def _attach_probe_reward(
    stats: Dict[str, Any],
    image_paths: Sequence[str],
    action_texts: Sequence[str],
    scorer: "WMConsistencyScorer",
) -> Dict[str, Any]:
    """DiaWM: r_probe so GRPO can seek WM cracks (training-time)."""
    sched = _load_schedule()
    on, alpha = _probe_mode_on(sched)
    stats = dict(stats)
    stats["probe_mode"] = 1.0 if on else 0.0
    stats["probe_alpha"] = float(alpha)
    stats.setdefault("r_probe", 0.0)
    if not on:
        return stats
    try:
        from curriculum.knowledge_memory import WorldKnowledgeMemory
        from curriculum.policy_transfer_probe import episode_to_transitions
        from curriculum.probe_policy import score_probe_episode
    except Exception:
        try:
            from knowledge_memory import WorldKnowledgeMemory  # type: ignore
            from policy_transfer_probe import episode_to_transitions  # type: ignore
            from probe_policy import score_probe_episode  # type: ignore
        except Exception as e:
            stats["probe_error"] = str(e)
            return stats
    mem = None
    mem_path = ""
    if sched:
        mem_path = str(sched.get("memory_path") or "")
    if not mem_path:
        mem_path = os.path.join(_default_ckpt_dir(), "memory_latest.json")
    if os.path.isfile(mem_path):
        try:
            mem = WorldKnowledgeMemory.load(mem_path)
        except Exception:
            mem = None
    try:
        trans = episode_to_transitions(image_paths, action_texts)
        pst = score_probe_episode(
            scorer.wm,
            trans,
            mem,
            device=scorer.device,
            uncert=float(stats.get("wm_uncert", 0.0) or 0.0),
        )
    except Exception as e:
        logger.warning("probe reward failed: %s", e)
        stats["probe_error"] = str(e)
        return stats
    stats["r_probe"] = float(pst.get("r_probe", 0.0))
    stats["gap_norm"] = float(pst.get("gap_norm", 0.0))
    stats["gap"] = list(pst.get("gap") or [])
    stats["probe_ranking"] = float(pst.get("ranking", 0.0))
    stats["probe_abstain"] = float(pst.get("abstain", 0.0))
    stats["wm_bonus_task"] = float(stats.get("wm_bonus", 0.0))
    # Mix diagnostic task bonus with probe reward (agent still does env + wm_bonus;
    # probe-mode env mix happens in the agent).
    stats["wm_bonus"] = (
        float(alpha) * float(stats["r_probe"])
        + (1.0 - float(alpha)) * float(stats["wm_bonus_task"])
    )
    return stats


def score_wm_bonus(
    image_paths: Sequence[str],
    action_texts: Sequence[str],
    *,
    lam: float,
    ckpt_dir: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Return (λ * unit_bonus, diagnostics). Episode-level (subsampled)."""
    lam, _lam_t = _effective_lambdas(lam)
    probe_on, _ = _probe_mode_on()
    if lam == 0.0 and _lam_t == 0.0 and not probe_on:
        return 0.0, {"wm_mse": 0.0, "wm_bonus_unit": 0.0, "n_pairs": 0.0, "wm_bonus": 0.0}
    scorer = get_scorer(ckpt_dir)
    if scorer is None:
        return 0.0, {"wm_mse": 0.0, "wm_bonus_unit": 0.0, "n_pairs": 0.0, "wm_bonus": 0.0, "error": "no_scorer"}
    try:
        stats = scorer.score_episode(image_paths, action_texts)
    except Exception as e:
        logger.warning("WM score failed: %s", e)
        return 0.0, {"wm_mse": 0.0, "wm_bonus_unit": 0.0, "n_pairs": 0.0, "wm_bonus": 0.0, "error": str(e)}
    bonus = lam * float(stats["wm_bonus_unit"])
    stats = dict(stats)
    stats["wm_bonus"] = bonus
    stats["wm_lambda"] = lam
    stats = _attach_transfer_essence(stats, image_paths, action_texts, scorer)
    bonus = float(stats.get("wm_bonus", bonus))
    return bonus, stats


def score_wm_turns(
    image_paths: Sequence[str],
    action_texts: Sequence[str],
    *,
    lam: float,
    ckpt_dir: Optional[str] = None,
    all_turns: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """Turn-level WM scoring for co-evolve credit assignment.

    Returns (episode_bonus, stats) where:
      episode_bonus = λ * mean(turn_unit)
      stats['turn_bonus'] = [λ * unit_t or 0.0]
      stats['turn_uncert'] = uncertainty per turn
    """
    # Schedule can override λ (MSE) and transfer λ.
    sched = _load_schedule()
    lam, _lam_t = _effective_lambdas(lam)
    empty = {
        "wm_mse": 0.0,
        "wm_bonus_unit": 0.0,
        "wm_bonus": 0.0,
        "wm_uncert": 0.0,
        "n_pairs": 0.0,
        "turn_mse": [],
        "turn_unit": [],
        "turn_bonus": [],
        "turn_uncert": [],
        "wm_lambda": lam,
    }
    probe_on, _ = _probe_mode_on(sched)
    if lam == 0.0 and _lam_t == 0.0 and not probe_on:
        return 0.0, empty
    scorer = get_scorer(ckpt_dir)
    if scorer is None:
        empty["error"] = "no_scorer"
        return 0.0, empty
    try:
        stats = scorer.score_turns(image_paths, action_texts, all_turns=all_turns)
    except Exception as e:
        logger.warning("WM turn score failed: %s", e)
        empty["error"] = str(e)
        return 0.0, empty
    stats = dict(stats)
    turn_unit = list(stats.get("turn_unit") or [])
    turn_bonus = [float(lam * u) if u is not None else 0.0 for u in turn_unit]
    units_valid = [float(u) for u in turn_unit if u is not None]
    ep_unit = float(sum(units_valid) / len(units_valid)) if units_valid else float(stats.get("wm_bonus_unit", 0.0))
    # Optional bottleneck credit: emphasize worst-predicted turns.
    mode = os.environ.get("ARES_WM_TURN_AGG", "mean").strip().lower()
    if mode == "min" and units_valid:
        ep_unit = float(min(units_valid))
    elif mode == "worst_k" and units_valid:
        k = max(1, int(os.environ.get("ARES_WM_TURN_WORST_K", "3")))
        worst = sorted(units_valid)[:k]
        ep_unit = float(sum(worst) / len(worst))
    bonus = lam * ep_unit
    stats["wm_bonus_unit"] = ep_unit
    stats["wm_bonus"] = bonus
    stats["wm_lambda"] = lam
    stats["turn_bonus"] = turn_bonus
    stats["turn_agg"] = mode
    if sched:
        stats["schedule_round"] = sched.get("round")
    stats = _attach_transfer_essence(stats, image_paths, action_texts, scorer)
    bonus = float(stats.get("wm_bonus", bonus))
    return bonus, stats
