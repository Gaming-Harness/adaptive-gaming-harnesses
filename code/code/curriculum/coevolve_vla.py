#!/usr/bin/env python3
"""Co-evolve Visual WM + Qwen VLA through MineStudio sandbox.

Architecture (all under anonymous):
  base VLA (READ-ONLY weights)
       │
       ▼
  act on POV ──► MineStudio sandbox (714) ──► (o,a,r,o')
       │                                         │
       │                                         ▼
       │                              VAE encode → (z,a,z')
       │                                         │
       └──── imagination (frozen/EMA WM) ◄───────┘
                         │
              update F_φ (on-policy + expert + L_sd)
              VLA stays base / RL still in ARES free_energy

VLA *policy gradient* remains in ARES
  tasks/.../minestudio_qwen3_vl_8b/config_free_energy.py
  (suffix free_energy_openha_32tasks_v2).
This loop adapts the **world model** to p_π and provides imagination signals.

Usage:
  export MINESTUDIO_TOKEN=...
  python -m curriculum.coevolve_vla --config curriculum/configs/coevolve_vla.yaml

Smoke without sandbox / 8B load:
  python -m curriculum.coevolve_vla --config curriculum/configs/coevolve_vla_stub.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from curriculum.action_codec import chunk_to_mg2, mg2_to_sandbox, sandbox_to_mg2, parse_actions_text
from curriculum.action_projection import (
    ActionProjection,
    chunk_to_feature,
    hard_codec_mg2,
    projection_align_loss,
)
from curriculum.cascaded_wam import CascadedWAM, InverseActionHead
from curriculum.ares_wm_bridge import list_episode_dirs
from curriculum.frozen_wm import FrozenWorldModel, ToyDynamics
from curriculum.qwen_vla import QwenVLAConfig, QwenVLAPolicy
from curriculum.replay_buffer import MixedReplayBuffer, Transition
from curriculum.sandbox_bridge import MineStudioSandbox, SandboxConfig, extract_pil, pov_to_uint8


class FrameVAEEncoder:
    """Encode POV frames with pretrained_wm Wanx VAE → latent block [16,f,H,W]."""

    def __init__(self, pretrained_model_path: str, device: str = "cuda", dtype: str = "float16"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype)
        from wan.vae.wanx_vae import get_wanx_vae_wrapper
        self.vae = get_wanx_vae_wrapper(pretrained_model_path, self.dtype)
        self.vae.requires_grad_(False).eval()
        self.vae = self.vae.to(self.device, self.dtype)
        self.tiler = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}

    @torch.no_grad()
    def encode_frames(self, frames_uint8: np.ndarray, F: int = 3) -> Dict[str, torch.Tensor]:
        """frames: [T,H,W,3] uint8. Returns latent [16,F,44,80] + cond fields."""
        from train.data.vpt_to_worldmodel import frame_process
        T_need = 4 * (F - 1) + 1
        if frames_uint8.shape[0] < T_need:
            pad = np.repeat(frames_uint8[-1:], T_need - frames_uint8.shape[0], axis=0)
            frames_uint8 = np.concatenate([frames_uint8, pad], axis=0)
        frames_uint8 = frames_uint8[:T_need]
        frames = torch.from_numpy(frames_uint8)
        pixel = frame_process(frames, (352, 640)).to(self.device, self.dtype)[None]
        first = pixel[:, :, 0:1]
        padding = torch.zeros_like(first).repeat(1, 1, 4 * (F - 1), 1, 1)
        img_cond_in = torch.cat([first, padding], dim=2)
        img_cond = self.vae.encode(img_cond_in, device=self.device, **self.tiler)
        mask = torch.ones_like(img_cond)
        mask[:, :, 1:] = 0
        cond_concat = torch.cat([mask[:, :4], img_cond], dim=1)
        latent = self.vae.encode(pixel, device=self.device, **self.tiler)
        visual_context = self.vae.clip.encode_video(first)
        return {
            "latent": latent[0].float().cpu(),
            "cond_concat": cond_concat[0].float().cpu(),
            "visual_context": visual_context[0].float().cpu(),
        }


class CoEvolveVLA:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if cfg.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.logdir = cfg.logdir
        os.makedirs(self.logdir, exist_ok=True)
        self.nfpb = int(cfg.num_frame_per_block)

        # WM student + EMA teacher
        init_wm = cfg.get("toy_wm_ckpt", None)
        wrap = FrozenWorldModel(backend="toy", device=str(self.device), toy_ckpt=init_wm)
        self.wm_teacher: ToyDynamics = wrap.toy
        for p in self.wm_teacher.parameters():
            p.requires_grad_(False)
        self.wm_student = ToyDynamics().to(self.device)
        self.wm_student.load_state_dict(self.wm_teacher.state_dict())
        self.wm_opt = torch.optim.AdamW(
            self.wm_student.parameters(), lr=float(cfg.wm_lr), weight_decay=float(cfg.weight_decay)
        )
        self.ema_tau = float(cfg.ema_tau)

        # Qwen VLA (collaborator_b base, read-only) — skip for ARES-dump WM-only updates
        self.vla = None
        if not bool(cfg.get("skip_vla", False)):
            vla_cfg = QwenVLAConfig(
                model_path=str(cfg.vla_model_path),
                mode=str(cfg.vla_mode),
                device=str(cfg.get("vla_device", "cuda")),
                dtype=str(cfg.get("vla_dtype", "bfloat16")),
                max_new_tokens=int(cfg.get("max_new_tokens", 256)),
                instruction=str(cfg.get("instruction", "Explore and progress the Minecraft task.")),
            )
            self.vla = QwenVLAPolicy(vla_cfg)
        else:
            print("[coevolve_vla] skip_vla=true (WM / ARES-ingest only)")

        # VLA action → WM conditioning (learnable; trained by L_dyn on dump data)
        self.action_proj = ActionProjection(
            hidden=int(cfg.get("proj_hidden", 256)),
            cond_dim=int(cfg.get("proj_cond_dim", 64)),
        ).to(self.device)
        self.train_action_proj = bool(cfg.get("train_action_proj", False))
        self.lambda_proj_align = float(cfg.get("lambda_proj_align", 0.0))
        proj_ckpt = cfg.get("action_proj_ckpt") or os.path.join(self.logdir, "action_projection.pt")
        if proj_ckpt and os.path.exists(str(proj_ckpt)):
            ck = torch.load(str(proj_ckpt), map_location="cpu", weights_only=False)
            self.action_proj.load_state_dict(ck["projection"], strict=False)
            print(f"[coevolve_vla] loaded action projection {proj_ckpt}")
        self.proj_opt = torch.optim.AdamW(
            self.action_proj.parameters(),
            lr=float(cfg.get("proj_lr", 1e-4)),
            weight_decay=float(cfg.weight_decay),
        )
        print(
            f"[coevolve_vla] train_action_proj={self.train_action_proj} "
            f"lambda_proj_align={self.lambda_proj_align}",
            flush=True,
        )

        # Cascaded WAM: F (wm_student) + InverseActionHead π(z)→a
        self.train_cascaded_wam = bool(cfg.get("train_cascaded_wam", False))
        self.lambda_act = float(cfg.get("lambda_act", 0.0))
        # Default 0: L_open poisons F when π is bad (async co-evolve).
        self.lambda_open = float(cfg.get("lambda_open", 0.0))
        self.action_head = InverseActionHead(
            hidden=int(cfg.get("act_hidden", 256)),
        ).to(self.device)
        wam_ckpt = cfg.get("cascaded_wam_ckpt") or os.path.join(self.logdir, "cascaded_wam.pt")
        if wam_ckpt and os.path.exists(str(wam_ckpt)):
            ck = torch.load(str(wam_ckpt), map_location="cpu", weights_only=False)
            if "action" in ck:
                self.action_head.load_state_dict(ck["action"], strict=False)
            elif "action_head" in ck:
                self.action_head.load_state_dict(ck["action_head"], strict=False)
            print(f"[coevolve_vla] loaded cascaded WAM action head {wam_ckpt}")
        self.act_opt = torch.optim.AdamW(
            self.action_head.parameters(),
            lr=float(cfg.get("act_lr", 1e-4)),
            weight_decay=float(cfg.weight_decay),
        )
        self.cascaded_wam = CascadedWAM(
            world=self.wm_student,
            action=self.action_head,
            lambda_act=self.lambda_act,
            lambda_open=self.lambda_open,
        )
        print(
            f"[coevolve_vla] train_cascaded_wam={self.train_cascaded_wam} "
            f"lambda_act={self.lambda_act} lambda_open={self.lambda_open}",
            flush=True,
        )

        # Sandbox (optional for stub / ARES-dump ingest)
        self.sandbox = None
        self.vae_enc = None
        if cfg.use_sandbox:
            sc = SandboxConfig.from_env()
            if cfg.get("minestudio_endpoint"):
                sc.endpoint = str(cfg.minestudio_endpoint)
            self.sandbox = MineStudioSandbox(
                sc, img_save_dir=os.path.join(self.logdir, "sandbox_images")
            )
        # VAE needed for real latents even when ingesting ARES dumps (no sandbox).
        if cfg.get("encode_vae", True) and cfg.vla_mode != "stub":
            self.vae_enc = FrameVAEEncoder(
                str(cfg.pretrained_model_path),
                device=str(self.device),
                dtype=str(cfg.get("vae_dtype", "float16")),
            )

        self.buf = MixedReplayBuffer(expert_ratio=float(cfg.expert_ratio))
        if cfg.get("data_root") and os.path.isdir(cfg.data_root):
            n = self.buf.load_expert_pt_dir(
                cfg.data_root, max_clips=cfg.get("max_clips", None),
                block_frames=self.nfpb, stride=int(cfg.get("block_stride", 1)),
            )
            print(f"[coevolve_vla] expert transitions: {n}")

        self.task_list = self._load_tasks(cfg.get("task_dir", None))
        self.step = 0
        self.history: List[Dict] = []

    def _load_tasks(self, task_dir: Optional[str]) -> List[str]:
        if not task_dir or not os.path.isdir(task_dir):
            return []
        files = sorted(
            os.path.join(task_dir, f)
            for f in os.listdir(task_dir)
            if f.endswith((".json", ".yaml", ".yml"))
        )
        print(f"[coevolve_vla] {len(files)} task configs from {task_dir}")
        return files

    def _encode_block(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Stack recent frames → latent block [16,f,H,W]."""
        arr = np.stack(frames, 0)
        if self.vae_enc is None:
            # Shared cheap encoder (aligned with ARES L2 scorer when VAE off).
            from curriculum.wm_reward_bridge import cheap_latent_from_frames
            return cheap_latent_from_frames(list(arr), nfpb=self.nfpb)
        out = self.vae_enc.encode_frames(arr, F=self.nfpb)
        return out["latent"]

    def collect_sandbox_episode(self) -> Dict[str, float]:
        assert self.sandbox is not None
        task = None
        if self.task_list:
            task = self.task_list[self.step % len(self.task_list)]
        obs = self.sandbox.reset(task_config_file_path=task)
        img = extract_pil(obs)
        if img is None:
            raise RuntimeError("sandbox reset returned no image")

        instruction = str(self.cfg.get("instruction", "Explore and progress the Minecraft task."))
        if isinstance(obs, dict):
            instruction = obs.get("instruction") or obs.get("task_text") or instruction

        max_turns = int(self.cfg.max_turns)
        frames_buf: List[np.ndarray] = [pov_to_uint8(img)]
        ep_reward = 0.0
        history_text: List[str] = []
        n_trans = 0

        for t in range(max_turns):
            assert self.vla is not None, "collect_sandbox_episode requires VLA (skip_vla=false)"
            out = self.vla.act_image(img, instruction=instruction, history_text=history_text)
            history_text.append(out["text"][:200])
            actions = out["actions"]
            # Bridge: store VLA features for data-driven ActionProjection training in update_wm.
            action_feat = None
            if actions:
                action_feat = chunk_to_feature(actions).mean(0)  # [D]
                kb_h, ms_h = hard_codec_mg2(actions)
                kb = kb_h.mean(0)
                ms = ms_h.mean(0)
                if self.train_action_proj:
                    # warm-start align step (optional; main learning is L_dyn)
                    loss_p = projection_align_loss(self.action_proj, actions)
                    self.proj_opt.zero_grad(set_to_none=True)
                    loss_p.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.action_proj.parameters(), float(self.cfg.max_grad_norm)
                    )
                    self.proj_opt.step()
            else:
                kb, ms = out["keyboard"].mean(0), out["mouse"].mean(0)

            # execute chunk on sandbox with raw VLA actions
            next_obs = self.sandbox.step(actions if len(actions) > 1 else actions[0])
            r = 0.0
            if isinstance(next_obs, dict):
                try:
                    r = float(next_obs.get("reward", 0.0) or 0.0)
                except (TypeError, ValueError):
                    r = 0.0
            ep_reward = r  # OpenHA uses final reward
            img2 = extract_pil(next_obs) or img
            frames_buf.append(pov_to_uint8(img2))

            # build transition every nfpb frames
            if len(frames_buf) >= self.nfpb + 1:
                z_t = self._encode_block(frames_buf[-(self.nfpb + 1):-1])
                z_tp1 = self._encode_block(frames_buf[-self.nfpb:])
                # expand projected pretrained_wm action to window for buffer / imagination
                T_a = max(1, 4 * (self.nfpb - 1) + 1)
                if kb.ndim == 1:
                    kb_w = kb.unsqueeze(0).expand(T_a, -1).contiguous()
                    ms_w = ms.unsqueeze(0).expand(T_a, -1).contiguous()
                elif kb.shape[0] < T_a:
                    kb_w = kb.mean(0, keepdim=True).expand(T_a, -1).contiguous()
                    ms_w = ms.mean(0, keepdim=True).expand(T_a, -1).contiguous()
                else:
                    kb_w, ms_w = kb[:T_a], ms[:T_a]
                self.buf.add_policy(Transition(
                    latent_t=z_t, keyboard=kb_w, mouse=ms_w, latent_tp1=z_tp1,
                    reward=float(r), source="policy",
                    action_feat=action_feat,
                    meta={"text": out["text"][:300], "turn": t},
                ))
                # imagination under teacher (use current proj if available)
                with torch.no_grad():
                    if action_feat is not None:
                        kb_i, ms_i, _ = self.action_proj(
                            action_feat.unsqueeze(0).to(self.device)
                        )
                    else:
                        kb_i = kb_w.mean(0, keepdim=True).to(self.device)
                        ms_i = ms_w.mean(0, keepdim=True).to(self.device)
                    z_hat = self.wm_teacher(
                        z_t.unsqueeze(0).to(self.device).float(),
                        kb_i if kb_i.ndim == 2 else kb_i.unsqueeze(0),
                        ms_i if ms_i.ndim == 2 else ms_i.unsqueeze(0),
                    ).squeeze(0).cpu()
                self.buf.add_imagined(Transition(
                    latent_t=z_t, keyboard=kb_w, mouse=ms_w, latent_tp1=z_hat,
                    reward=float(r), source="imagined",
                    action_feat=action_feat,
                ))
                n_trans += 1

            img = img2
            if out.get("done"):
                break
            if isinstance(next_obs, dict) and next_obs.get("terminated"):
                break

        return {"ep_reward": ep_reward, "n_trans": n_trans, "turns": t + 1}

    def collect_stub(self, n: int) -> Dict[str, float]:
        """Offline proxy: VLA stub actions on expert states (no sandbox)."""
        for _ in range(n):
            tr = self.buf.sample(1, prefer="expert")[0]
            # stub forward chunk
            from curriculum.action_codec import noop_action
            acts = [noop_action() for _ in range(4)]
            for a in acts:
                a["forward"] = 1
            kb, ms = chunk_to_mg2(acts)
            z = tr.latent_t.unsqueeze(0).to(self.device).float()
            with torch.no_grad():
                z_hat = self.wm_teacher(
                    z, kb.mean(0, keepdim=True).to(self.device),
                    ms.mean(0, keepdim=True).to(self.device),
                )
            self.buf.add_policy(Transition(
                latent_t=tr.latent_t, keyboard=kb.mean(0), mouse=ms.mean(0),
                latent_tp1=tr.latent_tp1, reward=0.0, source="policy",
            ))
            self.buf.add_imagined(Transition(
                latent_t=tr.latent_t, keyboard=kb.mean(0), mouse=ms.mean(0),
                latent_tp1=z_hat.squeeze(0).cpu(), reward=0.0, source="imagined",
            ))
        return {"ep_reward": 0.0, "n_trans": n, "turns": n}

    def update_wm(self) -> Dict[str, float]:
        B = int(self.cfg.batch_size)
        if len(self.buf.expert) + len(self.buf.policy) < B:
            return {
                "wm_loss": 0.0, "L_dyn": 0.0, "L_sd": 0.0, "L_exp": 0.0, "L_proj": 0.0,
                "L_act": 0.0, "L_open": 0.0, "n_proj": 0,
            }
        mix = self.buf.sample(B)
        z = torch.stack([t.latent_t.float() for t in mix]).to(self.device)
        z1 = torch.stack([t.latent_tp1.float() for t in mix]).to(self.device)

        # Data-driven action projection: policy samples with action_feat go through proj.
        kbs, mss = [], []
        proj_kbs, proj_mss, hard_kbs, hard_mss = [], [], [], []
        n_proj = 0
        for t in mix:
            if self.train_action_proj and t.action_feat is not None:
                feat = t.action_feat.float().to(self.device)
                if feat.ndim == 1:
                    feat = feat.unsqueeze(0)
                kb_p, ms_p, _ = self.action_proj(feat)
                kb_p, ms_p = kb_p.squeeze(0), ms_p.squeeze(0)
                kbs.append(kb_p)
                mss.append(ms_p)
                proj_kbs.append(kb_p)
                proj_mss.append(ms_p)
                kb_h, ms_h = t.keyboard.float(), t.mouse.float()
                hard_kbs.append((kb_h.mean(0) if kb_h.ndim == 2 else kb_h).to(self.device))
                hard_mss.append((ms_h.mean(0) if ms_h.ndim == 2 else ms_h).to(self.device))
                n_proj += 1
            else:
                kb0, ms0 = t.keyboard.float(), t.mouse.float()
                kbs.append((kb0.mean(0) if kb0.ndim == 2 else kb0).to(self.device))
                mss.append((ms0.mean(0) if ms0.ndim == 2 else ms0).to(self.device))
        kb = torch.stack(kbs)
        ms = torch.stack(mss)

        pred = self.wm_student(z, kb, ms)
        loss_dyn = F.mse_loss(pred, z1)
        with torch.no_grad():
            soft = self.wm_teacher(z, kb.detach(), ms.detach())
        loss_sd = F.mse_loss(pred, soft)

        loss_proj = pred.new_zeros(())
        if n_proj > 0 and self.lambda_proj_align > 0:
            loss_proj = F.mse_loss(torch.stack(proj_kbs), torch.stack(hard_kbs)) + F.mse_loss(
                torch.stack(proj_mss), torch.stack(hard_mss)
            )

                # Cascaded WAM action head: π(z)→a. L_open OFF unless lambda_open>0.
        loss_act = pred.new_zeros(())
        loss_open = pred.new_zeros(())
        if self.train_cascaded_wam:
            kb_tgt, ms_tgt = [], []
            for t in mix:
                kb0, ms0 = t.keyboard.float(), t.mouse.float()
                kb_tgt.append((kb0.mean(0) if kb0.ndim == 2 else kb0).to(self.device))
                ms_tgt.append((ms0.mean(0) if ms0.ndim == 2 else ms0).to(self.device))
            kb_tgt_t = torch.stack(kb_tgt)
            ms_tgt_t = torch.stack(ms_tgt)
            kb_hat, ms_hat = self.action_head(z)
            loss_act = F.mse_loss(kb_hat, kb_tgt_t) + F.mse_loss(ms_hat, ms_tgt_t)
            if self.lambda_open > 0:
                z_open = self.wm_student(z, kb_hat, ms_hat)
                loss_open = F.mse_loss(z_open, z1)

        if self.buf.expert:
            exp = self.buf.sample(B, prefer="expert")
            ze = torch.stack([t.latent_t.float() for t in exp]).to(self.device)
            z1e = torch.stack([t.latent_tp1.float() for t in exp]).to(self.device)
            kbe, mse = [], []
            for t in exp:
                kb0, ms0 = t.keyboard.float(), t.mouse.float()
                kbe.append(kb0.mean(0) if kb0.ndim == 2 else kb0)
                mse.append(ms0.mean(0) if ms0.ndim == 2 else ms0)
            loss_exp = F.mse_loss(
                self.wm_student(ze, torch.stack(kbe).to(self.device), torch.stack(mse).to(self.device)),
                z1e,
            )
        else:
            loss_exp = pred.new_zeros(())

        loss = (
            loss_dyn
            + float(self.cfg.lambda_sd) * loss_sd
            + float(self.cfg.lambda_expert) * loss_exp
            + self.lambda_proj_align * loss_proj
            + self.lambda_act * loss_act
            + self.lambda_open * loss_open
        )
        self.wm_opt.zero_grad(set_to_none=True)
        if self.train_action_proj:
            self.proj_opt.zero_grad(set_to_none=True)
        if self.train_cascaded_wam:
            self.act_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wm_student.parameters(), float(self.cfg.max_grad_norm))
        if self.train_action_proj:
            torch.nn.utils.clip_grad_norm_(
                self.action_proj.parameters(), float(self.cfg.max_grad_norm)
            )
        if self.train_cascaded_wam:
            torch.nn.utils.clip_grad_norm_(
                self.action_head.parameters(), float(self.cfg.max_grad_norm)
            )
        self.wm_opt.step()
        if self.train_action_proj and n_proj > 0:
            self.proj_opt.step()
        if self.train_cascaded_wam:
            self.act_opt.step()
        with torch.no_grad():
            for p_t, p_s in zip(self.wm_teacher.parameters(), self.wm_student.parameters()):
                p_t.data.mul_(self.ema_tau).add_(p_s.data, alpha=1.0 - self.ema_tau)
        return {
            "wm_loss": float(loss.item()),
            "L_dyn": float(loss_dyn.item()),
            "L_sd": float(loss_sd.item()),
            "L_exp": float(loss_exp.item()) if torch.is_tensor(loss_exp) else float(loss_exp),
            "L_proj": float(loss_proj.item()) if torch.is_tensor(loss_proj) else float(loss_proj),
            "L_act": float(loss_act.item()) if torch.is_tensor(loss_act) else float(loss_act),
            "L_open": float(loss_open.item()) if torch.is_tensor(loss_open) else float(loss_open),
            "n_proj": int(n_proj),
        }

    def train(self):
        t0 = time.time()
        for self.step in range(1, int(self.cfg.max_steps) + 1):
            if self.cfg.use_sandbox and self.sandbox is not None:
                inter = self.collect_sandbox_episode()
            else:
                inter = self.collect_stub(int(self.cfg.collect_n))
            wm_stats = self.update_wm()
            if self.step % int(self.cfg.log_interval) == 0:
                rec = {"step": self.step, **inter, **wm_stats,
                       "buf_P": len(self.buf.policy), "buf_I": len(self.buf.imagined)}
                self.history.append(rec)
                print(
                    f"[coevolve_vla] step {self.step:4d} | R {inter['ep_reward']:.3f} | "
                    f"trans {inter['n_trans']} | wm {wm_stats['wm_loss']:.4f} "
                    f"(dyn {wm_stats['L_dyn']:.4f} sd {wm_stats['L_sd']:.4f}) | "
                    f"buf P/I {rec['buf_P']}/{rec['buf_I']}",
                    flush=True,
                )
            if self.step % int(self.cfg.save_interval) == 0:
                self.save()
        self.save()
        with open(os.path.join(self.logdir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        if self.sandbox:
            self.sandbox.close()
        print(f"[coevolve_vla] done in {time.time()-t0:.1f}s -> {self.logdir}")

    def ingest_ares_rollouts(self, dump_root: str, max_eps: Optional[int] = None) -> Dict[str, float]:
        """Load ARES GRPO episode dumps → policy buffer via projection + encode."""
        eps = list_episode_dirs(dump_root)
        if max_eps is not None and max_eps > 0 and len(eps) > max_eps:
            # Prefer newest episode dirs (name contains epoch-ms).
            eps = sorted(eps, key=lambda p: os.path.basename(p))[-int(max_eps):]
        n_trans = 0
        for ep_dir in eps:
            meta_path = os.path.join(ep_dir, "meta.json")
            with open(meta_path) as f:
                meta = json.load(f)
            frames = []
            for name in meta.get("frames", []):
                p = os.path.join(ep_dir, name)
                if not os.path.isfile(p):
                    continue
                try:
                    frames.append(pov_to_uint8(Image.open(p).convert("RGB")))
                except Exception as e:
                    # Partial/corrupt dumps on shared filesystem — skip frame, keep episode.
                    print(f"[coevolve_vla] skip bad frame {p}: {e}", flush=True)
                    continue
            actions_path = os.path.join(ep_dir, "actions.jsonl")
            turns: List[Dict[str, Any]] = []
            if os.path.isfile(actions_path):
                with open(actions_path) as f:
                    for line in f:
                        turns.append(json.loads(line))
            if len(frames) < self.nfpb + 1:
                continue
            # pair consecutive frame windows with projected actions
            for t, turn in enumerate(turns):
                end = min(t + self.nfpb + 1, len(frames))
                start = end - (self.nfpb + 1)
                if start < 0:
                    continue
                window = frames[start:end]
                acts, _ = parse_actions_text(turn.get("action_text", ""))
                action_feat = None
                if acts:
                    action_feat = chunk_to_feature(acts).mean(0)  # [D] kept for L_dyn→proj
                    kb_h, ms_h = hard_codec_mg2(acts)
                    kb = kb_h.mean(0)
                    ms = ms_h.mean(0)
                else:
                    kb = torch.zeros(4)
                    ms = torch.zeros(2)
                T_a = max(1, 4 * (self.nfpb - 1) + 1)
                if kb.ndim == 1:
                    kb_w = kb.unsqueeze(0).expand(T_a, -1).contiguous()
                    ms_w = ms.unsqueeze(0).expand(T_a, -1).contiguous()
                elif kb.shape[0] < T_a:
                    kb_w = kb.mean(0, keepdim=True).expand(T_a, -1).contiguous()
                    ms_w = ms.mean(0, keepdim=True).expand(T_a, -1).contiguous()
                else:
                    kb_w, ms_w = kb[:T_a], ms[:T_a]
                z_t = self._encode_block(window[:-1])
                z_tp1 = self._encode_block(window[1:])
                r = float(turn.get("reward", 0.0) or meta.get("episode_score", 0.0) or 0.0)
                self.buf.add_policy(Transition(
                    latent_t=z_t, keyboard=kb_w, mouse=ms_w, latent_tp1=z_tp1,
                    reward=r, source="policy",
                    action_feat=action_feat,
                    meta={"ares_ep": meta.get("episode_id"), "turn": t},
                ))
                n_trans += 1
        return {"ep_reward": 0.0, "n_trans": n_trans, "turns": n_trans, "n_eps": len(eps)}

    def eval_wm(self, n_batches: int = 4) -> Dict[str, float]:
        """Hold-out-ish diagnostic: L_dyn / L_act on fresh samples (no grad)."""
        B = int(self.cfg.batch_size)
        if len(self.buf.expert) + len(self.buf.policy) < B:
            return {"eval_L_dyn": 0.0, "eval_L_act": 0.0, "n": 0}
        dyns, acts = [], []
        self.wm_student.eval()
        self.action_head.eval()
        with torch.no_grad():
            for _ in range(max(1, int(n_batches))):
                mix = self.buf.sample(B)
                z = torch.stack([t.latent_t.float() for t in mix]).to(self.device)
                z1 = torch.stack([t.latent_tp1.float() for t in mix]).to(self.device)
                kb_tgt, ms_tgt = [], []
                for t in mix:
                    kb0, ms0 = t.keyboard.float(), t.mouse.float()
                    kb_tgt.append((kb0.mean(0) if kb0.ndim == 2 else kb0).to(self.device))
                    ms_tgt.append((ms0.mean(0) if ms0.ndim == 2 else ms0).to(self.device))
                kb = torch.stack(kb_tgt)
                ms = torch.stack(ms_tgt)
                pred = self.wm_student(z, kb, ms)
                dyns.append(float(F.mse_loss(pred, z1).item()))
                if self.train_cascaded_wam:
                    kb_hat, ms_hat = self.action_head(z)
                    acts.append(
                        float(
                            (F.mse_loss(kb_hat, kb) + F.mse_loss(ms_hat, ms)).item()
                        )
                    )
        self.wm_student.train()
        self.action_head.train()
        out = {
            "eval_L_dyn": float(sum(dyns) / max(1, len(dyns))),
            "eval_L_act": float(sum(acts) / max(1, len(acts))) if acts else 0.0,
            "n": int(len(dyns) * B),
        }
        return out

    def update_from_ares(
        self,
        dump_root: str,
        wm_steps: int = 20,
        max_eps: Optional[int] = None,
        wm_epochs: Optional[float] = None,
    ) -> None:
        """Consume ARES dumps and run WM updates (sidecar for GRPO↔WM)."""
        inter = self.ingest_ares_rollouts(dump_root, max_eps=max_eps)
        print(f"[coevolve_vla] ingested ARES dumps: {inter}", flush=True)
        n_buf = len(self.buf.policy) + len(self.buf.expert)
        B = max(1, int(self.cfg.batch_size))
        steps = int(wm_steps)
        if wm_epochs is not None and float(wm_epochs) > 0 and n_buf > 0:
            auto = int(max(1, (n_buf + B - 1) // B * float(wm_epochs)))
            steps_min = int(self.cfg.get("wm_steps_min", 100))
            steps = max(steps_min, auto)
            print(
                f"[coevolve_vla] wm_epochs={wm_epochs} buf={n_buf} bs={B} "
                f"→ steps={steps} (min={steps_min}); lambda_open={self.lambda_open}",
                flush=True,
            )
        ev0 = self.eval_wm()
        print(
            f"[coevolve_vla] eval_wm before: L_dyn={ev0['eval_L_dyn']:.4f} "
            f"L_act={ev0['eval_L_act']:.4f} n={ev0['n']}",
            flush=True,
        )
        for self.step in range(1, int(steps) + 1):
            wm_stats = self.update_wm()
            if self.step % max(1, int(self.cfg.log_interval)) == 0 or self.step == int(steps):
                print(
                    f"[coevolve_vla/ares] step {self.step:4d}/{steps} | "
                    f"wm {wm_stats['wm_loss']:.4f} dyn {wm_stats['L_dyn']:.4f} "
                    f"proj {wm_stats.get('L_proj', 0):.4f} act {wm_stats.get('L_act', 0):.4f} "
                    f"open {wm_stats.get('L_open', 0):.4f} n_proj={wm_stats.get('n_proj', 0)}",
                    flush=True,
                )
            if self.step % int(self.cfg.save_interval) == 0:
                self.save()
        ev1 = self.eval_wm()
        print(
            f"[coevolve_vla] eval_wm after:  L_dyn={ev1['eval_L_dyn']:.4f} "
            f"L_act={ev1['eval_L_act']:.4f} "
            f"Δdyn={ev1['eval_L_dyn']-ev0['eval_L_dyn']:+.4f} "
            f"Δact={ev1['eval_L_act']-ev0['eval_L_act']:+.4f}",
            flush=True,
        )
        self.save()

    def save(self):
        def _atomic_torch_save(obj, path: str) -> None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = f"{path}.tmp.{os.getpid()}"
            try:
                torch.save(obj, tmp)
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

        logdir = os.path.abspath(self.logdir)
        os.makedirs(logdir, exist_ok=True)
        _atomic_torch_save(
            {"toy": self.wm_student.state_dict(), "step": self.step},
            os.path.join(logdir, "wm_student_latest.pt"),
        )
        _atomic_torch_save(
            {"toy": self.wm_teacher.state_dict(), "step": self.step},
            os.path.join(logdir, "wm_teacher_latest.pt"),
        )
        _atomic_torch_save(
            {"projection": self.action_proj.state_dict(), "step": self.step},
            os.path.join(logdir, "action_projection.pt"),
        )
        _atomic_torch_save(
            {
                "action": self.action_head.state_dict(),
                "toy": self.wm_student.state_dict(),
                "step": self.step,
                "lambda_act": self.lambda_act,
                "lambda_open": self.lambda_open,
            },
            os.path.join(logdir, "cascaded_wam.pt"),
        )
        # dump recent on-policy .pt for ARES / consistency finetune handoff
        handoff = os.path.join(logdir, "policy_transitions")
        os.makedirs(handoff, exist_ok=True)
        for i, tr in enumerate(self.buf.policy[-32:]):
            _atomic_torch_save({
                "latent": tr.latent_tp1,  # keep schema flexible
                "latent_t": tr.latent_t,
                "latent_tp1": tr.latent_tp1,
                "keyboard": tr.keyboard if tr.keyboard.ndim == 2 else tr.keyboard.unsqueeze(0),
                "mouse": tr.mouse if tr.mouse.ndim == 2 else tr.mouse.unsqueeze(0),
                "reward": tr.reward,
                "meta": tr.meta,
            }, os.path.join(handoff, f"pol_{self.step:06d}_{i:03d}.pt"))
        print(f"[coevolve_vla] saved @ {self.step} -> {logdir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="curriculum/configs/coevolve_vla.yaml")
    ap.add_argument("--update-from-dir", default=None,
                    help="ARES_WM_DUMP_DIR: ingest GRPO rollouts and update WM only")
    ap.add_argument("--wm-steps", type=int, default=None,
                    help="Fixed WM grad steps (overrides epoch auto if set >0)")
    ap.add_argument("--wm-epochs", type=float, default=None,
                    help="Passes over buffer: steps ≈ ceil(N/B)*epochs (min wm_steps_min)")
    ap.add_argument("--max-eps", type=int, default=64,
                    help="Max newest ARES episodes to ingest per round (0=all)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--reinit-wm-student", action="store_true",
                    help="Re-init F_φ from toy/random but keep ActionProjection weights")
    args = ap.parse_args()
    os.chdir(_ROOT)
    cfg = OmegaConf.load(args.config)
    if args.max_steps is not None:
        cfg.max_steps = int(args.max_steps)
    if args.update_from_dir:
        cfg.skip_vla = True
        cfg.use_sandbox = False
        # Keep VAE encode for ARES dump ingest (default True in coevolve_vla.yaml).
        cfg.encode_vae = bool(cfg.get("encode_vae", True))
        # L2 alignment: when ARES scores with cheap latent, WM must train on it too.
        # cheap_latent = shared RGB resize/pad (NOT Qwen3-VL vision tower).
        cheap = os.environ.get("ARES_WM_CHEAP_LATENT", "").strip().lower()
        if cheap in ("1", "true", "yes"):
            cfg.encode_vae = False
            print("[coevolve_vla] ARES_WM_CHEAP_LATENT=1 → encode_vae=false (L2-aligned cheap latent)", flush=True)
        # Force-safe default unless yaml explicitly sets otherwise already loaded.
        if cfg.get("lambda_open") is None:
            cfg.lambda_open = 0.0
    runner = CoEvolveVLA(cfg)
    if args.reinit_wm_student or os.environ.get("ARES_WM_REINIT_STUDENT", "").strip() in ("1", "true", "yes"):
        # Drop possibly L_open-poisoned F; keep proj / action_head.
        init_wm = cfg.get("toy_wm_ckpt", None)
        wrap = FrozenWorldModel(backend="toy", device=str(runner.device), toy_ckpt=init_wm)
        runner.wm_student.load_state_dict(wrap.toy.state_dict())
        runner.wm_teacher.load_state_dict(wrap.toy.state_dict())
        for p in runner.wm_teacher.parameters():
            p.requires_grad_(False)
        print("[coevolve_vla] REINIT wm_student/teacher from toy init; kept action_proj/action_head", flush=True)
    if args.update_from_dir:
        max_eps = None if int(args.max_eps) <= 0 else int(args.max_eps)
        wm_epochs = args.wm_epochs
        if wm_epochs is None:
            wm_epochs = float(cfg.get("wm_epochs", 3))
        wm_steps = int(args.wm_steps) if args.wm_steps is not None else 0
        runner.update_from_ares(
            args.update_from_dir,
            wm_steps=wm_steps,
            max_eps=max_eps,
            wm_epochs=wm_epochs if wm_steps <= 0 else None,
        )
    else:
        runner.train()


if __name__ == "__main__":
    main()
