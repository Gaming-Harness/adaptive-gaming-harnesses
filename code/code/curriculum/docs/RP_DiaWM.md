# Diagnosing the Unseen: Self-Evolving World Models via Adversarial Intervention & Knowledge Consolidation

**Working title / shorthand:** DiaWM (Diagnostic World Model)

**Scope:** Simulator-first (Minecraft / MineStudio / Procgen). Training-time diagnosis; deploy-time WM + memory only.

**Local codebase alignment (anonymous):**  
`project_root/curriculum/{probes,knowledge_memory,knowledge_consolidation,auto_vla_coevolve,wm_reward_bridge,eval_wm_visual}.py`

**Note:** VLA–WM **co-evolution** (async dump ↔ sidecar ↔ GRPO consistency) is a **separate** RP: `docs/RP_CoEvolve_VLA.md`. Do not merge the two stories.

---

## Abstract

Current world models (WMs) are typically scored by next-step pixel or latent reconstruction (MSE / LPIPS). Low error does **not** imply causal or interventional understanding: a WM can act as a high-fidelity “replay machine” that collapses under distribution shift.

We propose **DiaWM**, a diagnostic training framework that closes a three-role loop: (i) a **probe policy** (instantiated by a VLA) that actively seeks states/actions where the WM is uncertain or intervention-inconsistent; (ii) a **sandbox** that acts as a resettable verifier for those interventions; (iii) a **WM** that is not only fitted to trajectories but continually **diagnosed**, with failures consolidated into a **neural–symbolic memory** of verified transition residuals rather than raw rollouts.

Empirically, we target Minecraft / MineStudio (and Procgen ablations): higher **intervention ranking / sensitivity scores**, lower long-horizon latent drift under stress tests, and improved **downstream VLA** robustness under OOD task variants—without claiming a free lunch over Dreamer-style MSE on in-distribution next-step prediction alone.

---

## 1. Introduction

**World models as internal simulators.**  
Generalist agents need an internal dynamics model \(F_\phi(s,a)\mapsto s'\) to imagine, plan, and assign credit. In gaming and embodied settings this is typically a latent RSSM / transformer / video diffusion backbone.

**The false prosperity of MSE.**  
Optimizing \(\|F_\phi(s,a)-s'\|^2\) rewards co-occurrence and visual continuity. A model may correctly darken a frame after a “mining” animation while remaining insensitive to resource counters, inventory causality, or counterfactual action swaps—i.e., **connection patterns**, not transferable dynamics.

**Limits of existing VLA–WM co-evolution.**  
Frameworks that use real rollouts to calibrate a WM for policy reward (e.g., VLAW-style co-training) improve task return, but often treat the WM as a **black-box data generator**. They rarely ask: *what does the WM know, what does it not know, and when should it abstain?*

**Insight — break to build.**  
Rather than only waiting for on-policy task data, we treat the sandbox as a **reagent**: a VLA probe policy actively intervenes where WM confidence is lowest; the sandbox grades truth; failed diagnoses are consolidated into memory and used to update \(F_\phi\) and to reshape policy credit. Diagnosis is a **training-time supervisory curriculum**, not a deploy-time requirement for environment resets.

**Contributions.**

1. **DiaWM loop:** formal three-role co-evolution—VLA as adversarial probe executor, sandbox as intervention verifier, WM as the diagnosed and evolving core—with an explicit answer to the “why not just online RL?” paradox (probes are high-information training signals; deploy uses frozen WM + memory).
2. **Intervention-sensitivity probing + neural–symbolic consolidation:** latent probes (passive + active) emit a cognitive-gap descriptor; memory stores \((C,A,E,\mathrm{conf})\) slots with merge / decay / **revoke**, not raw trajectories; training uses \(\mathcal{L}_{pred}+\lambda_{ic}\mathcal{L}_{interv}+\lambda_{mem}\mathcal{L}_{memory}\).
3. **Evaluation protocol + downstream transfer:** Causal/intervention ranking scores, long-horizon stress metrics, knowledge coverage / abstain, and OOD VLA success with a fixed probe/eval suite (Minecraft-oriented) released for follow-up work.

---

## 2. Related Work

### 2.1 Predictive world models (Dreamer, video WMs, Sora-like)

**Gap:** pixel / latent fit ≠ interventional competence. We keep latent prediction but add **intervention consistency** and **diagnostic coverage** as first-class objectives.

### 2.2 VLA–environment interaction (RT-2, PaLM-E, VLAW-style)

**Gap:** WM used as black-box imagination / reward shaper; little **self-diagnosis** of the WM’s knowledge boundary. We invert the role: VLA can be a **probe** whose reward is WM uncertainty / gap, not only task reward.

### 2.3 Causal representation & intervention (CITRIS, etc.)

**Gap:** interventions often target **disentanglement**, not continual **WM self-correction**. We do **not** claim full causal discovery (SEM / IV); we claim **intervention sensitivity** and ranking fidelity under sandbox verification.

### 2.4 External memory (DNC, episodic buffers)

**Gap:** classic memories store trajectories. We store **probe-verified residuals** with confidence decay and revoke under non-stationarity.

---

## 3. Method

### 3.1 Problem setup

Let \(z_t\in\mathcal{Z}\) be a latent state (VAE / RSSM / cheap shared encoder). The WM is
\[
F_\phi:\mathcal{Z}\times\mathcal{A}\to\mathcal{Z},\qquad \hat z_{t+1}=F_\phi(z_t,a_t).
\]
Sandbox \(\mathcal{E}\) is **resettable** and supports action interventions (Minecraft / MineStudio).  

VLA \(\pi_\theta\) has two modes:

| Mode | Reward | Role |
|------|--------|------|
| **Task** | \(r_{\mathrm{env}}\) (+ optional diagnostic credit) | Downstream agent |
| **Probe** | \(r_{\mathrm{probe}}=\mathrm{Uncertainty}(F_\phi)+\eta\cdot\mathrm{GapScore}\) | Active stress tester |

Deploy-time inference uses \(F_\phi\) + memory readout only; probe policy and sandbox interventions are **training-time**.

### 3.2 Probing via intervention

**Passive probes (cheap, on collected \(\tau\))** — latent \(a\) vs \(a'\) rollouts of \(F_\phi\), compared to observed residuals (no extra reset):

- **Action sensitivity / ranking:** for fixed \(z\), actions \(\{a_i\}\), build predicted pairwise divergence \(D_{ij}=d(F(z,a_i),F(z,a_j))\) and sandbox (or buffer) divergence \(\tilde D_{ij}\); score \(=1-\mathrm{KL}(\mathrm{softmax}(D)\|\mathrm{softmax}(\tilde D))\) or ranking accuracy.
- **Counterfactual / transfer / horizon:** stress residual similarity across states and multi-step latent drift with uncertainty gating (chaos-aware: prefer calibration / ensembles over single-point pixel \(d_t\)).

**Active probes (high information)** — \(\pi_\theta\) maximizes uncertainty / gap near low-coverage memory regions; sandbox returns true \(z_{t+1}\).  

**Output:** each probe batch yields a **cognitive gap vector** \(g\in\mathbb{R}^d\) (aggregate of probe scores, novelty, abstain). If \(\|g\|\) or \(1-\mathrm{conf}\) exceeds threshold, trigger consolidation.

**Naming:** we report **intervention sensitivity**, not “causal discovery.”

### 3.3 Neural–symbolic memory \(\mathcal{M}\)

**Slot (differentiable KV):**
\[
m_i=(C_i,A_i,E_i,\mathrm{conf}_i),\quad
C=\mathrm{pool}(z_t),\;
A=\mathrm{pool}(a_t),\;
E=\mathrm{pool}(z_{t+1})-C.
\]

- **Neural bank \(\mathcal{K}\):** prototypes of \((C,A)\) (merge by cosine ≥ \(\tau_{\mathrm{merge}}\)).
- **Symbolic / value bank:** residual \(E\) + mask / meta (action family, probe scores).
- **Write \(G(\tau,\mathrm{Probe})\):** accept only if probes pass; else failure queue (intrinsic re-sample). Optional GNN / attention over local interaction graph can refine \(E\); MVP uses pooled residual + EMA merge (implemented).
- **Read:** retrieve nearest \((C,A)\); inject \(E\) as prior bias into prediction / loss:
  \[
  \tilde z_{t+1}=F_\phi(z_t,a_t)+\alpha\cdot\mathrm{unpool}(E^\star).
  \]
- **Non-stationarity:** confidence decay if unused; **re-verify**; **revoke** on repeated probe failure (avoids locked false priors).

**Coverage / abstain (self-awareness metrics):** family coverage, unknown rate, stale/revoke rate; \(\mathrm{abstain}(z,a)\in[0,1]\).

### 3.4 Self-evolving loss

\[
\mathcal{L}_{\mathrm{total}}
=\underbrace{\mathcal{L}_{\mathrm{pred}}}_{\|F_\phi(z,a)-z'\|^2}
+\lambda_1\underbrace{\mathcal{L}_{\mathrm{interv}}}_{\text{intervention consistency}}
+\lambda_2\underbrace{\mathcal{L}_{\mathrm{memory}}}_{\text{slot recall}}.
\]

**Intervention consistency (contrastive / ranking form).**  
For contrast actions \(a,a'\) (and sandbox or buffer pairs when available):
\[
\Deltâ=F_\phi(z,a)-F_\phi(z,a'),\qquad
\Delta^\star=z'(a)-z'(a')\ \text{or}\ z'-z\ \text{(proxy)},
\]
\[
\mathcal{L}_{\mathrm{interv}}=1-\cos(\Deltâ,\Delta^\star)
\]
(and optionally a listwise ranking loss so orderings of \(\|F(z,a_i)-F(z,a_j)\|\) match sandbox). This forces **directional** sensitivity to actions, not only magnitude MSE.

**Memory recall:**
\[
\mathcal{L}_{\mathrm{memory}}=\mathrm{conf}^\star\cdot\mathrm{sim}^\star\cdot
\|(\mathrm{pool}(F_\phi(z,a))-\mathrm{pool}(z))-E^\star\|^2.
\]

### 3.5 Co-evolution closed loop

```text
┌─────────────┐   low-confidence / gap region   ┌──────────────┐
│ Probe VLA π │ ───────────────────────────────► │   Sandbox E  │
└──────┬──────┘     intervene a_t, get z_{t+1}   └──────┬───────┘
       │                                                  │
       │         compare F_φ(z,a) vs truth                │
       ▼                                                  ▼
┌─────────────┐   write slots / revoke / decay   ┌──────────────┐
│ Probes + g  │ ───────────────────────────────► │ Memory M     │
└──────┬──────┘                                  └──────┬───────┘
       │  L_pred + L_interv + L_mem                     │
       ▼                                                  │
┌─────────────┐ ◄──────── readout prior E* ─────────────┘
│    WM F_φ   │
└──────┬──────┘
       │  schedule: λ_mse, λ_transfer, λ_slot, Skip/Learn
       ▼
┌─────────────┐
│ Task VLA RL │  (GRPO / PPO): r_env + diagnostic credit
└─────────────┘
```

1. Sandbox resets / continues; probe VLA picks \(a_t\) in high-uncertainty / low-coverage regions.  
2. Execute; obtain true \(z_{t+1}\).  
3. Probes compare; on failure → write \(m_i\), update \(\phi\) with \(\mathcal{L}_{\mathrm{total}}\).  
4. Periodic merge + decay + re-verify / revoke.  
5. Task-mode VLA still does RL; diagnostic credit (\(\mathrm{essence}\), slot align) **penalizes connection-only** high MSE fit.

**Paradox answer (for reviewers):** sandbox access during **training** is intentional—like a lab providing graded exams. At **deployment**, probe + sandbox drop out; only \(F_\phi\) and \(\mathcal{M}\) remain, cost ≈ standard WM.

---

## 4. Experiments

### Research questions

- **RQ1 (Diagnosis):** Can intervention probes separate “memorizing” WMs from intervention-sensitive WMs? (Train twin WMs: full data vs partial; compare probe ranking / KL on held-out interventions.)
- **RQ2 (Memory):** Does neural–symbolic memory reduce long-horizon latent error / ranking collapse vs MSE-only / PER?
- **RQ3 (Downstream OOD):** With task VLA fixed or lightly fine-tuned, does DiaWM improve OOD success (unseen crafts, texture/gravity-style shifts in Procgen)?

### Baselines

- DreamerV3 (or project ToyDynamics / pretrained_wm latent WM as controlled backbone)
- DreamerV3 + PER
- DreamerV3 + CITRIS-style causal regularizer (where applicable)
- VLAW-style co-evolve (WM calibrates policy reward without diagnostic memory)

### Metrics

| Metric | Meaning |
|--------|---------|
| Intervention ranking / Causal Score* | Ordering fidelity under action sets (*sensitivity, not SEM) |
| Long-horizon latent drift / mFID | Stress imagination quality |
| Knowledge coverage / abstain / revoke | Self-awareness |
| PSNR/SSIM (sandbox GT vs \(\hat s\)) | Visual diagnostic (optional) |
| Downstream success (ID / OOD) | Task transfer |

\*Report under the name **Intervention Ranking Accuracy** in the paper body to avoid causal-community rejection.

### Ablations

Probe on/off · Memory on/off · \(\mathcal{L}_{interv}\) on/off · Active probe reward on/off · Revoke on/off.

---

## 5. Expected contributions & timeline (4 months)

**Theory / framing:** Diagnostic co-evolution; memory as verified residuals with revoke; clear train vs deploy split.

**Systems / artifacts:** Minecraft intervention probe suite + coverage reports (open-sourced under anonymous).

| Month | Milestone |
|-------|-----------|
| 1 | Probe policy reward + passive/active probes on MineStudio; baseline WM |
| 2 | Memory write/read/merge/revoke; \(\mathcal{L}_{interv}+\mathcal{L}_{mem}\); Auto Learn/Skip |
| 3 | Main + ablations; OOD downstream; visual GT vs WM metrics |
| 4 | Write-up, benchmark packaging, submission |

---

## 6. Discussions & limitations (reviewer FAQ)

**Q: If you can query the sandbox, why not pure online RL?**  
A: Task RL optimizes return; DiaWM optimizes **WM knowledge boundary**. Probes deliberately spend budget on **high-gap interventions**, producing denser diagnostic gradients than incidental task trajectories. Deploy does not need the sandbox.

**Q: Is this causal discovery?**  
A: No. It is **intervention sensitivity + consolidation**. Confounders may remain; we measure ranking / residual agreement under controlled sandbox interventions.

**Q: Won’t memory lock stale knowledge?**  
A: Decay, re-verify, and **revoke** are mandatory; mastered is provisional.

**Q: Simulator-only?**  
A: Yes by design for resettable interventions. Real-world extension needs offline counterfactual pairs / limited physical resets—future work.

**Q: Are you claiming to beat Dreamer MSE?**  
A: Not as the primary claim. Primary claims: **diagnosis quality, coverage/abstain, OOD downstream**. MSE remains an auxiliary.

---

## 7. Mapping to current implementation (integrity check)

| RP module | Code (under `curriculum/`) |
|-----------|----------------------------|
| Passive probes | `probes.py`, `policy_transfer_probe.py` |
| Active probe reward \(r_{\mathrm{probe}}\), gap \(g\), ranking | `probe_policy.py` |
| Slots \((C,A,E)\) + coverage/abstain | `knowledge_memory.py` |
| \(\mathcal{L}_{interv},\mathcal{L}_{mem}\) + memory prior | `knowledge_consolidation.py` |
| Learn/Skip/Re-verify + probe_focus | `auto_scheduler.py`, `auto_vla_coevolve.py` |
| Diagnostic + probe mix → GRPO | `wm_reward_bridge.py` + Qwen agent `_apply_wm_reward` |
| Visual GT vs WM | `eval_wm_visual.py` |

Enable: `ARES_WM_PROBE_MODE=1` `ARES_WM_PROBE_ALPHA=0.55` (cluster scheduler auto script default on).

---

## One-sentence pitch

> **VLA finds the cracks, sandbox grades the truth, the world model learns what it did not know—and remembers it without confusing connection for understanding.**
