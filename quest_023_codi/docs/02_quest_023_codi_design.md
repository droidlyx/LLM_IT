# Quest 023 — Silent CoT v2 (CODI-Bi): Design Spec

**Date**: 2026-05-19
**Status**: Design approved, awaiting plan
**Supersedes**: portions of `quest_023_brief.md` (the "Method" section). Data pipeline, eval setup, BioREx comparator, and timeline from the original brief still hold.
**Predecessors**: Quest 022 (Silent CoT v1, DrugProt-only single-schema), smoke v4 (Quest 023 v4, multi-schema synth, 1000 steps on 11K samples)

---

## 1. Motivation: why redesign

### 1.1 v4 results that triggered the rethink

After Quest 022's v1 (positive-only) collapsed on full-RE (0.696 → 0.039), we built a multi-schema synth dataset and re-trained at scale:

| System | BioRED full-RE macro F1 | BioTriplex cross-schema macro F1 | Notes |
|---|---|---|---|
| v1 (DrugProt only, positive-only train) | 0.039 | — | full-RE collapse |
| v3 (5.4K synth, 500 steps) raw | 0.099 | — | overpredicts Association |
| v3 + post-hoc τ=0.20 | 0.158 | — | dev-tuned threshold |
| v4 (11K synth, 1000 steps) | **0.201** | **0.024** | best so far |
| BioREx (38K human labels) | ~0.796 | n/a | upper-bound comparator |

v4 plateaus far below BioREx. The diagnostic showed the cap is not data scale.

### 1.2 Two root causes from `embedding_diag.py`

**Cause A — Single-token bottleneck**: bi-encoder uses `last_hidden_state[:, -1, :]` — one 2560-dim vector compresses 600+ token context. For 27+ training labels with subtle direction (e.g., INDIRECT-UPREG vs INDIRECT-DOWNREG), one vector cannot carry directional information.

**Cause B — Pooling anisotropy**: LLM2Vec ablation across MTEB confirms last-token EOS pool is the worst choice for causal LMs. Mean inter-label cosine: training labels +0.401, BioTriplex labels +0.566. Worst pair: INDIRECT-UPREG ↔ INDIRECT-DOWNREG at **+0.936**.

Cheap fixes (mean pool, whitening) address B but leave A untouched. Architectural redesign is required.

### 1.3 Literature read

The dominant fix in 2024-2025 latent-reasoning literature is **multi-token continuous thoughts**:

- **Coconut (Hao et al., Dec 2024)**: feeds last hidden state back as next input embedding; `<bot>...<eot>` brackets a contiguous block of continuous thoughts. Each reasoning step = c continuous thoughts (c=2 for GSM8k); total latent positions = c × N_steps. Not a single bottleneck.
- **CODI (Shen et al., EMNLP 2025)**: jointly trains teacher (explicit CoT) and student (implicit CoT) in a single training pass; aligns hidden activations at designated token positions via self-distillation. **No curriculum, no catastrophic forgetting.** First implicit CoT to match explicit CoT at GPT-2 scale; +28.2% over prior SOTA.
- **"Distilling System 2 into System 1" (Yu et al., Jul 2024)**: explicitly warns that output-level distillation fails for CoT — hidden-state alignment is required.

Our Silent CoT v1 is structurally a **degenerate CODI**: post-hoc two-pass (vs CODI's joint single-pass) + single-token alignment (vs CODI's multi-position). Both degeneracies are fixable without retraining from scratch.

---

## 2. Method: Silent CoT v2 (CODI-Bi)

### 2.1 Input layout

```
[task input (~600 tok)] [<think>] [L_0 L_1 ... L_{k-1}] [<answer>]
```

- `<think>`, `<answer>`: 2 new special tokens added to tokenizer (resize embedding matrix). Init:
  - `<think>` = mean embedding of `[".", "think", "consider"]`
  - `<answer>` = mean embedding of `["the", "answer", "is"]`
- `L_0 ... L_{k-1}`: k trainable `d_model`-dim vectors stored as `nn.Parameter`. Inserted into input via `inputs_embeds` (bypassing tokenizer). Init: Gaussian noise, std=0.02.
- **k = 4** (initial value). Sweep target after smoke: [2, 4, 8].
- **Causal attention preserved**, no mask change. Latents attend to all prior task input; task input does not attend to latents (default causal).

### 2.2 Two-branch joint training

| Branch | Input |
|---|---|
| Teacher | `[task] [<think>] [reasoning text from synth labels] [<answer>]` |
| Student | `[task] [<think>] [L_0..L_{k-1}] [<answer>]` |

- **Same LoRA parameters** shared by both branches.
- Each training step: forward teacher batch + forward student batch separately; backprop summed loss in one optimizer step.
- Memory ~2x vs v4 (two forwards / batch).

### 2.3 Judgment vector

For both teacher and student, pool the last hidden states at the **k+1 positions** `{L_0, ..., L_{k-1}, <answer>}`:
- Teacher: same physical positions (right before `<answer>` are the last k tokens of reasoning text, then `<answer>` itself)
- Student: the k learnable latents + `<answer>`

```
judgment = mean pool over k+1 positions
judgment_centered = judgment - global_mean
judgment_vec = L2_normalize(judgment_centered)
```

`global_mean` is computed once before training on all label templates encoded through the base model, same as v4. Saved to `mean.pt`.

### 2.4 Label encoding

**Change from v4**: labels are encoded with the **same template structure** as inputs.

```
label_input = [template text] [<think>] [L_0..L_{k-1}] [<answer>]
label_emb = mean pool over {L_0..L_{k-1}, <answer>}, centered, normalized
```

- This forces label embeddings and judgment vectors into a **literally shared representation space** (same pool positions, same model).
- Cost: each template +6 tokens. 27 labels × 5 templates × 6 ≈ 800 extra tokens. Negligible.
- Cache: re-encode all labels every step via `LabelCache.sample_per_label()` (same pattern as v4).

### 2.5 Loss function

Four components:

```python
L_cls_S   = CE( cos(judgment_S, label_embs) / τ, gold )    # student task loss
L_cls_T   = CE( cos(judgment_T, label_embs) / τ, gold )    # teacher anchor
L_align   = MSE( judgment_S, judgment_T.detach() )         # CODI core: student → teacher in vec space
L_distill = KL( softmax(logits_S) || softmax(logits_T.detach()) )  # logit distribution match
```

Total:
```
L_total = L_cls_S + α·L_cls_T + β·L_align + γ·L_distill
```

Default weights: **α=0.5, β=1.0, γ=0.5**.

**Why all four**:
- L_cls_S: main objective; without it nothing learns to classify.
- L_cls_T: prevents teacher branch from drifting to wherever is easy to align. CODI paper does the equivalent.
- L_align: CODI's core mechanism. Joint single-pass alignment has gradient backprop on both branches simultaneously → no catastrophic forgetting (the failure mode of Quest 022's post-hoc distillation).
- L_distill: extra soft constraint on logit distribution. Helps in crowded label space where argmax is too coarse.

**Stop-gradient**: β and γ targets are `teacher.detach()`. Teacher is driven only by L_cls_T; student is driven by all four terms.

### 2.6 Warmup curriculum

First **200 steps**: only `L_cls_S + α·L_cls_T` active. β and γ ramped from 0 to target value linearly over steps [200, 400]. Reason: aligning student to a teacher that has not yet learned to classify means aligning to noise. Warmup lets each branch first acquire basic task competence before alignment starts.

### 2.7 Inference

Student branch only.

```
input = [task] [<think>] [L_0..L_{k-1}] [<answer>]
forward → pool → centered → normalized → judgment_vec
scores = judgment_vec @ label_embs_norm.T
pred = argmax(scores)
```

Label embeddings are re-encoded once at eval time with the trained adapter.

### 2.8 Post-hoc threshold calibration for no_relation

Retained from quest 023 v3/v4: on 10% dev split (only signal we use from the target dataset):

```
if (max_positive_score - no_relation_score) < τ_cal:
    pred = "no_relation"
```

τ_cal swept over dev for both macro F1 and BioREx-style F1. Not changed by CODI port.

---

## 3. Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| k (latent positions) | 4 | sweep [2, 4, 8] after smoke v5 |
| LoRA rank | 16 | same as v4 |
| Batch size | 4 | same as v4 |
| Learning rate | 2e-4 | AdamW betas=(0.9, 0.999), wd=0 |
| Cosine temperature τ | 0.07 | same as v4 |
| Total steps | 2000 | doubled vs v4's 1000 (multi-loss converges slower) |
| Warmup steps | 200 (cls only) + 200 (linear ramp) | β,γ active full at step 400 |
| α (L_cls_T) | 0.5 | |
| β (L_align) | 1.0 | |
| γ (L_distill) | 0.5 | |
| Grad clip | 1.0 | same as v4 |
| Max input length | 600 (a) + 800 (b) + 800 (gen) | same as v4 |
| Special token init | mean of related vocab | see 2.1 |
| Latent init | N(0, 0.02²) | std matches Qwen embed init |
| Precision | bf16 mixed | same as v4 |
| Optimizer | AdamW | same as v4 |

---

## 4. Implementation scope

### 4.1 Files to add / modify (from quest 022 worktree as reference)

| File | Action |
|---|---|
| `src/codi_model.py` (new) | LoRA backbone wrapper + k latent nn.Parameter + special token resize + custom forward that builds inputs_embeds for student branch |
| `src/codi_losses.py` (new) | `codi_step()` function: two-branch forward + 4 loss terms + warmup gating |
| `src/codi_data.py` (new) | Dataset class that yields `{task_input, reasoning_text, label_idx}` triples (same as silent_cot data but with `<think>`/`<answer>` markers inserted) |
| `src/codi_label_cache.py` (new) | Label encoding through current backbone with `<think>`/latents/`<answer>` suffix; re-encode every step |
| `scripts/train_codi.py` (new) | Replaces `smoke_v4.py`; same overall shape (data load → mean → train → save adapter) |
| `cross_eval/scripts/run_eval_codi.py` (new) | Same shape as `run_eval.py` but uses CODI student input + multi-pos pool for judgment & label embs |

### 4.2 Files unchanged

- All synth data generation (already complete in `/home/ds/cross_eval/scripts/synth/`)
- `/tmp/synth_smoke/training_v4/synth_train.jsonl` (11K samples) — reused
- `/tmp/synth_smoke/training_v4/label_dict.json` — reused
- BioRED full-RE / BioTriplex / DrugProt full-RE parsed test sets

### 4.3 Out of scope for this design

- k > 8 (memory budget at LoRA-r16)
- Bidi attention modification (separate experiment)
- Generative pivot (Option D, only if CODI-Bi fails)
- Hidden-state recursion (Coconut-style — separate experiment)

---

## 5. Smoke v5 validation plan

### 5.1 Run config
- Data: existing 11K synth (no regeneration)
- Steps: 2000
- k = 4
- ~8h wall time on single GPU

### 5.2 Acceptance gates

**Mandatory eval protocol**: all numbers MUST come from the **full-RE eval** (all valid entity-type-pair candidates including `no_relation`, parsed by `parse_biored_full.py` / `parse_drugprot_full.py` / `biotriplex_test.jsonl`). The positive-only protocol from Quest 022 is forbidden — reporting positive-only F1 does not validate the pivot. Micro F1 must exclude `no_relation` true negatives (TP/(TP+FP+FN) over positive predictions only); accuracy is not a substitute.

**Must-have to declare pivot success**:
- BioRED full-RE macro F1 ≥ **0.25** (vs v4's 0.20)
- Opposite-direction label cosine (INDIRECT-UPREG ↔ INDIRECT-DOWNREG): **< 0.7** (vs v4's 0.94)
  - Measured on judgment vecs of held-out positive-labeled samples, not on label embs alone

**Nice-to-have**:
- BioTriplex macro F1 ≥ **0.08** (vs v4's 0.024)
- DrugProt full-RE macro F1 ≥ **0.25** (vs v1's 0.039)

**Fail conditions → pivot to Option D (generative)**:
- BioRED < 0.20 (no improvement over single-token)
- Opposite-direction cosine still > 0.85 (multi-token pool not separating direction)
- Training instability (loss diverges, L_align dominates)

### 5.3 Diagnostics to run after smoke v5
- `embedding_diag.py` (already exists) — re-run on v5 adapter, check mean inter-label cosine
- Per-class confusion matrix on BioRED full-RE
- Ablation: which loss term carries the gain? Run 3 sub-configs: (α=0, β=0, γ=0 — student-only baseline), (β=0 — without alignment), (γ=0 — without distill)

---

## 6. Risk register

| Risk | Mitigation |
|---|---|
| Memory blows up with double forward at seq 600+ | Reduce batch to 2; grad accumulate; profile first |
| Latents collapse to constant during warmup | Add tiny std-regularizer on latent params (defer until observed) |
| Teacher branch dominates / student never catches up | Monitor `(judgment_S - judgment_T).norm()` per step; if not decreasing, raise β |
| Label embeddings drift away from judgment space mid-training | Re-encode every step (already in design); plot label-emb–judgment cosine for one held-out sample as sanity |
| 2000 steps insufficient | Checkpoint at 1000/1500/2000; pick best on dev |
| Special token IDs not registered properly in saved adapter | Save tokenizer alongside; verify load with simple roundtrip |

---

## 7. Open questions deferred to plan stage

- Exact LoRA target modules: keep v4's setting or expand to include lm_head (for resized embedding learning)?
- Whether to freeze `<think>`/`<answer>` embedding after init or train them
- Whether L_align should be on the pooled judgment_vec or on each of the k+1 positions independently
- Optimizer state at warmup boundary (200 steps): reset or continue?

These get resolved during writing-plans.

---

## 8. Timeline

| Day | Task |
|---|---|
| D1 | Implementation plan (writing-plans skill) |
| D2-D3 | Implement `src/codi_*.py` + `train_codi.py` + tests on tiny fixture |
| D4 | Smoke v5 dry-run (50 steps, verify no NaN, memory budget OK) |
| D5 | Smoke v5 full (2000 steps, ~8h) |
| D6 | Eval triplet (BioRED / BioTriplex / DrugProt) + diagnostics |
| D7 | Decision: scale to 30K synth + go quest 023 full, OR pivot to Option D |
