# Session Summary: Quest 023

**周期**: 2026-05-18 → 2026-05-20  起点：Quest 022 已交付ICLR paper（DrugProt 0.696 macro）

---

## 1. 关键发现：Quest 022的0.696是positive-only artifact

| Eval protocol | DrugProt macro F1 | BioRED macro F1 |
|---|---|---|
| Positive-only（quest 022报告） | 0.696 | 0.22 |
| Full-RE（含no_relation candidates） | **0.039** | **0.105** |

模型没在no_relation上训练 → 把每个entity-pair都分到最近的positive label。reviewers会拒。

## 2. Quest 023 v1：multi-schema synth + bi-encoder

**方案**：
- 6 schema unified labels（DrugProt/ChemProt 13 + DDI 4 + GDA 3 + BioRED 8 + CDR 1 + PharmGKB 1 + no_relation = 27）
- 从PubMed未标注文档切passage，Sonnet（via Claude Code Agent dispatch，cursor API exhausted）生成label+短reasoning
- 11K合成样本
- 训练时含no_relation；eval时post-hoc τ_cal阈值

**结果（single-token bi-encoder）**：

| 系统 | BioRED full-RE | BioTriplex cross-schema | 备注 |
|---|---|---|---|
| v1（DrugProt only） | 0.039 | n/a | 全部collapse到positive |
| v3（5.4K, 500步） | 0.099 raw, **0.158**+τ_cal | n/a | τ_cal=0.20 |
| v4（11K, 1000步） | **0.201** | **0.024** | 双远低于BioREx 0.796 |

**Teacher quality**：Sonnet标的负样本39% FP，real F1=0.523（teacher本身就不准）。

## 3. v4 root cause: embedding crowding

| 指标 | v4 |
|---|---|
| Mean inter-label cosine (training labels n=27) | +0.401 |
| Mean inter-label cosine (BioTriplex n=21) | +0.566 |
| INDIRECT-UPREG ↔ INDIRECT-DOWNREG cosine | **+0.936** |
| BioTriplex 8/20 标签 collapse到 Biomarker/Therapeutic | — |

两个独立成因：
- **A. Single-token bottleneck**：pool `last_hidden_state[:, -1, :]`，600+ token context压缩成单个2560-d向量
- **B. Anisotropic last-token EOS pool**：LLM2Vec ablation证实是causal LM最差的pool

## 4. Quest 023 v2 设计 (CODI-Bi)

**文献依据**（详见`02_quest_023_codi_design.md` §1.3）：
- Coconut（Meta 2024）：multi-token continuous thoughts，hidden-state recursion
- CODI（EMNLP 2025）：joint teacher/student forward，hidden state alignment at designated positions
- "Distilling System 2 into System 1"：output-level distill失败，hidden state alignment是必经之路

**核心变化**：

| | v4 (bottleneck) | v5 (CODI-Bi) |
|---|---|---|
| Pool | last_hidden_state[-1] (1 token) | mean over {k=4 latents + `<answer>`} (5 tokens) |
| 训练 | Pass A / Pass B 两次forward, post-hoc distill | Joint teacher/student forward, single optimizer step |
| Loss | L_cls(B) + KL(A||B) + L_gen | L_cls_S + αL_cls_T + βL_align + γL_distill |
| Label encoding | label text → last token | label text + `<think>` + k latents + `<answer>` → 同样5-token pool |
| 特殊token | 无 | `<think>` `<answer>` + 4 learnable latents (nn.Parameter) |
| Warmup | 无 | 200步 cls-only → 200步 linear ramp β,γ |

权重：α=0.5, β=1.0, γ=0.5, τ=0.07, lora_r=16

## 5. Smoke v5: 训练成功，但crowding gate失败

**训练**（2000步 / 58分钟 / 13GB GPU mem）：

| 步数区间 | avg_loss | acc_S | acc_T |
|---|---|---|---|
| 0-199（warmup, no align） | 3.733 | 0.374 | 0.436 |
| 800-999（peak） | 2.010 | 0.639 | **0.835** |
| 1800-1999（end，mild overfit） | 2.694 | 0.647 | 0.772 |

- align_gap：1.13 → 0.40（CODI joint training机制工作）
- Student plateau 0.65 vs teacher 0.83 → **compression bottleneck still exists**

**Embedding diagnostic on v5 — GATE FAILED**：

| 指标 | v4 | v5 | 期望 | 结果 |
|---|---|---|---|---|
| Mean inter-label cosine | +0.401 | **+0.546** | < +0.40 | ❌ 更差 |
| INDIRECT-UPREG ↔ DOWNREG | +0.936 | **+0.900** | < +0.70 | ❌ 仍crowded |
| Association ↔ no_relation | n/a | **+0.909** | — | 新发现的alarming pair |
| ACTIVATOR ↔ Positive_Correlation | n/a | +0.928 | — | 跨schema同义collapse |

**F1 evals（BioRED/BioTriplex/DrugProt full-RE）：因容器迁移中断，未跑完。**

## 6. 两个被证伪的hypothesis

- **H1（Quest 022）**：Silent CoT post-hoc distillation泛化到full-RE  → 被full-RE eval证伪
- **H2（Quest 023 v2 CODI-Bi）**：multi-token pool + joint training 解决embedding crowding → 被v5 diagnostic证伪

## 7. 下一步候选方向（详见 05_next_steps.md）

按改动量从小到大：
- **A** Per-label latents：k×n_labels个latent（label-conditional reasoning）
- **B** Drop `<answer>` from pool，只pool k latents（消除`<answer>`聚合主导）
- **C** Cosine-loss-as-align替代MSE（梯度magnitude放大）
- **D** Coconut-style hidden-state recursion（latent_t input = latent_{t-1} output）
- **E** Bidi attention on latent block（NV-Embed latent attention风格）
- **F** Generative pivot：完全抛弃bi-encoder，<think>→answer text generation

## 8. 这个分支包含什么

```
quest_023_codi/
├── src/                  # 7 CODI模块 + 6 unit test文件（all pass）
├── scripts/              # compute_mean / smoke_tiny / train_codi / eval_codi / embedding_diag_v5
├── tests/                # 7 test files
├── docs/                 # 本目录: 00 summary, 01 brief v1, 02 design v2, 03 plan, 04 results, 05 next
├── artifacts/            # smoke_v5_train.log, smoke_tiny.log, diag.log
├── cross_eval/
│   ├── scripts/          # parse_biored_full.py, parse_drugprot_full.py, parse_pubtator.py, run_eval.py
│   └── label_dicts/      # biored_full.json, biotriplex.json, ddi.json, gda.json, drugprot_full.json, training_v4_labels.json
├── conftest.py, pyproject.toml
```

**未committed的大文件**：
- 11K synth数据 (`/tmp/synth_smoke/training_v4/synth_train.jsonl`, ~20MB)
- v5 adapter (`/home/ds/quest_023_codi/artifacts/smoke_v5/lora_adapter/`, ~1GB)
- 需在新容器中重生成或rsync。
