# 项目总结：生物医学关系抽取（Biomedical RE）

> 给项目外的读者：这份文档让你1分钟看懂我们在做什么、试过什么、当前在哪、下一步去哪。

---

## 0. 问题是什么

**生物医学关系抽取**：从PubMed论文中找出实体（药物、蛋白、基因、疾病）之间的关系。

例：「Aspirin inhibits COX-2 in inflammation」→ 关系是 `INHIBITOR(Aspirin, COX-2)`。

为什么难：
1. **关系类别多、细**：DrugProt 13种，BioRED 8种，整合所有数据集27+种，很多语义相近（如 UPREGULATOR vs DOWNREGULATOR）
2. **每个数据集schema不同**：换一个benchmark就要重标数据；现有最强模型BioREx用了38K人工标注才达到~0.8 macro F1
3. **绝大多数实体对没有关系**：DrugProt里97%的候选pair是 `no_relation`（负样本），但研究界长期只evaluate正样本，导致虚高的报告数

我们的目标：**用LLM + 极少标注做到接近BioREx的效果**（少标注成本，多schema覆盖）。

---

## 1. 一句话讲完Quest 015-021

之前的research chain做了大量探索：

| Phase | 尝试的方向 | 结果 |
|---|---|---|
| Quest 015-020 | 9种post-hoc方法（在frozen baseline上做修正/重排/过滤） | **全部失败**或微弱效果 |
| Quest 015-020 | 2种**训练数据层**的干预（entity rename、KB-hint） | **都有正效果**（BC8 +1pt） |
| **关键结论** | 在SFT data层动手，比在inference时纠错有效一个数量级 | — |
| Quest 021 | 用verbalized prompts + SapBERT相似度做RE | **结构性失败** DrugProt 0.51 / BioRED 0.13 |

详见`[ds-quest-chain-findings]` memory entry。

---

## 2. Quest 022（已交付ICLR论文）

### 方法：Silent CoT

**核心想法**：训练时让模型看到"教师推理文本"，推理时不看。

```
训练：
  Pass A: [文档 + 实体1 + 实体2 + 问题]          → 学生hidden state h_A
  Pass B: [Pass A前置教师推理]                   → 教师hidden state h_B
  Loss: KL(softmax(cos(h_A, label_embs)) || stop_grad(softmax(cos(h_B, label_embs))))

推理：
  只用Pass A的forward，cosine相似度找最近label
```

骨架是 **Qwen3-4B-Base + LoRA r=16 + bi-encoder（cosine to label embeddings）**。教师推理由Claude Sonnet生成，预先做好缓存进训练数据。

### 结果（论文报告的数字）

| 指标 | v1 baseline | Silent CoT (Quest 022) | 相对提升 |
|---|---|---|---|
| DrugProt dev macro-F1 | 0.5126 | **0.6958** | +35.7% |
| ChemProt zero-shot | — | **0.8918** | (跨schema, 近-ontology) |
| BioRED zero-shot | 0.1330 | 0.2209 | (诚实表述：not solved) |

训练成本：6.3 GPU-小时, RTX 4090。论文已在disk上。

---

## 3. Quest 023的起点：发现Quest 022的数字是**positive-only artifact**

### 发现

Quest 022的0.696 DrugProt是在**只评估有正样本标注的候选pair**的协议下计算的。把 `no_relation` 负样本（占97%）也加入evaluation：

| Eval protocol | DrugProt macro F1 | BioRED macro F1 |
|---|---|---|
| Positive-only（论文报告） | 0.696 | 0.22 |
| **Full-RE（含no_relation）** | **0.039** | **0.105** |

为什么塌：模型训练时没见过 `no_relation`，把每个候选pair都强行分到最相近的positive标签。Reviewers会拒。

**Quest 023的使命**：做正确的full-RE，并扩展到multi-schema。

---

## 4. Quest 023 v1：多schema合成数据 + bi-encoder

### 方案

```
未标注PubMed文档
       ↓
Claude Sonnet生成 (label, reasoning) 对  ← Claude Code Agent dispatch
       ↓
27个统一标签 = DrugProt/ChemProt 13 + DDI 4 + GDA 3 + BioRED 8 + CDR 1 + PharmGKB 1 + no_relation
       ↓
11K合成样本 → Qwen3-4B + LoRA训练 → bi-encoder cosine评估
```

API为什么用Sonnet via Claude Code Agent dispatch：cursor.scihub的API key耗尽，只能借agent作为Sonnet proxy。

### 渐进结果

| 系统 | BioRED full-RE | BioTriplex（跨schema） | 备注 |
|---|---|---|---|
| Quest 022 v1 | 0.039 | n/a | 没有no_relation训练 |
| Smoke v3（5.4K, 500步） | 0.099 raw / 0.158 w/ τ_cal | n/a | post-hoc阈值 τ_cal=0.20 |
| Smoke v4（11K, 1000步） | **0.201** | **0.024** | 当前最好 |
| **BioREx（38K人工标注）** | **0.796** | — | 上限 |

我们达到BioREx的 25%（0.201/0.796）。教师质量诊断显示Sonnet给负样本39% FP（teacher本身就不准），real F1 = 0.523。

---

## 5. Quest 023 v4诊断：根本瓶颈是什么

跑embedding几何分析：

```
所有27个label互相的余弦相似度：
  平均：+0.401  （想要：< +0.3，否则区分不开）
  最差对：INDIRECT-UPREGULATOR ↔ INDIRECT-DOWNREGULATOR = +0.936
  （两个语义完全相反的标签，模型却觉得它们几乎一样）

BioTriplex的21个label互相余弦：+0.566（更糟）
  8/20的BioTriplex label在训练空间里都collapse到 "Biomarker" 或 "Therapeutic"
```

**两个独立成因**：
- **A. 单token瓶颈**：bi-encoder用`last_hidden_state[:, -1, :]`，把600+ token的context压成1个2560维向量
- **B. Pool方式anisotropic**：LLM2Vec论文证实causal LM的last-token EOS pool是最差的pool方式

---

## 6. Quest 023 v2 redesign: CODI-Bi

文献调研（详见`02_design.md` §1.3）：
- **Coconut (Meta 2024)**：多token continuous thoughts，hidden state循环喂回
- **CODI (EMNLP 2025)**：师生**同步forward**，在指定token位置对齐hidden state
- **"Distilling System 2 into System 1"**：output-level distill失败，必须做hidden state alignment

**结论**：Quest 022的Silent CoT v1是CODI的**退化版本**（单token + 两次forward post-hoc蒸馏）。CODI修正了两个退化点。

### CODI-Bi设计

```
输入：[task input] [<think>] [L₀ L₁ L₂ L₃] [<answer>]
                              ^^^^^^^^^^^
                              4个可训练的latent embedding（k=4）
                              「思考空间」

师生分支：
  Teacher: [task] [<think>] [真实reasoning文本] [<answer>]
  Student: [task] [<think>] [4个learnable latents] [<answer>]
  
  两个分支同一组LoRA参数，joint forward，single optimizer step

Judgment向量：mean pool over {L₀, L₁, L₂, L₃, <answer>} 5个位置
            （之前是只pool 1个位置）

Label encoding：用同样的5-token结构，让label embedding和judgment活在同一representation space

4-Loss：
  L_cls_S  + α·L_cls_T  + β·L_align  + γ·L_distill
  (主任务)   (教师锚定)    (vec距离)   (logit分布)

Warmup：前200步只跑分类loss，再200步线性ramp β,γ到全开
```

K=4起步，warmup 200+ramp 200，α=0.5, β=1.0, γ=0.5，τ=0.07。

### 实施

14-task TDD plan（`03_plan.md`）。Subagent-driven execution，全部unit test通过：

```
src/
├── codi_tokenizer.py   <think>/<answer> 特殊token
├── codi_latents.py     k个可训练nn.Parameter
├── codi_model.py       Qwen3-4B + LoRA + specials + latents
├── codi_inputs.py      师生分支input构造 + pool
├── codi_label_cache.py label用同模板编码
├── codi_losses.py      4-loss + warmup
└── codi_data.py        Dataset + collator
```

---

## 7. Smoke v5结果：CODI training机制work，但crowding gate失败

### 训练（2000步，58分钟，13GB显存）

```
步数区间    avg_loss  acc_S   acc_T   align_gap
  0- 199    3.733    0.374   0.436   1.13 → 1.00
800- 999    2.010    0.639   0.835   ~0.40  ← teacher peak
1800-1999   2.694    0.647   0.772   ~0.40  ← teacher mild overfit
```

学生准确率plateau在0.65，**始终低于教师的0.83**。说明k=4 latents仍无法compress reasoning中的判别信号。

### Embedding crowding诊断

| 指标 | v4 (single-token) | v5 (CODI k=4) | 期望 | 结果 |
|---|---|---|---|---|
| Mean inter-label cosine | +0.401 | **+0.546** | < +0.40 | ❌ **更糟** |
| UPREG↔DOWNREG cosine | +0.936 | +0.900 | < +0.70 | ❌ 几乎没变 |
| Association ↔ no_relation | — | +0.909 | — | 新发现的alarming pair |

**结论：multi-token pool + learnable latents alone 没解决问题。**

### F1 evals缺失

容器迁移前没跑完 BioRED / BioTriplex / DrugProt full-RE 三个evaluation。命令保留在`04_smoke_v5_results.md` §4。

---

## 8. 已被证伪的假设

| 假设 | 来源 | 证伪手段 |
|---|---|---|
| H1: Silent CoT post-hoc蒸馏泛化到full-RE | Quest 022 | Full-RE eval：0.696 → 0.039 |
| H2: Multi-token pool + joint CODI解决crowding | Quest 023 v2 | v5 diagnostic：+0.401 → +0.546（更糟） |

---

## 9. 下一步候选（详见`05_next_steps.md`）

按改动成本从低到高：

| 方向 | 改动 | 预期 |
|---|---|---|
| **A. Per-label latents** | k×n_labels个latent，每个label有自己的latent slice | 避免4个共享latent collapse到同一mode |
| **B. Drop `<answer>` from pool** | 只pool 4个latent，去掉`<answer>` | 防止`<answer>`聚合主导（因为causal attention下它看到所有latent） |
| **C. Cosine-loss替代MSE for align** | L_align的scale从1e-4放大到1e-1 | 让align梯度真正起作用 |
| D. Coconut-style hidden state recursion | latent间循环喂回 | 真"连续推理链"，结构改造大 |
| E. Bidi attention on latent block | latent间双向attention | 配合flash_attn复杂 |
| F. Generative pivot | 抛弃cosine head，直接generate label name token | Method-level rewrite |

**推荐smoke v6 = A + B + C 组合**（都是非breaking changes，可一次性验证）。

数据层面备选：
- 重生成synth，要求Sonnet写200+ token长reasoning（当前只30 token）
- 在synth pipeline显式构造UPREG vs DOWNREG对比样本对+contrastive loss

---

## 10. 这个分支提供什么

```
quest_023_codi/
├── README.md          ← 入口
├── docs/
│   ├── 00_session_summary.md   ← 你正在读的这份
│   ├── 01_quest_023_brief_v1.md   ← v1方案（synth + bi-encoder）
│   ├── 02_quest_023_codi_design.md  ← CODI-Bi完整spec
│   ├── 03_quest_023_codi_plan.md   ← 14-task实施plan
│   ├── 04_smoke_v5_results.md    ← v5训练曲线 + diagnostic + 未跑完的eval命令
│   └── 05_next_steps.md          ← 6个改造路径
├── src/               ← 7个CODI模块（all unit tests pass）
├── tests/             ← 单元测试
├── scripts/           ← compute_mean / smoke_tiny / train_codi / eval_codi / embedding_diag_v5
├── artifacts/         ← 训练 & diagnostic log
└── cross_eval/        ← BioRED/BioTriplex/DrugProt full-RE的parse和eval脚本
```

**未committed**（在新容器中需重生成或rsync）：
- 11K合成训练数据（~20MB）
- v5 trained adapter（~1GB）

新容器恢复指南见 README.md。
