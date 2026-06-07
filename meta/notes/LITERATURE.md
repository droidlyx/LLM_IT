# BioRE 相关文献 — 按想法类别梳理

**整理时间**: 2026-05-28 (初版) / 2026-05-30 (新增 L/M/N/O 4 个方向,共 19 篇) / 2026-06-07 (round-3 再扩,新增 P/Q/R/S/T/U/V/W/X/Y/Z 11 个方向,共 30 篇)
**目的**: Ch 3 (BioRE) 相关文献按"研究想法"分组,方便对照本工作的定位与差异。
**信息来源**: deepxiv 文献调研 (2024-06 ~ 2026-05) + 用户提供 PDF (BioREx, REaMA)。

> **导航**: 实验结果见 [EXPERIMENTS.md](EXPERIMENTS.md)。整体方向规划见 [RESEARCH_NOTES.md](RESEARCH_NOTES.md)。

---

## 类别 A. 多数据集统一训练 / Schema harmonization

> 共同假设: 跨数据集训练能扩大语义覆盖 + 提升 OOD 泛化,但需解决 schema 异质性。

### A1. BioREx — Lai et al., arXiv:2306.11189 (2023-06, 21 citations)
- **路线**: PubMedBERT + 8 数据集 harmonize
- **5 个 harmonization tricks**:
  1. 实体类型规范化
  2. 关系类型映射到统一 schema
  3. `[Dataset]` token 注入提示
  4. 实体类型嵌入到 entity marker
  5. inverse-frequency loss reweight
- **关键数字** (评估口径要小心):
  - 论文报 0.796 是 **binary** (`to_binary=True` 默认 — 所有 rel type 折叠成 "Association")
  - 多类实测 BioRED ~0.60 / BC8 ~0.56 (用户实测,论文 p14 也有 56.2 / 61.5 数字)
- **代码**: github.com/ncbi/BioREx
- **与本工作差异**:
  - 它做 schema align 但**不质疑数据集内 label 一致性**
  - 评估口径过于宽松,论文 headline 数字 misleading
  - 我们已实现它的 trick 3+4+5,但发现 trick 5 (loss reweight) 与多数据集结合反而有害 — 见 [EXPERIMENTS.md §4](EXPERIMENTS.md#4-4-变体-ablation-2026-05-28)

### A2. REaMA — Zhang et al., IEEE TNNLS 2025 (Sichuan U)
- **路线**: LLaMA-2 7B/13B + IT on **REInstruct** (150K instructions, 8 数据集)
- **关键设计**:
  - 两种 instruction template: 命名实体描述 vs 仅类型
  - `@DiseaseSrc$...@/DiseaseSrc$` 边界标签 (类似我们的 `{i|name|type}` 但更冗长)
  - 11:1 None:non-None 不平衡 → V1 子集平衡采样
  - CoT 子集 (GPT-4 生成 reasoning) 显著提升 OOD 泛化
- **关键数字**:
  - BioRED 单测 ~0.68 (多类无 novelty 无 direction)
  - 7 数据集平均 81.28 (被小数据集拉高,headline 数字也有误导)
- **代码**: github.com/stzpp/REaMA
- **与本工作差异**:
  - 现成的 multi-dataset LLM-IFT 蓝本,但**没有 mechanism-aware labels**
  - 没有 `[Dataset]` 显式 token
  - 用 full FT 而非 LoRA
  - **我们 8B + LoRA + BioRED only 已经持平它的 13B + full FT + 8 数据集** — 说明扩规模/扩数据的边际收益已趋平

### A3. InstructUIE / UIE / USM (2022-2023)
- **通用 IE 路线**: 把 NER/RE/EE 统一成 instruction following
- **与本工作差异**: 通用方法,非 biomed-specific,且评估在 ACE/CoNLL 等通用 benchmark,与 BioRED 不可直接对照

### A4. Knowledge-augmented PLMs for BioRE — Wickramasinghe et al., arXiv:2505.00814 (2025-05, 4 citations)
- **路线**: 系统对比 **大小 PLM + 不同外部上下文** 在 BioRE 上的增益
  - 测试增益类型: entity descriptions, KG embeddings, molecular structures
  - 评估基线: PubMedBERT vs BioLinkBERT-Large 等
- **关键数字**: PubMedBERT + entity descriptions + max_seq=512 → 平均 F1 **89.2%** (跨多个 RE 数据集)。增益幅度大模型上仅 0.9-1.2pp
- **核心结论**: **大模型 (BioLinkBERT-Large) 已经隐式编码了 entity description / KG 信息**,外部知识注入对它增益有限;**小模型才显著受益**
- **与本工作差异**:
  - BERT 路线,非 LLM-IFT;但结论与我们发现一致:"8B + LoRA + BioRED only 持平 13B + full FT + 8 数据集" — 大模型靠规模隐式吸收外部知识
  - 它的结论也可解释为什么我们的 multi-dataset (B) 增益微小 — 8B 已经"够大",不需要再补 DrugProt/DDI 这种正交知识
- **方法启示**: 如果走 LLM-IFT 路线,不必再堆 entity desc / KG embedding (BiomedRAG 路线的简化版),收益不大

### A5. 我们的发现 (新)
- **多数据集 + loss reweight 互相抵触** (见 [EXPERIMENTS.md §4](EXPERIMENTS.md#4-4-变体-ablation-2026-05-28))
- 反频权按合并语料统计 → 扭曲 target dataset 自身的稀有类分布
- **可能的方法创新**: target-domain-aware reweighting

---

## 类别 B. LLM-IFT 路线代表作

> 与本工作直接对标,关注 LLM 在 BioRE 上的表现。

### B1. REaMA (同 A2)
见上。

### B2. R1-RE — arXiv:2507.04642 (2025-07, 9 citations)
"Cross-Domain Relation Extraction with RLVR"
- **路线**: RL with Verifiable Reward (GRPO) + "annotation guideline reasoning"
- **核心 thesis**: SFT 学到 shallow memorization;用 RL 强制模型按 annotation guideline 显式推理
- **关键数字**: OOD 上 +30pp over SFT,匹配 GPT-4o
- **与本工作差异**:
  - 它假设 label 对、模型不会推理
  - 我们的诊断显示 BioRED label 本身有 7+ 条暗规则 (见 [EXPERIMENTS.md §5](EXPERIMENTS.md#5-biored-schema-的-7-条暗规则))
  - **可作为 Ch 3 baseline 对比** — 同样 OOD generalization 目标但方法 path 不同
- **可能的整合**: 在我们的 mechanism-aware label 基础上加 RLVR,推理目标变成"用 mechanism tag 区分 association vs causation"

### B3. BioNCERE — arXiv:2410.23583 (2024-10)
- 非对比学习 + transfer learning for BioRE
- 不依赖 NER 标签
- **与本工作差异**: 训练方法可借鉴,但 BioRE 不主流

---

## 类别 C. 方向 / 角色增强

> 给关系加 subject/object 等结构信息。

### C1. BioREDirect — arXiv:2501.14079 (2025-01, 6 citations, NCBI 团队)
- **路线**: 给 BioRED 加 subject/object 方向标注,扩展到 10,864 pairs
- **方法**: multi-task (rel + novelty + entity role) + soft-prompt + context chunking
- **关键数字**: 跑赢 GPT-4 / Llama-3 (PEFT)
- **数据**: 现成的 bioredirect 文件就是这篇的产物,直接可用
- **与本工作差异**:
  - 它的 "directionality" 是角色 (subject/object)
  - **不是 commitment strength** (causal vs correlational)
  - 跟我们的 mechanism-aware 维度**正交**,可同时叠加

---

## 类别 D. 机制 / 不确定性信号 (Ch 3 Framing Y 的关键组件)

> 自动给关系打 causal/correlational/hedging 等元标签的工具。

### D1. Causal Language in Observational Studies — arXiv:2502.12159 (2025-02)
- **路线**: 80k observational study abstracts 上做 **causal vs correlational** 二分类
- **现成工具**: BioBERT-based classifier 已开源
- **仓库**: github.com/junwang4/causal-language-use-80k-observational-studies
- **本工作用法**: 给 BioRED 每个 gold relation 自动打 causal/correlational tag,作辅助监督
- **关键先决条件**: 需先验证它在 BioRED 上能区分 Pos_Corr/Neg_Corr (causal) vs Association (correlational) — 见 [RESEARCH_NOTES.md P1](RESEARCH_NOTES.md)

### D2. UnScientify — arXiv:2503.11376 (2025-03)
- **路线**: 规则 + pattern matching 检测 scientific uncertainty (research limitations, variability, ungeneralizable conclusions)
- **关键数字**: 0.808 accuracy,**比 LLM 更准**
- **本工作用法**: 跟 D1 互补 — D1 看 causal commitment,D2 看 hedging confidence

### D3. Uncertain Biomedical Knowledge — arXiv:2112.02570 (2021-12)
- Information Entropy + Uncertainty Rate 量化 SPO triple 不确定性
- 限定中国心血管文献
- **与 D1 D2 相近,但聚焦 KG 构建,没接 RE benchmark**

---

## 类别 E. 评估方法论 / Benchmark critique

> 不是新方法,而是质疑现有评估或基准。

### E1. Beyond the Numbers — arXiv:2411.05224 (2024-11, 2 citations)
"Transparency in Relation Extraction Benchmark Creation"
- **Position paper** (无具体实证数据)
- 批判 TACRED/NYT 的 transparency 不足 + class imbalance + noisy labels
- 呼吁 dataset datasheets + class-level analysis
- **与本工作关系**: 它给"empirical critique"提供模板,我们做的是 BioRED 实证版本 (见 [EXPERIMENTS.md §5](EXPERIMENTS.md#5-biored-schema-的-7-条暗规则)) — **它没批 BioRED**

### E2. LLM-as-judge in BioRE — arXiv:2506.00777 (2025-06, 10 citations)
- LLM-as-judge 在 BioRE 上**不可靠**
- structured output formatting 修复 +15%
- **与本工作关系**: 警示我们不能用 Sonnet 当 judge / teacher (已实证 Sonnet F1=0.43 << 8B F1=0.69)

### E3. Revisiting DocRED — arXiv:2205.12696 (2022)
- 重标 DocRED 解决 false negative
- **BioRED-Revisited 风格的范例**
- **不走这条路** — critique-only 撑不起博士主线

---

## 类别 F. Claim grounding (Ch 4 衔接)

> 验证 paper claim 是否被证据支持。GO-CAM 因果声明也需要这种 grounding。

### F1. SciClaims — arXiv:2503.18526 (2025-03)
- Llama3 7B + Elasticsearch claim 检索 + LLM 验证
- 全 pipeline 无 fine-tune
- **本工作用法**: Ch 4 GO-CAM 的 building block (claim grounding)

### F2. CER — arXiv:2509.13888 (2025-09)
"Combating Biomedical Misinformation through Multi-modal Claim Detection"
- 多模态 (text/web/video) claim 检测 + PubMed 检索 + LLM reasoning
- HealthFC / BioASQ-7 / SciFact SOTA
- **关联**: claim verification 与 GO-CAM 因果声明验证衔接

---

## 类别 G. 新基准 / Mention vs Concept-level RE

### G1. GUT-BRAINIE — arXiv:2602.04320 (2026-02, 最新)
"A Domain-Specific Curated Benchmark for E and Doc RE"
- 新做 gut-brain 领域 1,647 abstracts benchmark
- **关键贡献**: 引入 Mention-level RE (M-RE) vs Concept-level RE (C-RE) 区分
- **与本工作关系**: M-RE vs C-RE 区分可支撑"变异 granularity 不一致"问题 (见 [EXPERIMENTS.md §5](EXPERIMENTS.md#5-biored-schema-的-7-条暗规则) 暗规则 5)

---

## 类别 H. 预测时增强 / 通用 RE 技巧

### H1. Instance-Adapted Predicate Descriptions — arXiv:2503.17799 (2025-03)
- dual-encoder + 现成 predicate 模板
- +1-2.1% F1
- **可借鉴**: prompt engineering 思路,但增益较小

### H2. Zero-shot doc-level BioRE via scenario prompt — arXiv:2505.01077 (2025-05)
- ChemDisGene / CDR 上接近 fine-tune
- **与本工作关系**: zero-shot 路线,我们走 fine-tune 路线,作为对照

---

## 类别 J. RAG-Augmented BioRE

> 检索增强生成路线 — 推理时检索相关文档/示例注入 LLM。与本工作 (parametric fine-tune) 正交。

### J1. BiomedRAG — Li et al., arXiv:2405.00465 (2024-05, **73 citations**)
"A Retrieval Augmented Large Language Model for Biomedicine"
- **路线**: 在 LLaMA2 / GPT-4 上做 RAG,**不改 cross-attention**,直接把 chunk 拼到 prompt 里
  - 关键创新: **LLM-supervised chunk scorer** — 用 LLM 的 loss 监督 retriever (让 retriever 学到哪些 chunk 真的对下游有用)
  - chunk 数 m=5 最优
- **关键数字**:
  - ChemProt **88.83 micro-F1** (RE)
  - GIT **81.42 micro-F1** (triple extraction)
  - 4 tasks × 8 datasets, 跑赢 RT-5 / OneRel
- **与本工作差异**:
  - 推理时方法,与 fine-tune 正交,**理论上可叠加**到我们的 LoRA 模型上
  - 但 ChemProt 是 5-class 简单 schema,跟 BioRED 8-class + novelty 不可比;不能 cite "88.83 > 我们 0.65"
  - 它**不解决** OOD commitment calibration 问题 (我们诊断出来的 BC8 上 1582 个 directional→Association 错配)
- **何时考虑接入**: 若 Ch 3 主体方法定下后还有提升空间,可作为推理时 add-on (类似 BioREDirect + RAG)
- **代码**: 论文页面有

---

## 类别 K. 相关但非 Ch 3 直接路线

> LLM-as-encoder / embedding 等方向,与 Ch 1/2 VANER2 更相关。列在这里供未来工作引用。

### K1. LLM2Vec-Gen — arXiv:2603.10913 (2026-03, 最新)
"Generative Embeddings from Large Language Models"
- **路线**: 自监督训练 LLM-based text encoder 去编码 **模型潜在 response**,而非 input
  - 在 query 后加 trainable suffix tokens (thought / compression tokens)
  - 双目标: response reconstruction + embedding alignment
  - LLM backbone 冻结
- **关键数字**:
  - MTEB **+9.3%** over best unsupervised teacher (SOTA)
  - BRIGHT (reasoning) **+29.3%**
  - 安全性: harmful retrieval -43.2%
- **与本工作关系**:
  - **非 BioRE 路线**,通用 embedding
  - 跟 VANER2 (LLM-as-encoder for NER) 思路相通,可作为 Ch 1/2 后续工作 motivation
  - 对 Ch 3 BioRE 不直接相关 — 我们做的是 generative IFT 不是 embedding
- **未来可能**: 如果 Ch 4 GO-CAM 需要把 RE 三元组 embed 到图谱里,这套方法可用

---

# 新增 (2026-05-30): 基于 D variant prior-shift 诊断的新方向文献

> 新加的 4 个方向 (L/M/N/O) 都是 broad lit-scan 的命中,直接对应 [EXPERIMENTS.md](EXPERIMENTS.md) 的最新失败模式诊断 (prior shift / `[Dataset]` token 零迁移 / multi-task 冲突 / BC8 under-commit)。

## 类别 L. Prior shift / Logit adjustment / Long-tail 校准

> **跟本工作的关联**: 4-variant ablation 显示 reweight 是唯一稳定 OOD 增益;深度分析显示 cdr/pharmgkb 上 pred/gold 比例严重失配 (0.58 vs 4.73),是经典 prior shift 问题。这里的方法用来**显式校准 prior 而不是 empirical reweight**。

### L1. Prior2Posterior (P2P) — arXiv:2412.16540 (2024-12, 1 citation)
"Model Prior Correction for Long-Tailed Learning"
- **路线**: 后置校正,用模型 a posteriori 概率拟合 effective prior,而非 empirical class frequency
- **关键发现**: trained 模型的 effective prior ≠ training distribution prior,有残余 class bias
- **方法**: 训练完后用 unlabeled validation 估计 effective prior,从 logit 减去 log effective prior + log target prior
- **理论**: 在 cross-entropy 与 logit-adjusted loss 下证明 Bayes-optimal
- **关键数字**: CIFAR100-LT / ImageNet-LT / iNaturalist18 上 SOTA,无需重训
- **与本工作差异**:
  - 原工作是 CV / classifier,我们要适配 LLM-IFT 生成式范式 (first-token logit / sequence logprob)
  - 我们的 prior shift 更严重 (cdr 0.58, pharmgkb 4.73),P2P 几乎为我们这个场景而生
  - 可与现有 D variant 的 empirical reweight 互补 (训练时 reweight + 推理时 P2P)

### L2. Neural Prior Estimation (NPE) — arXiv:2602.17853 (2026-02)
"Learning Class Priors from Latent Representations"
- **路线**: 从 latent feature 学 class log-prior,不依赖 empirical count
- **方法**: Prior Estimation Module + 单向 logistic loss,在 Neural Collapse 假设下恢复 effective prior
- **应用**: NPE-LA = adaptive logit adjustment,动态校正 imbalance
- **理论**: 证明可恢复 log-prior up to additive constant
- **关键数字**: CIFAR-LT + semantic segmentation 一致增益,尤其小类
- **代码**: github.com/masoudya/neural-prior-estimator
- **与本工作差异**:
  - 比 P2P 更适合"无 target gold label"的场景 → 真正零样本适配陌生数据集
  - 但 LLM-IFT 没有自然的 classifier head,需要把 NPE 模块挂到 hidden state 上,工程量比 P2P 大

### L3. Class Confidence Aware Reweighting — arXiv:2601.15924 (2026-01)
- **路线**: 训练时按 class frequency × prediction confidence 双重调权 cross-entropy
- **核心**: 用 Ω(p_t, f_c) 函数抑制 head-class 高 confidence 样本梯度,放大 tail-class 低 confidence 样本梯度
- **关键性质**: 不改 logit / margin / inference,单阶段训练
- **关键数字**: CIFAR-100-LT, ImageNet-LT, iNaturalist 2018 上有效
- **与本工作差异**:
  - 训练时 reweight 我们已实现 (D 变体),但是 frequency-only
  - L3 引入 confidence 维度,可能解决"reweight ⊥ multi-dataset"问题 — 因为 confidence 是 per-sample 的,不被 cross-dataset 频率扭曲
  - 实施成本中等 (改 loss function)

## 类别 M. Multi-task gradient surgery / Task vector merging

> **跟本工作的关联**: 4-variant ablation 发现 B(multi-dataset) + C(multi+reweight) 都不如 D(reweight only) — 即 multi-dataset 与 reweight 在我们设置下负向交互。这里的方法用来**在 gradient 或 task-vector 层面消除冲突**,可能救回 multi-dataset 路线。

### M1. SAM-GS Gradient Similarity Surgery — arXiv:2506.06130 (2025-06, 4 citations)
"Gradient Similarity Surgery in Multi-Task Deep Learning"
- **路线**: similarity-aware gradient surgery,根据任务梯度相似度动态调整
- **方法**: 冲突场景下做 gradient equalisation,对齐场景下保留 first-order momentum
- **关键性质**: 同时处理 angle 与 magnitude 两种 gradient 冲突
- **与本工作差异**:
  - 我们的"reweight ⊥ multi-dataset"很可能就是 gradient 冲突的表现 (不同 dataset 的 None 频率天差地别)
  - SAM-GS 可替换 D 变体的 frequency-based reweight,在 B/C 类似的多数据集设置下重测
  - 可能是"救回多数据集"最直接的工具

### M2. Task Singular Vectors (TSV) — arXiv:2412.00081 (2024-12, **100 citations**)
"Reducing Task Interference in Model Merging"
- **路线**: 对 per-layer task matrix 做 SVD,发现低秩结构
- **TSV-Compress**: task matrix 缩 90% 仍保 99% 准确率
- **TSV-Merge**: 通过 SVD whitening 减少任务干扰,比现有方法 +15% accuracy
- **代码**: github.com/AntoAndGar/task_singular_vectors
- **与本工作差异**:
  - 适合"每个数据集训一个 LoRA adapter,然后 SVD 合并"的方案
  - 比 multi-task joint training 更灵活,可以选择性合并
  - 是 model merging 方向的强 baseline (100 citations)

### M3. Task Arithmetic in Trust Region — arXiv:2501.15065 (2025-01, 11 citations)
- **路线**: training-free model merging,通过 eigenvalue + thresholding 学 removal basis + binary mask
- **关键性质**: 不需要 retraining,只用 unlabeled data
- **与本工作差异**: 与 TSV 类似但用 trust region 框架处理 knowledge conflict,可作为 M2 的 alternative

## 类别 N. Schema novelty / Label space alignment

> **跟本工作的关联**: cdr 上 `[Dataset]=BC5CDR` 是训练时未见过的 tag,所有变体都难以输出正确的 CID label;这是典型的 schema novelty 问题。这里的方法处理 **训练 label vocab vs target label vocab 不一致**。

### N1. Open Set Label Shift (OSLS) — arXiv:2505.05868 (2025-05, 1 citation)
"Open Set Label Shift with Test Time OOD Reference"
- **路线**: 在不重训 classifier 的前提下,test-time 估计 source/target label distribution shift
- **方法**: EM 算法估计 target 域的 in-distribution label distribution 和 data ratio
- **代码**: github.com/ChangkunYe/OpenSetLabelShift
- **与本工作差异**:
  - 我们的 cdr/pharmgkb 失配本质上是 OSLS (target label set 与 source 部分重叠)
  - OSLS 提供理论框架估计 shift 量,可作为 L1/L2 的预处理

### N2. Training-Free Label Space Alignment (TLSA) — arXiv:2509.17452 (2025-09)
"for Universal Domain Adaptation"
- **路线**: 用 VLM (CLIP) 在 target 域发现并精炼私有 class,无需 fine-tune
- **方法**: 3 阶段过滤 — WordNet synonym 对齐 → CLIP embedding 语义对齐 → frequency-based noise 过滤
- **关键数字**: DomainBed +7.9% H-score, +6.1% H3-score
- **与本工作差异**:
  - CV 方法,但思路可借鉴: 用预训练 LM embedding 在 target dataset label vocab 上做 synonym mapping
  - 比如 cdr 的 "CID" 与 BioRED 的 "Positive_Correlation" / "Negative_Correlation" 可能在 embedding space 中可对齐

### N3. Few-shot Open RE with Gaussian Prototype (GPAM) — arXiv:2410.20320 (2024-10)
"and Adaptive Margin"
- **路线**: 解决 Few-shot RE with None-of-the-Above (NOTA) 边界问题
- **方法**: GMM 原型 + 自适应 margin + 对比学习
- **关键洞察**: NOTA 边界混淆 ≈ 我们的"什么时候保持 None / 什么时候降级为 Association"边界问题
- **与本工作差异**:
  - 思路: 把 "None" 和已知类做 prototype-level 区分,可能解决 BC8 上 1725 个 Positive_Correlation 漏标问题

## 类别 O. CoT mechanism / Data arrangement for OOD

> **跟本工作的关联**: REaMA 报告 CoT 子集 (GPT-4 reasoning) 显著提升 OOD;我们当前没用 CoT。这里探索"为什么 CoT 帮助 OOD" + "训练数据排列如何影响泛化"。

### O1. Unveiling Mechanisms of Explicit CoT Training — arXiv:2502.04667 (2025-02, 3 citations)
- **路线**: 信息论分析揭示 CoT training 形成 two-stage generalizing circuit
- **关键发现**:
  - CoT 在浅层解析中间推理,深层完成后续步骤,形成天然 OOD 泛化
  - non-CoT 模型在 OOD 上失败是因为没暴露 reasoning composition
- **关键数字**: 控制实验 + 真实数据集均验证 near-perfect OOD generalization
- **代码**: github.com/chen123CtrlS/T-CotMechanism
- **与本工作差异**:
  - 解释了为什么 REaMA 的 CoT 子集对 OOD 关键
  - 给"用 GPT-4 生成 CoT 数据训练 D 变体"提供理论依据,值得跑实验

### O2. Data Arrangement Affects Zero-Shot Generalization — arXiv:2406.11721 (2024-06, 3 citations)
"The Right Time Matters"
- **路线**: instruction tuning 中"何时见到什么数据"影响泛化
- **关键发现**:
  - 零样本泛化不是真正零样本,而是 instance-level 相似度驱动
  - 早期暴露相似 fine-grained 训练样本 → 更好泛化
- **方法**: Test-centric Multi-turn Arrangement (TMA),按 test 相似度组织训练数据
- **代码**: github.com/thunlp/Dynamics-of-Zero-Shot-Generalization
- **与本工作差异**: 解释为什么 multi-dataset 朴素混合伤害 OOD — 数据排列顺序与 test 相似度未对齐

## 类别 H 补充: 通用 RE / 新作

### H3. CPTuning — arXiv:2501.02196 (2025-01)
"Contrastive Prompt Tuning for Generative RE"
- **路线**: 把 RE reformulate 为 Seq2Seq text-infilling,处理 entity pair overlap (EPO)
- **方法**: 对比学习 + Trie-constrained decoding + label smoothing
- **关键洞察**: 不分配 one-hot label,而是 above/below threshold 的概率质量
- **与本工作差异**: 我们已用受限输出,但 Trie-constrained decoding 可能进一步压制 hallucination

### H4. RAG-RE Fine-Tuned — arXiv:2406.14745 (2024-06, 7 citations)
"Relation Extraction with Fine-Tuned LLMs in RAG"
- **路线**: 微调 LLM + RAG,同时解决 implicit relation + 检索增强
- **关键数字**: TACRED / TACREV / Re-TACRED / SemEval 上 SOTA
- **代码**: github.com/sefeoglu/RAG4RE-extension
- **与本工作差异**: 与开题报告 RAG 路线吻合,可作为推理时增强的现成范例

### H5. GLiDRE — arXiv:2508.00757 (2025-08, 2 citations)
"Generalist Lightweight model for Document-level RE"
- **路线**: 轻量级 zero-shot 文档级 RE,dual-encoder + localized context pooling
- **关键数字**: Re-DocRED / FREDo / Re-FREDo 上 SOTA (few-shot + low-resource)
- **代码**: github.com/robinarmingaud/glidre
- **与本工作差异**: 不需要 pre-defined entity pair,与我们 pair-enumeration 范式不同,可作为对比

### H6. Few-Shot LLM for RE Domain Adaptation — arXiv:2408.02377 (2024-08, 2 citations)
- **路线**: LLM 用 schema-constrained example prompt 生成 in-domain 训练数据,训 SpERT-style classifier
- **AECO domain** 上验证,有效的 domain adaptation 路线
- **与本工作差异**: 数据合成 + 小 classifier,与 Q022/Q023 蒸馏路线相近但目标不同

## 类别 A 补充: Biomedical multi-task LLM-IT

### A6. BioMistral-NLU — arXiv:2410.18955 (2024-10, 7 citations)
"Towards More Generalizable Medical Language Understanding through Instruction Tuning"
- **路线**: 统一 prompt 框架 + 7 medical NLU task 的 instruction tuning
- **MNLU-Instruct** 数据集 (开源 medical 语料)
- **关键发现**: task diversity (非数据量) 是 generalization 的关键
- **关键数字**: BLUE / BLURB 上跑赢原 BioMistral + ChatGPT / GPT-4 (zero-shot)
- **代码**: github.com/uw-bionlp/BioMistral-NLU
- **与本工作差异**:
  - 它是 multi-task IT (NER+QA+IE+...),我们专注 RE
  - "task diversity > data volume" 的发现支持我们走深 RE 一条而非广 IT 多条

### A7. Balanced Fine-Tuning (BFT) for Biomedical LLM — arXiv:2511.21075 (2025-11, 1 citation,腾讯 AI Lab)
"Aligning LLMs with Biomedical Knowledge using Balanced Fine-Tuning"
- **路线**: token-level + sample-level adaptive weighting
- **token-level**: loss 按 prediction probability rescale,防止 sparse low-confidence 数据过拟合
- **sample-level**: "minimum group confidence" 识别困难样本增强学习
- **关键数字**: 数学推理上匹敌 RL 方法,生物过程推理超越 GeneAgent
- **代码**: github.com/TencentAILabHealthcare/BFT
- **与本工作差异**:
  - **几乎为我们这个场景定制**: 是 LLM-IFT + biomed + 平衡重训
  - 我们的 D 变体只用 frequency reweight,BFT 加 token + sample 维度
  - 实施成本: 改 loss function,与我们框架兼容,可作为 D 变体的升级版

## 类别 D 补充: Abstention / Selective Prediction (对应 BC8 under-commit)

### D4. AbstentionBench — arXiv:2506.09038 (2025-06, **66 citations**, Meta)
"Reasoning LLMs Fail on Unanswerable Questions"
- **路线**: 系统揭示现代 LLM 在 unanswerable / underspecified 查询上的弃权失败
- **关键发现**:
  - reasoning post-training 平均**减少 24% abstention** — 反直觉地让模型更过度自信
  - 即便在 math/science 域也成立
- **数据**: 20 datasets benchmark
- **代码**: github.com/facebookresearch/AbstentionBench
- **与本工作差异**:
  - 给我们 BC8 上 under-commit 现象提供 narrative 框架 (model 不善于在不确定时弃权或降级)
  - 不是直接方法,但适合做 framing / 引言

### D5. Uncertainty-Driven Reliability (PhD thesis) — arXiv:2508.07556 (2025-08, 2 citations)
- **路线**: post-hoc selective prediction,利用训练动态 (intermediate checkpoint 之间的预测分歧) 作为不确定性信号
- **关键性质**: 不改架构 / loss,适用 classification / regression / time series
- **代码**: github.com/cleverhans-lab/selective-classification
- **与本工作差异**: 与 L1 P2P 类似的"训练完之后再校准"思路,可作为 prior-shift 之外的另一层

---

## 类别 P. 概念统一 / Doc-level BioRE 最新方法 (2024-2026)

> 直接对应我们的 BioRE 设定,可作为对照或借鉴。

### P1. ADRCM — arXiv:2501.05155 (2025-01, 3 citations)
"Biomedical RE via Adaptive Document-Relation Cross-Mapping and Concept Unique Identifier"
- **路线**: 三件套 — Doc-Relation Cross-Mapping fine-tune + CUI RAG + Iteration-of-REsummary 合成数据
- **核心点**:
  - **CUI RAG**: 用 UMLS Concept Unique Identifier 作 entity index → 跨数据集别名归一化 (`MESH:D000077152` ≡ `MESH:D000088'2` 等)
  - **ADRCM**: 把 doc 和 relation 互相 attention,处理跨句推理
  - **IoRs**: LLM 迭代生成 relation-focused summary 当合成训练样本
- **与本工作差异**:
  - 我们的 multi-dataset 失败原因之一是 entity ID 不跨数据集对齐 — **CUI RAG 直接对应这个问题**
  - 工程量大 (要接 UMLS API + RAG 检索)
  - 可作 Ch 3 主方法之一: 用 CUI 桥接 cdr/disgenet/pharmgkb 的实体空间

### P2. AI-assisted Knowledge Discovery — arXiv:2412.08900 (2024-12, 2 citations)
- **路线**: 实证对比 PubTator 3.0 / BioBERT / LLMs 在 BioRED 上的 NER+RE
- **关键数字**: BioBERT F1=0.79 (RE) > 多个 LLM (full-text 上),BioRED 上 BERT 路线仍领先
- **本工作用法**: 旁证"LLM-IFT 路线 0.65-0.69 离 BERT SOTA ~0.79 还有约 10 pt 差距"的结论

### P3. LMRC — arXiv:2408.13889 (2024-08, **12 citations**)
"LLM with Relation Classifier for Document-Level Relation Extraction"
- **路线**: 两阶段 — pre-classifier 先过滤 NA 实体对,再送 LLM 分类
- **核心发现**: doc-level RE 中 NA 实体对压倒性多数,会**稀释 LLM 注意力**导致 recall 下降
- **关键数字**: DocRED / Re-DocRED 上显著提升,接近 BERT SOTA
- **代码**: github.com/wisper12933/LMRC
- **与本工作差异**:
  - 我们目前是一次性把全部对放进 prompt,LLM 直接判 None vs 关系 — 跟 LMRC 诊断的"NA 稀释"问题完全对应
  - **可直接借鉴**: 第一阶段用一个轻量分类器 (BERT or LoRA 小模型) 过滤明显 None 的 pair,第二阶段才上 LLM
  - 与我们 D variant 的 reweight 思路互补 (D 是事中,LMRC 是事前)

---

## 类别 Q. Ontology / KG / KB-grounded biomed IE

> 共同想法: 用结构化 biomedical 知识 (UMLS / Biolink / MeSH) 约束或增强 LLM extraction。

### Q1. RELATE — arXiv:2509.19057 (2025-09, 1 citation)
"Relation Extraction in Biomedical Abstracts with LLMs and Ontology Constraints"
- **路线**: 三段 pipeline — ontology embedding preprocess + SapBERT similarity retrieval + LLM rerank with explicit negation
- **关键数字**: ChemProt exact-match **52%**,HEAL 真实摘要 rejection 率 0.4%
- **代码**: github.com/RENCI-NER/pred-mapping
- **与本工作差异**:
  - **直接对应 `[Dataset]` token 零迁移问题**: RELATE 把 LLM 输出 map 到 Biolink/ChemProt 标准 schema,跨数据集自动归一
  - 工程量中等 (要接 Biolink + SapBERT)
  - **Phase 2 实验候选**: post-hoc 加 ontology mapping 层,把陌生 dataset tag 自动 map 到 BioRED 8 类

### Q2. Generalized KG-enhanced biomed framework — arXiv:2408.06618 (2024-08, 2 citations)
- **路线**: 分离 **General Knowledge (GK)** + **Task-Specific Knowledge (SK)** 两个 graph,GK 用 BioBERT 编码 external KG,SK 从输入文本动态构建,二者融合
- **关键性质**: GK 可跨任务复用,SK 任务自适应
- **代码**: github.com/mpnguyen2/bio_kg_nlp
- **与本工作差异**: 思路类似 BiomedRAG 但 KG-flavored,主要 baseline 是 BioRelEx / ADE,**没在 BioRED 上验证**

### Q3. KB-Guided Generation — arXiv:2407.10021 (2024-07, **8 citations**)
"Document-level Clinical Entity and Relation Extraction via KB-Guided Generation"
- **路线**: MetaMap 抽 UMLS concept → 过滤药理学物质 → 注入 dynamic prompt → GPT 生成
- **关键发现**: 比 baseline few-shot 和标准 RAG 都强,**精度概念 grounding** 解决 GPT 在药品缩写/品牌名上的识别问题
- **与本工作差异**:
  - 走 prompt-engineering 路线,无需 fine-tune,与我们 LoRA 路线互补
  - **跨数据集对照启示**: pharmgkb/disgenet 的实体也是药/基因,可借用 MetaMap 工程做实体规范化

### Q4. Continual Pretrain vs GraphRAG — arXiv:2604.16422 (2026-04, 0 citations)
"Injecting Structured Biomedical Knowledge into LMs"
- **路线**: 用 UMLS 3.4M concept / 34.2M relation 构 100M-token corpus,对比 (a) continual pretrain BERT/BioBERT (b) GraphRAG LLaMA-3-8B
- **关键发现**:
  - BERT continual pretrain 涨幅大,BioBERT 已饱和涨幅小
  - GraphRAG 直接给 LLaMA-3-8B 加 +3 到 +5 pp,**无需重训**,且可多跳推理
- **代码**: github.com/jaaferklila/UMLS_knowledge_graph
- **与本工作差异**: 强证据表明 LLM 路线 RAG > continual pretrain (避免 catastrophic forgetting),为我们 RAG 加挡 (J1 BiomedRAG) 提供更新支持

---

## 类别 R. Conformal prediction / Set-valued / Adaptive abstention (LLM 通用)

> 直接对应 D4 AbstentionBench 揭示的 "现代 LLM 不会弃权" 问题,提供可证明 coverage 的方法。

### R1. TECP — arXiv:2509.00461 (2025-08, 2 citations)
"Token-Entropy Conformal Prediction for LLMs"
- **路线**: token-level entropy 作为 nonconformity score,split conformal prediction
- **核心性质**: **black-box**,不需要 white-box logit / 标注数据,直接从生成文本算 entropy
- **关键数字**: 6 LLM × 2 benchmark,coverage 全过,prediction set 比 prior self-uncertainty 方法更紧
- **与本工作差异**:
  - **Phase 1 可直接用**: 我们生成 None / 关系 token,直接算其 entropy → CP 校准弃权阈值
  - 几乎零工程量 (一个 calibration set 即可)

### R2. Set-Valued LLM Prediction — arXiv:2603.22966 (2026-03, no cites)
"Set-Valued Prediction with Feasibility-Aware Coverage Guarantees"
- **路线**: Learn-Then-Test (LTT) 校准,显式建模 **Minimum Achievable Risk Level (MRL)** — 当有限采样下 valid response 可能根本没生成出来
- **关键性质**: 当目标 risk > MRL 时给出有效 set,否则诚实标记 infeasible
- **与本工作差异**: 比 R1 更严谨地处理"LLM 可能压根没采到正确关系"的情况,但实现复杂

### R3. Conformal Abstention Policies — arXiv:2502.06884 (2025-02, **14 citations**)
- **路线**: RL 学动态阈值,把弃权 threshold 作为 action
- **关键数字**: hallucination 检测 +22%,calibration error -70~85%,coverage 维持 90%
- **代码**: github.com/sinatayebati/vlm-uncertainty
- **与本工作差异**: 比 R1/R2 更强但需 RL 训练流程,Phase 2 候选

### R4. Knowing When to Quit — arXiv:2604.18419 (2026-04, no cites)
"Dynamic Abstention in LLM Reasoning"
- **路线**: 把弃权当 KL-regularized RL 的一个 action,**生成过程中**就可以弃权
- **关键证明**: 动态 value-thresholding 严格 dominates 不弃权与固定位置弃权
- **与本工作差异**: 主要针对 reasoning,RE 上动态弃权意义不大 (我们生成短),但理论可借

### R5. Conformal Long-Tail — arXiv:2507.06867 (2025-07, 3 citations)
"Conformal Prediction for Long-Tailed Classification"
- **路线**:
  - **PAS (Prevalence-Adjusted Softmax)**: 用 class prevalence 校正 softmax,瞄准 **macro-coverage**
  - **FUZZY CP**: classwise / standard CP 的加权插值,rare class 也能紧凑
- **关键数字**: Pl@ntNet / iNaturalist 上 coverage-size 帕累托前沿改善
- **代码**: github.com/tiffanyding/long-tail-conformal
- **与本工作差异**:
  - **几乎完美对应**: 我们 BioRED 8 类天然长尾,Cause / Conversion / Drug_Interaction 都是 rare class
  - **Phase 1 候选**: 把 D variant 输出过 PAS 校准,看 rare-class recall 是否上升

---

## 类别 S. LoRA 几何 / 遗忘 / 合并 (我们用的就是 LoRA)

### S1. Subspace Geometry of LoRA Forgetting — arXiv:2603.02224 (2026-02, 3 citations)
"Subspace Geometry Governs Catastrophic Forgetting in LoRA"
- **核心定律**: forgetting ∝ **(1 − cos²θ_min)**,θ_min 是任务梯度子空间间最小主角
- **关键启示**:
  - 任务接近正交时,**LoRA rank 几乎不影响 forgetting** — 调 rank 没用
  - 任务相似时 rank 才有作用
- **与本工作差异**:
  - 解释 B variant 在 BC8 上没明显下降但 cdr 上 recall 大跌 — 因为 cdr 与 BioRED 在梯度子空间上**夹角小** (都是 CID 同义关系),冲突更严重
  - 不是直接方法,但是**诊断工具** — 可量化 B 失败原因

### S2. ALoRA — arXiv:2509.25414 (2025-09, 1 citation)
"Rethinking Parameter Sharing for LLM Fine-Tuning with Multiple LoRAs"
- **路线**: 多 LoRA 共享 **B** 矩阵 (传统共享 A),证明共享 A 只是初始化巧合,共享 B 才编码跨任务通用知识
- **代码**: github.com/OptMN-Lab/ALoRA
- **与本工作差异**: 若走"每数据集一个 LoRA"路线,ALoRA 是直接 baseline

### S3. Pico — arXiv:2604.16826 (2026-04, no cites)
"Crowded in B-Space: Calibrating Shared Directions for LoRA Merging"
- **路线**: 合并多 LoRA 时,B 矩阵的 over-represented direction 会过度对齐 → 识别并降权再合并
- **关键数字**: 3.4-8.3 pt 提升,**无需数据 / 重训**
- **与本工作差异**: 与 M2/M3 是同一脉络的 LoRA 专版,合并 per-dataset LoRA 候选方案

---

## 类别 T. MoE / Multi-domain expert routing

### T1. DES-MoE — arXiv:2509.16882 (2025-09, **5 citations**)
"Dynamic Expert Specialization: Forgetting-Free Multi-Domain MoE Adaptation"
- **路线**: adaptive router + 实时 expert-domain 关联追踪 + 三段 progressive specialization 训练
- **关键数字**: **-89% forgetting**,-68% 收敛时间,与单域 SFT 持平
- **与本工作差异**: 走 MoE 路线需改架构,长线候选;对 multi-corpus 训练失败是直接的"另一种解法"

### T2. MedINST — arXiv:2410.13458 (2024-10, **7 citations**)
"Meta Dataset of Biomedical Instructions"
- **数据**: 133 NLP tasks,7M+ samples,**MedINST32** 跨任务 benchmark
- **代码**: github.com/aialt/MedINST
- **与本工作差异**:
  - 作为**预训练混料**: 在 LLM-IFT 前先在 MedINST 上预训得到 biomed-instructable LLM,再 LoRA 上 BioRED
  - 对应 A6 BioMistral-NLU 的"task diversity > data volume"假设

### T3. Dynamic Data Mixing — arXiv:2406.11256 (2024-06, **19 citations**)
"Dynamic Data Mixing Maximizes IT for Mixture-of-Experts"
- **路线**: 用 token routing 的 **gate-load L2 distance** 衡量 dataset 差异,动态调整采样权重
- **代码**: github.com/Spico197/MoE-SFT
- **与本工作差异**:
  - 给我们 B variant 失败提供新解法: 不再 uniform mix BioRED+cdr+disgenet+pharmgkb,而是动态权重
  - 但需要 MoE 架构 (Qwen3-8B-Base 是 dense,不直接适配),工程改造成本中等

---

## 类别 U. 对抗训练 / robustness (biomed IE-specific)

### U1. RanAT4BIE — arXiv:2509.11191 (2025-09, 1 citation)
"Random Adversarial Training for Biomedical Information Extraction"
- **路线**: PubMedBERT + Bernoulli 随机采样 adversarial perturbation (而非每 batch 全员加扰动)
- **关键性质**: 计算开销显著降低,**robustness 不退化**;biomed 域优势明显 (与 vision 上 adversarial 退化的现象对比)
- **关键数字**: BioNER + BioRE 上 baseline 之上提升
- **与本工作差异**:
  - 我们 LoRA + 8B,可借鉴 Bernoulli 随机扰动思路 (PGD-lite)
  - 对解决 cross-corpus 标签噪声有帮助
  - Phase 2 工程量适中

---

## 类别 V. Counterfactual / 一致性 RE

### V1. CovEReD — arXiv:2407.06699 (2024-07, 5 citations)
"Consistent Document-Level RE via Counterfactuals"
- **路线**: 替换 Re-DocRED 中实体,生成 plausible 但 factually 错的 counterfactual doc,保留 relation map / context embedding 相似性
- **核心发现**: 真实数据训练的 RE 模型**依赖 entity bias 不依赖 context reasoning**;加 counterfactual 训练显著降低对实体的虚假依赖
- **代码**: github.com/amodaresi/CovEReD
- **与本工作差异**:
  - 直接命中我们 BC8 上 1725 Positive_Correlation 被漏 / 1582 directional 折叠成 Association 的失败模式 — 模型在用 entity prior 而非 context evidence
  - **Phase 2 强候选**: 在 BioRED 上做 entity-replacement DA,看 BC8 directional / mechanism 类是否回升

### V2. CIT-CRE — arXiv:2508.12031 (2025-08, 2 citations)
"Learning Wisdom from Errors: LLM Continual Relation Learning"
- **路线**: 按 correctness 把训练 / memory 拆 easy / hard;hard 样本配 analytical / reason-based prompt 引导
- **关键数字**: TACRED / FewRel 上 SOTA
- **与本工作差异**:
  - 我们已有的 error analysis (`8b_dev_error_analysis.txt` / `bc8_error_analysis.txt`) 可直接当 hard memory
  - **Phase 2**: 加一轮 error-guided IT,把 1725 Positive_Correlation miss 当作 reason-based prompt 训练样本

---

## 类别 W. Schema-aware 结构化生成

### W1. PARSE — arXiv:2510.08623 (2025-10, 4 citations)
"LLM Driven Schema Optimization for Reliable Entity Extraction"
- **路线**: 把 JSON schema 当 "natural language contract",LLM 自己 reflectively 优化 schema (ARCHITECT) + 解码时 hybrid guardrail (SCOPE)
- **关键数字**: SWDE +64.7%,first-retry error -92%
- **与本工作差异**: 主要做 entity extraction 不是 RE,但 schema co-optimization 思路可借给 ontology mapping 模块 (与 Q1 RELATE 互补)

---

## 类别 X. Data selection / mixture for IT

### X1. GradFiltering — arXiv:2601.13697 (2026-01, no cites)
"Uncertainty-Aware Gradient SNR Data Selection for IT"
- **路线**: frozen GPT-2 + LoRA ensemble 算 per-sample gradient → G-SNR (gradient drop + ensemble variance)
- **核心性质**: objective-agnostic,无需 task signal
- **与本工作差异**: 可用于挑选高价值 BioRED 子集 (与 dataset diversity 配合用),但 GPT-2 proxy 略弱,需验证

---

## 类别 Y. Uncertainty / introspective IE

### Y1. LLM Uncertainty Survey — arXiv:2503.00172 (2025-02, **44 citations**)
- **核心 taxonomy**: aleatoric vs epistemic,consistency-based + semantic clustering
- **与本工作差异**: reference 综述,做 framing / 引言用

### Y2. APIE Reflect-then-Learn — arXiv:2508.10036 (2025-08, no cites)
"Active Prompting for IE Guided by Introspective Confusion"
- **路线**: 双层 introspective 信号 — **format uncertainty** (parsing fail / generation variance) + **content uncertainty** (semantic divergence across samples)
- **关键性质**: 无需训练 / 标注
- **与本工作差异**: zero-shot ICL 设定,可作 prompt-engineering baseline

---

## 类别 Z. RE 数据增强 / 合成数据告诫

### Z1. Hallucination in Synthetic RE — arXiv:2410.08393 (2024-10, **8 citations**)
"Effects of Hallucinations in Synthetic Training Data for RE"
- **关键发现**: 相关 hallucination 让 RE recall 掉 **19.1-39.2%**,**无关** hallucination 影响微小;hallucination 检测可 F1 0.92
- **代码**: github.com/BigPanda042/Relation-Extraction-Hallucination-Study
- **本工作应用**: 严重警示 — 我们若用 LLM (Sonnet) 合成数据,必须先检测 hallucination,否则 recall 直接崩

### Z2. Self-Prompting Zero-shot RE — arXiv:2410.01154 (2024-10, **18 citations**)
- **路线**: relation synonym expansion + entity filter + sentence rephrase → 三段合成 in-context demo
- **关键数字**: 优于既有 LLM zero-shot RE
- **与本工作差异**: zero-shot 路线 (无 fine-tune),与我们正交

---

## 类别 I. 已被否定的关联想法 (避免重复)

| 想法 | 否定原因 | 来源 |
|---|---|---|
| Sonnet → 8B distillation | Sonnet F1=0.43 < 8B F1=0.69,数学上不可能提升 | [EXPERIMENTS.md §2](EXPERIMENTS.md#2-sonnet-teacher-quality-诊断) |
| Naive multi-dataset union | path bug 修复后实测 in-distribution 微涨 OOD 持平,不足以撑论文 | [EXPERIMENTS.md §4](EXPERIMENTS.md#4-4-变体-ablation-2026-05-28) |
| 27-label unified synth (Q023 v1) | 教师质量 39% FP,real F1 = 0.523 | [BRANCH_INDEX.md](BRANCH_INDEX.md) |
| CODI-Bi multi-token latents (Q023 v2) | embedding crowding 反而 worse | [BRANCH_INDEX.md](BRANCH_INDEX.md) |
| RAG-only 路线 (作为主方法) | 文献已饱和;且 RAG 不解决 OOD commitment calibration 问题 | F1, F2, **J1** |
| BioRED-Revisited 路线 | critique-only,博士不够撑 3-4 章 | E3 |

---

## 整体 landscape 表 (条件性更新)

| 工作 | 路线 | BioRED 多类 | BC8 多类 | 本工作关系 |
|---|---|---|---|---|
| **BioREx** | BERT + 8-data harmonize | **0.60** | **0.56** | OOD 鲁棒性 reference |
| **REaMA** | LLaMA-2 IFT + 8 数据集 | ~0.68 | 未报 | LLM-IFT 路线,已持平 |
| **我们 8B baseline (A)** | Qwen3-8B LoRA + BioRED only | **0.6367** | **0.5589** | in-dist 微领先,OOD 接近 BioREx |
| **我们 8B +reweight (D)** | +inverse-freq loss | **0.6489** | **0.5646** | 单一最优变体 |
| BioREDirect | BERT + subject/object | (不同口径) | — | 正交维度,可叠加 |
| R1-RE | RL teach guidelines | (通用,非 biomed) | OOD 同目标,可对比 |
| InstructUIE / UIE | 统一 IE 多任务 | (通用) | 通用方法 |
| BioNCERE | 非对比学习 | (非主流) | 训练方法可借鉴 |
| Causal Language | causal classifier 工具 | — | **Ch 3 Framing Y 组件** |
| UnScientify | uncertainty 检测工具 | — | **Ch 3 Framing Y 组件** |
| GUT-BRAINIE | M-RE vs C-RE | (新 benchmark) | motivation 引用 |
| Beyond the Numbers | RE benchmark critique | — | framing 模板 |
| LLM-as-judge BioRE | judge 不可靠 | — | 警示 (我们已绕开) |
| **KAPLM-BioRE** | BERT + 知识增强 | ChemProt 89.2 avg | "大模型隐式吸收外部知识" — 印证我们 8B≈13B |
| **BiomedRAG** | LLaMA2/GPT-4 + RAG | ChemProt 88.83 | 推理时正交增强,可叠加 |
| LLM2Vec-Gen | LLM-as-encoder | (非 BioRE) | 与 VANER2 关联,非 Ch 3 路线 |
| **P2P (L1)** | post-hoc prior correction | (CV benchmark) | **直接对应 prior shift,零训练成本** |
| **NPE (L2)** | latent-based prior estimation | (CV benchmark) | 零样本适配陌生数据集 |
| **BFT (A7)** | biomed LLM-IFT 平衡微调 | (LLM benchmark) | **几乎为我们场景定制,可替换 D 的 reweight** |
| **SAM-GS (M1)** | gradient surgery for MTL | (MTL benchmark) | 可能救回 multi-dataset 路线 |
| **TSV (M2)** | task vector SVD merge | (CV benchmark) | per-dataset LoRA 合并方案 |
| **AbstentionBench (D4)** | LLM 弃权失败诊断 | (20 datasets) | BC8 under-commit 的 narrative 框架 |
| **CoT mechanism (O1)** | 解释 CoT 为何助 OOD | (合成 + 真实) | 给"+CoT 训 D"提供理论 |
| **ADRCM + CUI RAG (P1)** | doc-relation cross-map + UMLS RAG | (BioRED-style) | **跨数据集实体空间对齐的直接答案** |
| **LMRC (P3)** | pre-classifier + LLM 两阶段 | DocRED 接近 SOTA | **直接解决"None 稀释 LLM 注意力"** |
| **RELATE (Q1)** | LLM + ontology constraint | ChemProt 52 EM | post-hoc ontology mapping,解 `[Dataset]` 零迁移 |
| **TECP (R1)** | token-entropy CP | (NLG benchmark) | **零工程量弃权机制** (Phase 1 候选) |
| **Conformal Long-Tail (R5)** | PAS + FUZZY CP | iNaturalist | **直接对应长尾 BioRED 关系类型** |
| **Subspace Geometry LoRA (S1)** | 1−cos²θ_min 遗忘律 | (continual learning) | 量化 B variant 失败,无需改方法 |
| **CovEReD (V1)** | entity-replacement CF | Re-DocRED 一致性升 | 直击 BC8 entity-bias 漏标 |
| **DES-MoE (T1)** | dynamic expert MoE | -89% forgetting | multi-corpus 失败的强候选解法 |
| **Dynamic Data Mixing (T3)** | gate-load 自适应权重 | LLaMA + MoE | B variant 平衡问题的新路 |
| **RanAT4BIE (U1)** | Bernoulli 随机扰动 PGD | BioNER/RE 涨 | biomed 域 robustness,工程量适中 |
| **Hallucination in Synth RE (Z1)** | LLM 合成数据告诫 | recall -19~39% | 严重警示走 distillation 路 |

---

## 关键判断 (更新 2026-05-30)

1. **LLM-IFT 路线 BioRED 多类天花板 ~0.65-0.69**,REaMA 没真正突破
2. **BioREx 0.796 是 binary 评估优势**,不是"LLM-IFT 还差 10 个点的目标"
3. **真正的 LLM-IFT 上升空间**: 0.69 → 0.79+ (追平 BERT 路线 5-pair binary SOTA),约 10 个点 — 但要先想清楚是否值得追,因为 BERT/LLM 在不同 scope 下不可直接比
4. **没人系统对比** LLM-IFT vs BERT 在 BioRE 上的能力上限差异原因
5. **没人把 causal classifier (D1) 接进 RE pipeline**
6. **没人把 commitment-strength 维度作为 BioRE 的辅助标签**
7. **没人发现** multi-dataset 与 loss reweight 的负向交互 — 本工作的新发现 (见 [EXPERIMENTS.md §4](EXPERIMENTS.md#4-4-变体-ablation-2026-05-28))
8. **新发现 (2026-05-30)**: D variant 上 pred/gold 比例在 cdr (0.58) / pharmgkb (4.73) 严重失配 — **是经典的 prior shift 问题**,empirical reweight 只是机械拉动方向,不解决根本。**P2P/NPE/BFT 直接对应**,L1+L2+A7 是当前最高优先级的方法学候选 (见 [EXPERIMENTS.md §9](EXPERIMENTS.md#9))
9. **新发现 (2026-05-30)**: `[Dataset]` token 训练对**陌生** dataset tag 零迁移 (cdr/disgenet/pharmgkb 都是训练时未见 tag),且多数据集训练让模型对陌生 tag 更保守 (B 在 cdr 上 recall 0.04) — schema novelty 是 OOD 主要瓶颈,N1/N2/N3 提供可借鉴框架
10. **新发现 (2026-06-07)**: round-3 lit-scan 揭示三条更新路线
    - **CUI/Ontology 桥接**: P1 ADRCM + Q1 RELATE 都把跨数据集实体/关系 schema 用 UMLS / Biolink 标准化 — **正面解决我们 multi-dataset 标签空间不一致问题**,比单纯 reweight 更根本
    - **Conformal coverage 弃权**: R1 TECP / R5 Conformal Long-Tail 提供可证明 coverage 的 post-hoc 弃权机制,几乎零工程量,可立即接入
    - **CovEReD entity-counterfactual**: V1 揭示 RE 模型本质依赖 entity prior 而非 context — 恰好对应我们 BC8 上 1725 Positive_Correlation 漏标的失败模式,DA 路线候选

---

## 方法尝试优先级排序 (基于 2026-06-07 文献 + 失败分析)

> 综合考量: **(1) 与失败模式匹配度** × **(2) 工程量** × **(3) 论文 contribution 强度** × **(4) 风险**

### Tier 1: 立即可试 (零~低工程量,直接对应已识别失败模式)

| # | 方法 | 来源 | 目标改进 | 工程量 | 风险 |
|---|---|---|---|---|---|
| 1 | **Post-hoc logit adjustment (oracle prior)** | L1 P2P | cdr 0.42→0.55+,pharmgkb 0.26→0.40+ | 0.5 天 (解码层加 logit bias) | 低 (oracle 验证) |
| 2 | **TECP token-entropy conformal** | R1 | BC8 弃权 + 不掉 F1 (报"我不知道"换 precision) | 0.5 天 (calibration set) | 低 |
| 3 | **PAS prevalence-adjusted softmax** | R5 | rare relation (Conversion / Drug_Interaction) recall 升 | 0.5 天 (decode 时改概率) | 低 |
| 4 | **NPE-lite estimator** | L2 | 当 target prior 未知时 (Phase 1 + 1) | 1 天 (LoRA ensemble) | 中 (需多 seed) |

### Tier 2: 1-2 周可试 (中工程量,机理对路)

| # | 方法 | 来源 | 目标改进 | 工程量 | 风险 |
|---|---|---|---|---|---|
| 5 | **CUI RAG / Ontology mapping post-hoc** | P1 + Q1 | `[Dataset]` 零迁移,陌生 tag 自动归一 | 1 周 (UMLS API + 嵌入检索) | 中 (本体覆盖) |
| 6 | **CovEReD entity-counterfactual DA** | V1 | BC8 directional 漏标,跨实体类型泛化 | 1 周 (entity 替换 + 训练) | 中 (合成质量) |
| 7 | **LMRC pre-classifier + LLM 二阶段** | P3 | None 稀释 + per-relation recall 改善 | 1 周 (训练轻量分类器) | 中 (分类器 recall 上限) |
| 8 | **BFT token+sample 维度 reweight** | A7 | 直接替代 D variant reweight | 1 周 (改 loss) | 中 |
| 9 | **Dynamic Data Mixing (LoRA 版改)** | T3 | B variant 多数据集不退化 | 2 周 (gate-load 替代) | 中-高 (架构非 MoE) |
| 10 | **RanAT4BIE 随机 adversarial** | U1 | 跨数据集 robustness | 1 周 (LoRA + Bernoulli PGD) | 低 |

### Tier 3: 实验性 / 长线 (架构变动或方法新颖)

| # | 方法 | 来源 | 目标改进 | 工程量 | 风险 |
|---|---|---|---|---|---|
| 11 | **SAM-GS gradient surgery** | M1 | reweight ⊥ multi-dataset 的根因解 | 2 周 | 中-高 |
| 12 | **CIT-CRE error-guided continual IT** | V2/Z3 | hard error 增量训练 | 2 周 | 中 |
| 13 | **TSV / Pico per-dataset LoRA 合并** | M2 / S3 | 显式分解任务向量 | 2-3 周 | 中-高 (评估口径) |
| 14 | **DES-MoE 多域 MoE** | T1 | 端到端解决 multi-corpus 干扰 | 3+ 周 (改架构) | 高 |
| 15 | **GradFiltering 数据选择** | X1 | 训练数据混料质量 | 2 周 | 中 |
| 16 | **MedINST 大规模 biomed IT 预训** | T2 | base model 升级 | 1+ 月 (算力) | 中 |

### Tier 4: 主要作 framing / 辅助 / 报告对照

| # | 方法 | 来源 | 用途 |
|---|---|---|---|
| 17 | AbstentionBench | D4 | 引言 narrative (LLM 不会弃权) |
| 18 | Subspace Geometry LoRA | S1 | 诊断 B variant 失败 |
| 19 | Hallucination in Synth RE | Z1 | 警示 Sonnet 合成路 |
| 20 | LLM Uncertainty Survey | Y1 | taxonomy reference |
| 21 | LLM-as-judge BioRE (E2) | (已用) | 警示 Sonnet teacher 路 |
| 22 | AI-assisted Discovery (P2) | (旁证) | 论证 0.65→0.79 差距 |
| 23 | KB-Guided / KG-aug Q2-Q4 | (备选) | 若走 RAG 路对照 |

### 推荐执行路径

1. **本周**: 跑 Tier 1 的 (1)(2)(3),都是 < 1 天工程,可立刻验证 prior shift / abstention / long-tail 三个假设
2. **第 2-3 周**: 根据 Tier 1 结果选 1 个 Tier 2 方法深入 — **首选 (5) CUI/Ontology mapping**,因为它对应 Ch 3 contribution 最强的故事 (cross-corpus schema 桥接)
3. **第 4 周**: 多 seed 跑 + ablation 补全,准备 paper draft
4. **保留备份**: 若 Tier 1 全失效,fallback 到 (8) BFT 或 (10) RanAT4BIE,都是 reweight 系的升级版
