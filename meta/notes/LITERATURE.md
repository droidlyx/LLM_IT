# BioRE 相关文献 — 按想法类别梳理

**整理时间**: 2026-05-28 (初版) / 2026-05-30 (新增 L/M/N/O 4 个方向,共 19 篇)
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
