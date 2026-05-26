# BioRE 研究方向调研与实验笔记

**作者**: 刘宇轩(yuxliu21@m.fudan.edu.cn)
**整理时间**: 2026-05-24
**目的**: 整合 deepxiv 文献调研 + LLM_IT 实验诊断 + 数据集清单,作为 Ch 3(BioRE)thesis chapter 规划的工作底稿。

---

## 0. 上下文

**博士论文标题(开题报告)**: 基于大语言模型的生物医学文献复杂知识抽取技术研究

**Thesis 三章结构**:
- Ch 1/2 — **VANER2**(已发表一作): LLM-as-encoder + 39 BioNER 数据集统一训练,跨数据集泛化
- Ch 3 — **BioRE**(进行中,本文档对应): 沿 VANER2 思路想做多数据集 unified RE,**但已实验验证不行**
- Ch 4 — **GO-CAM 因果网络生成**(未启动): 需要 mechanistic 关系作为 input

**核心约束**: 走 BioRED-critique 路线不适合博士主线,需要 method-driven contribution,且要衔接 Ch 4(GO-CAM 需要因果机制关系)。

---

## 1. 当前 BioRE 任务的关键性能数字

| 系统 / 数据集 | F1 | 备注 |
|---|---|---|
| Llama-3.1-8B SFT (LLM_IT baseline) on BioRED dev | **0.69** | Pair-wise enumeration prompt,user 的当前 baseline |
| 同模型在 BC8 test | **0.52** | OOD,差 16 pt |
| 同模型在 BioRED + 方向 | -0.10 左右 | 加方向更难 |
| BioREx (38K 跨数据集监督) on BioRED | **0.796** | 当前公认 SOTA / 上限 |
| BioRED IAA 上限 | **0.79–0.85** | 原论文报告 |

**关键观察**: 0.65–0.69 是各种方法都难以突破的瓶颈,IAA 上限只比 0.69 高约 15 个点。说明剩余 gap 中很大一部分是 **annotation-level noise / inconsistency**,不是模型容量问题。

---

## 2. 实验诊断关键结果(本次 session 跑出来的)

### 2.1 Sonnet teacher quality 测试(50 BC8 docs)

| 方法 | Type-strict F1 | Pair-only F1(忽略类型) |
|---|---|---|
| Sonnet free-form 生成 | 0.420 | 0.57 |
| Sonnet 强制 per-pair classification | **0.431** | **0.630** |
| Llama-3.1-8B SFT(对比 baseline) | **0.69** | — |

**Sonnet 即便强制 per-pair,TYPE F1 也只 0.43,比 8B SFT 低 26 个点**。

**结论**: **任何 "Sonnet → 蒸馏到 8B" 的路线在数学上不可能让 8B 超过 0.43**。教师质量是 hard ceiling,不是软约束。Q022 / Q023 路线由此被否定。

### 2.2 8B 错误分解(100 BioRED dev docs)

| 错误类别 | 数量 | 占比 | 性质 |
|---|---|---|---|
| CORRECT | 1622 | — | — |
| Type-confusion(pair 对,type 错) | 344 | 41% of INCORRECT | 可修 |
| Pair hallucination(pair 不在 gold) | 488 | 59% of INCORRECT | 难修 |
| True miss(pair 完全没预测) | 322 | — | 跟训练数据强相关 |
| **Pair-level any-type F1 上限**(只修 type) | **0.83** | — | 接近 IAA 上限 |

**Type confusion 矩阵的核心模式**:
- `Association → Positive_Correlation`(111)+ `Positive_Correlation → Association`(89)= **200 个 Association ↔ P_Corr 错配**
- `Association ↔ Negative_Correlation` 共 100 个错配
- **87% 的 type confusion 都集中在 Association ↔ directional 的边界上**

### 2.3 BioRED dev vs BC8 行为完全翻转

| | BioRED dev | BC8 test |
|---|---|---|
| F1 | 0.69 | 0.52 |
| `Association → directional`(over-commit) | 163 | 470 |
| `directional → Association`(under-commit) | 137 | **1582** ⚠️ |

**关键发现**: BioRED dev 上模型倾向 over-commit;BC8 上完全反转,under-commit 是 over-commit 的 3.4 倍。

**这不是 "模型不会做 RE",是 "模型在 OOD 上对 directional commitment 的 calibration 失效"**。1582 个 directional→Association 错配是 BC8 的最大失分项。

### 2.4 BioRED 标注 schema 的 7+ 条暗规则(本次手工解码)

从 4 篇高代表性 doc + 50 doc Sonnet 测试,反推出 BioRED 标注员实际遵循的规则:

1. **实体类型门**: 关系两端必须都在 6 类(GeneOrGeneProduct/Disease/Chem/Variant/Organism/CellLine)
2. **论文断言门**: 只标论文明确声称/发现的关系,背景知识跳过
3. **Null result 不标**(不是标 Negative_Correlation,而是不标)
4. **Polarity 翻转**: 药物治疗疾病 → Negative_Correlation
5. **Mention 合并**: 同 ID 合并;基因层与变异层用不同关系强度
6. **通用 vs 具体取具体**: cancer vs breast cancer → 用具体的
7. **Novel vs No**: 论文新发现 vs 已知背景
8. **笛卡尔展开**(case-report 类 doc): 集体性陈述展开成 pairwise

**8B 学到了规则 1-7 的大部分(in-distribution),但规则 2/3/8 在 OOD 上崩盘**。

### 2.5 数据集清单(15 个 pubtator 文件)

| 数据集 | docs | rels | rel 类型数 | rel 标签 |
|---|---|---|---|---|
| **biored** train_dev | 500 | 5383 | 8 | Association(52%)/Pos/Neg/Bind/Cotreatment/... |
| **biored** test | 100 | 1164 | 8 | 同上 |
| **biored** bc8 | 400 | 6036 | 7 | 同上 |
| **bioredirect**(三个 split) | 同上 | +方向信息 | **解析器报告 1743 类** ⚠️ | 文件中混入了"方向标记行",需 parser fix |
| **drugprot** | 4250 | 18181 | 13 | INHIBITOR/DIRECT-REG/SUBSTRATE/ACTIVATOR/... |
| **ddi** | 4501 | 4956 | 4 | effect/mechanism/advise/int |
| **gda** | 8000 | 8136 | 3 | Negative/Biomarker/Therapeutic(弱标签) |
| **cdr** | 1500 | 3116 | 1 | CID |
| **disgenet/aimed/hprd50/emu/pharmgkb** | 共 2833 | 共 2938 | 1 each | 全 "Association" |

**Schema 异质度**:
- 跨 9 个训练数据集共 29 个 relation 类型
- **唯一跨数据集的 label 是 `Association`**(6 个数据集都有),但语义不一致
- 24 unique entity types,**命名都不统一**(Gene / GENE-Y / GeneOrGeneProduct 等)

---

## 3. 文献综述:RE / BioRE 现状(via deepxiv 2024-06 ~ 2026-05)

### 3.1 当前主流范式(2023-2026)

**生物医学 RE 仍以 fine-tuned encoder 为 SOTA**:
- BioBERT / PubMedBERT / SciBERT + classification head
- **BioREx (arXiv:2306.11189)** 是公认上限: 跨多个 BioRE 数据集 schema harmonization,38K 监督,BioRED F1 = 0.796

**通用 RE 三条并行**:
- Generative RE(REBEL / GenIE / TplinkerPlus)— T5 seq2seq
- LLM In-context(GPT-4 + CoT + retrieval)
- Unified IE(UIE 2022 / **InstructUIE** 2023 / USM)

**新兴方向(2024-2026)**:
- Schema-free / instruction-tuned IE
- 自蒸馏 / consistency training
- **RL 训练 RE**(R1-RE,见下)

### 3.2 高威胁的同类工作

#### (a) **R1-RE**(arXiv:2507.04642,2025-07,9 citations)
"Cross-Domain Relation Extraction with RLVR"
- 用 **RL with Verifiable Reward**(GRPO)+ "annotation guideline reasoning"
- **OOD 上 +30pp over SFT**,匹配 GPT-4o
- **核心 thesis**: SFT 学到 shallow memorization,他们用 RL 强制模型按 guideline 推理
- **跟你方向的差异**: 他们假设 label 对、模型不会推理;你的诊断显示 label 本身有问题
- **可作为 Ch 3 baseline 对比**(同样 OOD generalization 目标)

#### (b) **BioREDirect**(arXiv:2501.14079,2025-01,6 citations,NCBI 团队)
"Enhancing Biomedical Relation Extraction with Directionality"
- 给 BioRED 加 **subject/object 方向标注**(扩展到 10,864 pairs)
- multi-task: 联合预测 rel + novelty + entity role
- soft-prompt + context chunking
- 跑赢 GPT-4 / Llama-3 (PEFT)
- **跟你方向的差异**: 他们的 "directionality" 是角色(subject/object),**不是 commitment strength**。你做的"机理 vs 统计"维度跟它正交,可同时叠加
- **数据**: bioredirect 文件就是这篇的产物,直接可用

#### (c) **BioREx**(arXiv:2306.11189,2023-06,21 citations,user 的 baseline 上限)
- 跨数据集 schema alignment + harmonized training
- BioRED F1 从 74.4 → 79.6
- **跟你方向的差异**: 它只 align 不同数据集的 schema,假设每个数据集自己的 label 是对的;你想 question 单数据集内的 label 一致性
- **数据**: 已开源 (github.com/ncbi/BioREx)

#### (d) **Beyond the Numbers**(arXiv:2411.05224,2024-11,2 citations)
"Transparency in Relation Extraction Benchmark Creation and Leaderboards"
- **Position paper**(没具体实证数据,只是呼吁)
- 批判 TACRED/NYT 的 transparency 不足 + class imbalance + noisy labels
- 呼吁 dataset datasheets + class-level analysis
- **跟你方向**: 它给"empirical critique"提供了模板;你做的是 BioRED 的实证版本。**它没批 BioRED**

### 3.3 衔接 / 工具型相关工作

#### (e) **2502.12159 Causal Language in Observational Studies**(2025-02)
- 在 80k observational study abstracts 上做 **causal vs correlational 二分类**
- 已有 **BioBERT-based classifier 开源**: github.com/junwang4/causal-language-use-80k-observational-studies
- **可直接用作 Ch 3 工具组件**: 给 BioRED 关系自动打 causal/correlational tag,作为辅助监督信号

#### (f) **2503.11376 UnScientify**(2025-03)
"Annotating Scientific Uncertainty"
- 规则 + pattern matching 检测 scientific uncertainty
- **比 LLM 更准**(0.808 accuracy)
- 检测 "research limitations / variability / ungeneralizable conclusions" 等多种不确定性,超越传统 hedging
- **可用于** Ch 3: 跟 (e) 互补,(e) 看 causal commitment,(f) 看 hedging confidence

#### (g) **2112.02570 Uncertain Biomedical Knowledge**(2021-12)
- 用 Information Entropy + Uncertainty Rate 量化 SPO triple 的不确定性
- 限定中国心血管文献
- **跟 (e) (f) 思路相近,但聚焦 KG 构建,没接入 RE benchmark**

#### (h) **2503.18526 SciClaims**(2025-03)
"End-to-End Generative System for Biomedical Claim Analysis"
- Llama3 7B + Elasticsearch claim 检索 + LLM 验证
- 全 pipeline 无 fine-tune
- **可作为 Ch 4 GO-CAM 的 building block**(claim grounding)

#### (i) **2509.13888 CER**(2025-09)
"Combating Biomedical Misinformation through Multi-modal Claim Detection"
- 多模态(text/web/video)claim 检测 + PubMed 检索 + LLM reasoning
- HealthFC / BioASQ-7 / SciFact SOTA
- **关联**: claim verification 跟 GO-CAM 因果声明验证有衔接价值

#### (j) **2602.04320 GUT-BRAINIE**(2026-02,最新)
"A Domain-Specific Curated Benchmark for E and Doc RE"
- 新做 gut-brain 领域 1,647 abstracts benchmark
- 引入 **Mention-level RE (M-RE) vs Concept-level RE (C-RE) 区分**
- **跟你方向**: 它的 M-RE vs C-RE 区分可用来支撑你发现的"变异 granularity 不一致"问题

### 3.4 其他可能有用的发现

- **2506.00777**(2025-06,10 citations): LLM-as-judge 在 BioRE 上不可靠,structured output formatting 修复 +15%。**评估方法论**角度。
- **2505.01077**(2025-05): Zero-shot doc-level BioRE via scenario prompt,ChemDisGene / CDR 上接近 fine-tune
- **2503.17799**(2025-03): Instance-Adapted Predicate Descriptions,dual-encoder + 现成 predicate 模板,+1-2.1% F1
- **2410.23583 BioNCERE**(2024-10): non-contrastive learning + transfer learning for BioRE,不依赖 NER 标签
- **2205.12696 Revisiting DocRED**(2022): 重标 DocRED 解决 false negative —— **BioRED-Revisited 风格的范例**(但你不走这条路)

### 3.5 文献 landscape 表

| 工作 | 角度 | 跟你 Ch 3 关系 |
|---|---|---|
| BioREx | 跨数据集 schema alignment | 直接 baseline / 上限 |
| BioREDirect | 加 subject/object 方向 | 正交维度,可叠加 |
| R1-RE | RL teach guidelines | OOD generalization 同目标,可对比 |
| InstructUIE / UIE | 统一 IE 多任务 | 通用方法,非 biomed-specific |
| BioNCERE | 非对比学习 | 训练方法,可借鉴 |
| **Causal Language paper** | **causal classifier 工具** | **building block** |
| **UnScientify** | **uncertainty 检测工具** | **building block** |
| GUT-BRAINIE | M-RE vs C-RE 概念 | 引用作为 motivation |
| Beyond the Numbers | RE benchmark critique | framing 模板(但你做 method 不只 critique) |
| Revisiting DocRED / Re-TACRED | dataset re-annotation | 不走这条路 |

**关键判断**: 截至 2026-05,**没人对 BioRED 做过 TACRED-Revisited 等级的实证 critique**;**没人把 causal classifier(2502.12159)接进 RE pipeline**;**没人把 commitment-strength 维度作为 BioRE 的辅助标签**。这三块是真空地带。

---

## 4. 备选 Ch 3 framing(基于以上)

### Framing X: Schema-Decoupled Two-Stage RE
- Stage 1: schema-free 自然语言关系描述(所有数据集共用)
- Stage 2: 每个 dataset 一个轻量 mapper(LoRA / linear head)
- Stage 1 是 GO-CAM 的天然 input
- **跟现有工作差异**: 不是 align schemas(BioREx),不是 RL guideline(R1-RE),是分层 representation

### Framing Y: Mechanism-Aware Hierarchical Labels
- 不动多数据集训练,只用 BioRED + bioredirect
- 用 2502.12159 + UnScientify 自动给 gold 关系打 mechanism tag(causal / statistical / observational)
- 双 head: 原 label + mechanism tag
- mechanism tag 在 OOD(BC8)提供 transferable signal
- 直接 feed Ch 4(GO-CAM 只接受 causal 关系)

### Framing Z: Negative Transfer Diagnostics in BioRE
- 用 user 失败的多数据集实验作为 Ch 3 引子
- 理论分析为什么 RE 比 NER 难 unify
- 提出针对性 mitigation(可能是 X 或 Y)
- 跟 VANER2 (NER 成功 unify) 形成对比 case study

**初步推荐**: Y(技术 risk 最低,衔接 Ch 4 最自然,跟现有工作正交度最高)。但需要先跑 (e) Causal Language classifier 在 BioRED 上验证可分性。

---

## 5. 数据集决策

### 5.1 可用于多数据集训练(rich schema)
| 数据集 | docs | rels | 加入训练? |
|---|---|---|---|
| biored | 500 | 5383 | ✅ 主数据集 |
| bioredirect | 同 + 方向 | — | ⚠️ 先修 parser |
| drugprot | 4250 | 18181 | ✅ 互补 domain |
| ddi | 4501 | 4956 | ✅ 互补 domain |

### 5.2 单标签 / 弱标签数据集(慎用)
| 数据集 | 标签 | 问题 |
|---|---|---|
| gda | Neg/Biomarker/Therapeutic | DisGeNET 置信度推断,非真标注 |
| cdr | 仅 CID | 单标签,信号窄 |
| disgenet/aimed/hprd50/emu/pharmgkb | 全 Association | collapse 所有 schema 信号 |

**建议**: 不要把 Association-only 数据集加入训练 —— 会让模型学到"啥都是 Association"。

### 5.3 已知数据问题
- **bioredirect 文件**: 含方向标记行(`PMID\tID1\tID2\tSubject:IDX` 格式),原 parser 把方向标记的实体 ID 当成 rel type 解析,导致看起来有 1743 个 rel 类型。需 parser fix。
- **train_llm.py:385 路径 bug**: `./dataset/biomedical/*.pubtator`(小写)→ 实际是 `./dataset/Biomedical/processed/*.pubtator`。**之前 user 跑的多数据集实验很可能 glob 没匹配到任何文件,所谓 "略有下降" 可能只是随机性**。需重做实验。

---

## 6. 推荐的下一步执行序列

按优先级 + 依赖关系:

### P1. 验证 Causal Language classifier 在 BioRED 可分性(决定 Framing Y 是否成立)
- 拉 github.com/junwang4/causal-language-use-80k-observational-studies
- 在 BioRED train_dev 5383 个 gold relation 上跑
- 看 Pos_Corr / Neg_Corr 关系在 sentence 层是否被 classifier 分到 causal,Association 是否分到 correlational
- 输出:每类 BioRED label 的 causal/correlational 分布表

### P2. 修 train_llm.py path bug + bioredirect parser,重做 multi-dataset 实验
- 修 `./dataset/biomedical/*.pubtator` → `./dataset/Biomedical/processed/*.pubtator`
- 加 explicit allowlist 参数(不要 glob 所有 *.pubtator)
- 修 prepro.py 让 `read_biored` 能识别 bioredirect 的方向标记行
- 跑 ablation: biored only / +drugprot / +ddi / +drugprot+ddi
- 记录每个组合在 BioRED dev / BC8 上的 F1 + per-relation-type F1

### P3. 8B baseline 重跑 + per-doc-type 分析
- 已有 dev_results.txt(F1=0.69)和 bc8_results.txt(F1=0.52)
- 加一个 per-doc clustering(case-report vs experimental vs review),看哪类 doc 上 OOD gap 最大
- 这是 Ch 3 motivation section 的核心数据

### P4. 写 Ch 3 paper plan(method + experiments + ablation)
- 基于 P1 结果选 Framing X / Y / Z
- 列具体 baseline(BioREx / BioREDirect / R1-RE / vanilla SFT)
- 列实验:in-distribution + 3-4 个 OOD eval

### P5. 跟 Ch 4(GO-CAM)对接的接口设计
- Ch 3 输出格式应包含 (entity1, relation, entity2, mechanism_tag, confidence)
- mechanism_tag 决定 Ch 4 是否纳入 GO-CAM 图
- 设计这个接口能让 Ch 4 顺利启动

---

## 7. 已被否定的方向(避免重复探索)

- ❌ Sonnet → 8B distillation:**Sonnet F1=0.43 < 8B F1=0.69**,数学上不可能提升
- ❌ Naive multi-dataset union 训练:user 已实验验证 hurt(虽然 path bug 让结果存疑,但理论上 schema conflict 必然存在)
- ❌ BioRED-Revisited 作为 thesis 主线:critique-only,博士不够撑 3-4 章
- ❌ 27-label unified synth(Q023 v1):合成数据教师质量 39% FP,real F1 = 0.523
- ❌ CODI-Bi multi-token learnable latents(Q023 v2):embedding crowding 反而变 worse(+0.401 → +0.546)
- ❌ RAG-only 路线:文献已饱和(SciClaims / CER / RAG benchmark 等),难做出创新

---

## 8. 关键资源清单

### 工具
- **deepxiv** SDK(arxiv 搜索 + 段落级阅读): pip install deepxiv-sdk,token 在 ~/.env
- **Sonnet via Claude Code sub-agent**: 无 API key 需求,适合 batch 推理
- **2502.12159 Causal Classifier**: github.com/junwang4/causal-language-use-80k-observational-studies
- **UnScientify**(待确认仓库链接)

### 关键数据
- BioRED: `dataset/biored/processed_*.pubtator`
- bioredirect: `dataset/Biomedical/bioredirect/*.pubtator`
- 其他: `dataset/Biomedical/processed/*.pubtator`
- 8B baseline 预测: `/root/gpufree-data/dev_results.txt`, `bc8_results.txt`
- 数据 inventory: `teacher_eval/results/dataset_inventory.json`
- Sonnet teacher 诊断: `teacher_eval/results/eval_schema.json`, `eval_pair.json`
- 文献 scan: `teacher_eval/results/lit_scan.json`, `lit_scan_recent.json`

### 代码 entry points
- 训练: `scripts/llm_biored.sh` → `train_llm.py`
- 测试: `test_llm.py`
- Prompt 模板: `meta/baseline/extract.txt`
- 数据预处理: `prepro.py`(read_biored)
- 数据归一化: `dataset/unify_pubtator.py`

---

## 附录 A: BioRED 8 种 relation 类型分布(train_dev)

| Type | Count | % |
|---|---|---|
| Association | 2789 | 51.8% |
| Positive_Correlation | 1443 | 26.8% |
| Negative_Correlation | 983 | 18.3% |
| Bind | 80 | 1.5% |
| Cotreatment | 41 | 0.8% |
| Comparison | 33 | 0.6% |
| Drug_Interaction | 11 | 0.2% |
| Conversion | 3 | 0.1% |

## 附录 B: 8B baseline 跨数据集对比详表

| 指标 | BioRED dev | BC8 test | Delta |
|---|---|---|---|
| docs | 100 | 400 | — |
| CORRECT | 1622 | 6154 | — |
| MISSED | 666 | 4846 | — |
| INCORRECT | 832 | 6276 | — |
| Type-confusion of INCO | 41% | 37% | -4% |
| Pair-hallucination of INCO | 59% | 63% | +4% |
| Association → P_Corr/N_Corr | 163 | 470 | +307 |
| P_Corr/N_Corr → Association | 137 | **1582** | **+1445** |
