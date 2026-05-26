# 分支索引

记录所有非 `main` 分支上的探索性工作,方便从主线检索。

---

## `origin/quest_023_codi`

**最后更新**: 553a859 (README: complete file tree with annotations)
**状态**: smoke v5 训完;embedding crowding gate **失败**;BioRED / DrugProt / BioTriplex 的 full-RE F1 评估 **未跑完**
**最近 commit 链**(新→旧):
- 553a859 README: complete file tree with annotations + data/ section + clearer resume guide
- b2c2450 Add synth training data (v2/v3/v4 gzipped) to avoid regen cost
- 8b02cac Fix: positive-only eval was our Q022 mistake, not community convention
- 7e54126 Rewrite session summary: cover full quest 015-023 chain
- a661e35 Quest 023: CODI-Bi implementation + smoke v5 results

### Quest chain 概览(从 Q015 到 Q023)

| Phase | 尝试 | 结果 |
|---|---|---|
| Q015-020 | 9 种 post-hoc 方法(frozen baseline 上做修正/重排/过滤) | 全部失败或微弱效果 |
| Q015-020 | 2 种**训练数据层**干预(entity rename / KB-hint) | 都有正效果(BC8 +1pt) |
| **关键 takeaway** | **SFT data 层动手 > inference 时纠错(差一个数量级)** | — |
| Q021 | verbalized prompts + SapBERT 相似度做 RE | 结构性失败:DrugProt 0.51 / BioRED 0.13 |
| **Q022 (ICLR)** | **Silent CoT**: 单 token bi-encoder + post-hoc 蒸馏 | DrugProt 0.696,**但 eval protocol 错** |
| Q023 v0 发现 | 重做 full-RE eval(含 no_relation) | DrugProt **0.039** / BioRED **0.105** —— Q022 数字塌掉 |
| Q023 v1 | 27-label 合成数据 + bi-encoder | BioRED 0.201 (BioREx 上限 0.796 的 25%) |
| **Q023 v2 (CODI-Bi)** | k=4 learnable latents + 师生联合 forward + 4-loss | smoke v5 训完,**embedding crowding 反而变更糟** |

### Smoke v5 关键数字

| 指标 | v4 (single-token) | v5 (CODI k=4) | 期望 | 结果 |
|---|---|---|---|---|
| Mean inter-label cosine | +0.401 | **+0.546** | < +0.40 | ❌ 更糟 |
| UPREG↔DOWNREG cosine | +0.936 | +0.900 | < +0.70 | ❌ 几乎没变 |
| Student acc plateau | — | 0.65 | — | 始终低于教师 0.83 |

**结论**: multi-token pool + learnable latents alone 没解决 embedding crowding。teacher 质量天花板也低(Sonnet 给负样本 39% FP)。

### 分支文件结构

```
quest_023_codi/
├── README.md                       入口
├── docs/                           6 份文档
│   ├── 00_session_summary.md       Q015→Q023 完整 arc(给外行人也能懂)
│   ├── 01_quest_023_brief_v1.md    v1 方案: multi-schema synth + bi-encoder
│   ├── 02_quest_023_codi_design.md CODI-Bi 完整 spec(input layout, joint training, 4-loss, hyperparams)
│   ├── 03_quest_023_codi_plan.md   14-task TDD implementation plan
│   ├── 04_smoke_v5_results.md      v5 训练曲线 + diagnostic + 未跑完的 eval 命令
│   └── 05_next_steps.md            6 个候选改造路径(A-F)+ 推荐 smoke v6 组合
├── src/                            7 个 CODI 模块(全部 TDD,unit tests pass)
│   ├── codi_tokenizer.py           <think>/<answer> 特殊 token + embedding init
│   ├── codi_latents.py             k 个可训练 d-dim latent 向量
│   ├── codi_model.py               Qwen3-4B + LoRA + specials + latents loader
│   ├── codi_inputs.py              师生分支 input + pool_positions
│   ├── codi_label_cache.py         label 用同模板编码
│   ├── codi_losses.py              codi_step + CODILossConfig + warmup
│   └── codi_data.py                CODIDataset + collator
├── tests/                          7 个 unit test
├── scripts/                        5 个可执行脚本
│   ├── compute_mean.py             计算 centering mean
│   ├── smoke_tiny.py               50 步 sanity smoke
│   ├── train_codi.py               2000 步完整训练
│   ├── eval_codi.py                full-RE evaluation
│   └── embedding_diag_v5.py        inter-label cosine 诊断
├── data/                           三轮合成训练数据(gzipped, 共 2.5MB)
│   ├── training_v2/                1.6K 样本(早期 pipeline 验证)
│   ├── training_v3/                5.4K 样本(smoke v3 用)
│   └── training_v4/                11.3K 样本(smoke v4 + v5 用)
│       ├── synth_train.jsonl.gz    1.5MB gzipped (21MB raw)
│       ├── label_dict.json         27 个统一标签 × 5 templates
│       └── mean.pt                 v4 single-token 架构的 mean(CODI 需重算)
├── artifacts/                      运行 log
│   ├── smoke_v5_train.log          2000 步完整训练 log
│   ├── smoke_tiny.log              50 步 sanity log
│   └── diag.log                    v5 embedding crowding 诊断输出
├── cross_eval/                     外部 eval 辅助
│   ├── scripts/
│   │   ├── parse_biored_full.py    BioRED pubtator → full-RE JSONL
│   │   ├── parse_drugprot_full.py  DrugProt TSV → full-RE JSONL
│   │   ├── parse_pubtator.py       generic pubtator parser
│   │   └── run_eval.py             Q022 时代的 eval 脚本(参考)
│   └── label_dicts/                6 个 label dict
│       ├── biored_full.json        BioRED 9-label
│       ├── biotriplex.json         BioTriplex 21-label
│       ├── drugprot_full.json      DrugProt 14-label
│       ├── ddi.json, gda.json
│       └── training_v4_labels.json 27-label 统一 vocab
├── conftest.py                     pytest sys.path setup
└── pyproject.toml                  pytest config
```

### 未在分支里(容量太大需重生成或 rsync)

- v5 adapter `artifacts/smoke_v5/lora_adapter/` (~1GB)
- v4 evaluation 结果 `cross_eval/results/`

### 切到该分支恢复工作

```bash
git checkout quest_023_codi
# 解压合成数据
gunzip -k quest_023_codi/data/training_v4/synth_train.jsonl.gz
mkdir -p /tmp/synth_smoke/training_v4
cp quest_023_codi/data/training_v4/synth_train.jsonl /tmp/synth_smoke/training_v4/
cp quest_023_codi/data/training_v4/label_dict.json /tmp/synth_smoke/training_v4/
# 复跑
python quest_023_codi/scripts/compute_mean.py        # → artifacts/mean.pt
python quest_023_codi/scripts/train_codi.py           # → artifacts/smoke_v5/
# F1 evals 命令在 docs/04_smoke_v5_results.md §4
```

### 为什么没走完(决策记录)

`05_next_steps.md` 提了 6 条候选改造路径:
- A. Per-label latents
- B. Drop `<answer>` from pool
- C. Cosine-loss 替代 MSE for align
- D. Coconut-style hidden state recursion
- E. Bidi attention on latent block
- F. Generative pivot

**未执行**。当前 session 的诊断(见 `meta/notes/RESEARCH_NOTES.md` §2.1)显示 **Sonnet 当 teacher 的 F1 只有 0.43,比 8B SFT 的 0.69 低 26 个点**,所以**任何蒸馏路线都有 hard ceiling**。Q023 v2 路线(CODI-Bi)在此基础上继续推已无 ROI。

### 跟当前主线的关系

主线(`origin/main`)已切换到:
- LLM_IT 8B SFT pair-enumeration baseline
- Multi-dataset 集成训练(尝试中)
- Ch 3 framing 候选: 见 `meta/notes/RESEARCH_NOTES.md` §4

quest_023_codi 上的合成数据 / CODI 代码可能在未来作为 Ch 3 实验对照(比如对比"教师蒸馏" vs "直接 SFT")时复用,**但不再作为主推路线**。

---

(后续若开新分支,在此追加索引)
