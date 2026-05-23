# Quest 023 — Annotation-Efficient Multi-Schema Biomedical RE (CODI-Bi)

**Status**: smoke v5 trained; embedding-crowding gate failed; F1 evals pending in new container.

## Quick read

- **[docs/00_session_summary.md](docs/00_session_summary.md)** — 整个session的总结，从quest 022到smoke v5的关键节点和数字
- **[docs/04_smoke_v5_results.md](docs/04_smoke_v5_results.md)** — CODI-Bi smoke v5的训练曲线、embedding diagnostic、未跑完的eval命令
- **[docs/05_next_steps.md](docs/05_next_steps.md)** — A/B/C/D/E/F 6个候选改造路径 + 推荐smoke v6组合

## Background reads

- **[docs/01_quest_023_brief_v1.md](docs/01_quest_023_brief_v1.md)** — 最初的multi-schema synth + bi-encoder brief
- **[docs/02_quest_023_codi_design.md](docs/02_quest_023_codi_design.md)** — CODI-Bi完整设计spec（input layout, joint training, 4-loss, label encoding, hyperparams, acceptance gates）
- **[docs/03_quest_023_codi_plan.md](docs/03_quest_023_codi_plan.md)** — 14-task TDD implementation plan

## 分支文件分布

```
quest_023_codi/
├── README.md                       本文件（入口 + 完整file map + 恢复指南）
│
├── docs/                           ← 文档（6份，按阅读顺序）
│   ├── 00_session_summary.md       Quest 015→023完整arc，给外行人也能懂
│   ├── 01_quest_023_brief_v1.md    最初v1方案：multi-schema synth + bi-encoder
│   ├── 02_quest_023_codi_design.md CODI-Bi完整设计spec
│   ├── 03_quest_023_codi_plan.md   14-task TDD implementation plan
│   ├── 04_smoke_v5_results.md      v5训练曲线 + diagnostic + 未跑完的eval命令
│   └── 05_next_steps.md            6个候选改造路径（A-F）+ 推荐smoke v6
│
├── src/                            ← 7个CODI模块（全部TDD，unit tests pass）
│   ├── codi_tokenizer.py           <think>/<answer>特殊token + embedding init
│   ├── codi_latents.py             k个可训练d-dim latent向量
│   ├── codi_model.py               Qwen3-4B + LoRA + specials + latents loader
│   ├── codi_inputs.py              师生分支input + pool_positions
│   ├── codi_label_cache.py         label用同模板编码
│   ├── codi_losses.py              codi_step + CODILossConfig + warmup
│   └── codi_data.py                CODIDataset + collator
│
├── tests/                          ← 7个unit test文件
│
├── scripts/                        ← 5个可执行脚本
│   ├── compute_mean.py             计算centering mean
│   ├── smoke_tiny.py               50步sanity smoke
│   ├── train_codi.py               2000步完整训练
│   ├── eval_codi.py                full-RE evaluation
│   └── embedding_diag_v5.py        inter-label cosine诊断
│
├── data/                           ← 三轮合成训练数据（gzipped, 共2.5MB）
│   ├── README.md                   数据说明 + 解压使用 + teacher quality caveat
│   ├── training_v2/                1.6K样本（早期pipeline验证）
│   ├── training_v3/                5.4K样本（smoke v3用）
│   └── training_v4/                11.3K样本（smoke v4 + v5用）
│       ├── synth_train.jsonl.gz    1.5MB gzipped (21MB raw)
│       ├── label_dict.json         27个统一标签 × 5 templates
│       └── mean.pt                 v4 single-token架构的mean（CODI需重算）
│
├── artifacts/                      ← 运行log
│   ├── smoke_v5_train.log          2000步完整训练log
│   ├── smoke_tiny.log              50步sanity log
│   └── diag.log                    v5 embedding crowding诊断输出
│
├── cross_eval/                     ← 外部eval辅助
│   ├── scripts/
│   │   ├── parse_biored_full.py    BioRED pubtator → full-RE JSONL
│   │   ├── parse_drugprot_full.py  DrugProt TSV → full-RE JSONL
│   │   ├── parse_pubtator.py       generic pubtator parser
│   │   └── run_eval.py             quest 022时代的eval脚本（参考）
│   └── label_dicts/                6个label dict
│       ├── biored_full.json        BioRED 9-label
│       ├── biotriplex.json         BioTriplex 21-label
│       ├── drugprot_full.json      DrugProt 14-label
│       ├── ddi.json, gda.json
│       └── training_v4_labels.json 27-label统一vocab
│
├── conftest.py                     pytest sys.path setup
└── pyproject.toml                  pytest config
```

**未在分支里**（容量太大，需在新容器重生成或rsync）：
- v5 adapter `artifacts/smoke_v5/lora_adapter/` (~1GB)
- v4 evaluation结果 `cross_eval/results/`

## To resume in a new container

1. Clone + checkout: `git clone ... && git checkout quest_023_codi`
2. Setup env: `pip install transformers peft torch flash-attn`（Qwen3-4B-Base 在known path）
3. **解压合成数据**：
   ```bash
   gunzip -k quest_023_codi/data/training_v4/synth_train.jsonl.gz
   mkdir -p /tmp/synth_smoke/training_v4
   cp quest_023_codi/data/training_v4/synth_train.jsonl /tmp/synth_smoke/training_v4/
   cp quest_023_codi/data/training_v4/label_dict.json /tmp/synth_smoke/training_v4/
   ```
4. 跑：`python quest_023_codi/scripts/compute_mean.py` → `artifacts/mean.pt`
5. 跑：`python quest_023_codi/scripts/train_codi.py` → `artifacts/smoke_v5/`
6. F1 evals 命令在 [docs/04_smoke_v5_results.md](docs/04_smoke_v5_results.md) §4

OR：直接做下一轮改进，参考 [docs/05_next_steps.md](docs/05_next_steps.md)。

## Key numbers

| Stage | DrugProt full-RE | BioRED full-RE | BioTriplex cross | inter-label cos |
|---|---|---|---|---|
| Quest 022 v1 (pos-only train) | 0.039 | 0.105 | — | — |
| Smoke v3 (5.4K, 500 steps) | — | 0.099 → 0.158 + τ_cal | — | — |
| Smoke v4 (11K, 1000 steps) | — | **0.201** | **0.024** | mean **+0.401**, worst **+0.936** |
| Smoke v5 (11K, 2000 steps, CODI k=4) | **pending** | **pending** | **pending** | mean **+0.546**, worst **+0.900** |
| Upper bound (BioREx 38K human) | 0.808 | 0.796 | — | — |
