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

## Code layout

```
src/                7 modules (TDD'd)
  codi_tokenizer.py   <think>/<answer> special tokens + embedding init
  codi_latents.py     k trainable d-dim latent vectors
  codi_model.py       Qwen3-4B + LoRA + specials + latents loader
  codi_inputs.py      build_student/teacher_inputs + pool_positions
  codi_label_cache.py encode label templates via same student template
  codi_losses.py      codi_step + CODILossConfig + warmup
  codi_data.py        CODIDataset + collator

tests/              7 test files, all pass
scripts/
  compute_mean.py     compute global mean for centering
  smoke_tiny.py       50-step smoke (sanity)
  train_codi.py       2000-step full training
  eval_codi.py        full-RE evaluation
  embedding_diag_v5.py inter-label cosine diagnostic

artifacts/          training/diagnostic logs (no adapter binaries)
cross_eval/
  scripts/          parse_biored_full.py, parse_drugprot_full.py, parse_pubtator.py, run_eval.py
  label_dicts/      biored_full, biotriplex, ddi, gda, drugprot_full, training_v4_labels
```

## To resume in a new container

1. Setup env: `pip install transformers peft torch flash-attn` (use Qwen3-4B-Base at known path)
2. Copy synth data to `/tmp/synth_smoke/training_v4/synth_train.jsonl` (or regenerate via Sonnet agent dispatch)
3. Run `python scripts/compute_mean.py` → produces `artifacts/mean.pt`
4. Run `python scripts/train_codi.py` → produces `artifacts/smoke_v5/`
5. Run F1 evals per [docs/04_smoke_v5_results.md](docs/04_smoke_v5_results.md) §4

OR: implement next iteration per [docs/05_next_steps.md](docs/05_next_steps.md).

## Key numbers

| Stage | DrugProt full-RE | BioRED full-RE | BioTriplex cross | inter-label cos |
|---|---|---|---|---|
| Quest 022 v1 (pos-only train) | 0.039 | 0.105 | — | — |
| Smoke v3 (5.4K, 500 steps) | — | 0.099 → 0.158 + τ_cal | — | — |
| Smoke v4 (11K, 1000 steps) | — | **0.201** | **0.024** | mean **+0.401**, worst **+0.936** |
| Smoke v5 (11K, 2000 steps, CODI k=4) | **pending** | **pending** | **pending** | mean **+0.546**, worst **+0.900** |
| Upper bound (BioREx 38K human) | 0.808 | 0.796 | — | — |
