# Phase 1: Post-hoc calibration on variant D

Tier-1 实验:不重训,只在变体 D 现成 LoRA checkpoint 上做后置校准,验证三个假设:

1. **Prior shift**(cdr 0.58 / pharmgkb 4.73 pred/gold 比)→ P2P / LA 应该能拉直
2. **Long-tail rare relations**(Conversion / Drug_Interaction)→ LA / PAS 可提 rare-class recall
3. **Over-confident wrong commits**(BC8 1725 Positive_Correlation 漏标)→ TECP 高熵弃权

## 文件结构

```
posthoc/
├── posthoc_methods.py     # 五种方法的纯函数 (CPU, numpy only)
├── sanity_test.py         # 10 个合成单元测试 (CPU)
├── score_pairs.py         # 每个 (pair, label) 的 sequence logprob (需 GPU)
├── eval_adjusted.py       # 应用 5 方法 + F1 对比 + per-class breakdown
├── _make_synthetic_scores.py  # 生成合成 scores JSON (smoke test 用)
├── run_phase1.sh          # 端到端 runner
└── results/               # 输出目录 (推荐 gitignore)
    ├── *_scores.json      # stage 1 输出
    └── *_adjusted.json    # stage 2 输出
```

## 数据流

```
变体 D LoRA checkpoint
        │
        │  stage 1: score_pairs.py (GPU, vLLM)
        ▼
*_scores.json  ←  per (doc, pair, label) sequence logprob
        │
        │  stage 2: eval_adjusted.py (CPU)
        ▼
*_adjusted.json ←  baseline + LA(τ=0.5/1.0/2.0) + P2P(oracle) + P2P(uniform)
                              + TECP(α=0.1) + P2P+TECP
                              + per-class P/R/F1
```

## 一键运行(换上有卡之后)

```bash
# 完整流水线
bash posthoc/run_phase1.sh

# 只跑 scoring(评估稍后做)
bash posthoc/run_phase1.sh score_only

# 只跑评估(已有 scores JSON)
bash posthoc/run_phase1.sh eval_only

# 5-doc 烟测,< 5 min
bash posthoc/run_phase1.sh smoke
```

环境变量可覆盖默认路径:

```bash
VARIANT_DIR=results/biored_finetune/A/checkpoint \
OUT_DIR=posthoc/results_A \
  bash posthoc/run_phase1.sh
```

## 现在无卡能做的验证

```bash
# 1) 单元测试(纯 CPU,5 秒)
python posthoc/sanity_test.py
# 期望: ALL PASSED (10/10)

# 2) 合成数据 smoke test
python posthoc/_make_synthetic_scores.py --scenario cdr --out_path posthoc/results_synthetic/cdr_scores.json
python posthoc/eval_adjusted.py --scores_json posthoc/results_synthetic/cdr_scores.json
# 期望: P2P oracle 比 baseline 高 +3~5 pt F1
```

## 五种方法的数学

记 `logits[c]` 是 LLM 对类别 c 的 sequence logprob(其实是 ln-prob,不是真正 logit,但 argmax 等价)。`p_train` / `p_target` / `p_eff` 是分布向量。

| 方法 | 公式 | 何时用 |
|---|---|---|
| **Baseline** | `argmax(logits)` | 对照 |
| **LA (tau)** | `argmax(logits - tau · log p_train)` | 训练集严重长尾,推理时拉平 |
| **PAS** | `argmax(softmax(logits) / p_train · norm)` ≡ LA tau=1 | 同 LA(注:argmax 等价) |
| **P2P (oracle)** | `argmax(logits - log p_eff + log p_target)` | 训练与目标分布不同,有 p_target 可估 |
| **P2P (uniform)** | `p_target = uniform`,放大 effective bias | 不知道 p_target 时的 fallback |
| **TECP** | `predict argmax UNLESS entropy(softmax(logits)) > τ_cp` | 高熵自动弃权降级到 None |
| **P2P + TECP** | 先 P2P 校,再 TECP 弃权 | 同时有 prior shift + over-confident |

P2P 的 `p_eff = mean_i softmax(logits_i)` 是模型自己暴露的有效先验。在 cdr 上我们 D variant 实测 `p_eff[None] = 0.85` 而 `p_target[None] ≈ 0.65`,P2P 把 None logit 减约 0.27,CID logit 加约 0.34 — 应该能把 F1 从 0.42 推到 0.55+。

## 预期结果(根据失败模式诊断)

| 数据集 | Baseline F1 | 期望 Best F1 | 期望方法 |
|---|---|---|---|
| BioRED dev | 0.6489 | 0.66~0.70 | LA tau=0.5 或 P2P |
| BC8 test | 0.5646 | 0.58~0.62 | P2P 或 P2P+TECP |
| cdr | **0.4209** | **0.55+** | P2P oracle |
| disgenet | 0.8485 | 0.85+ | (已饱和,几乎不动) |
| pharmgkb | **0.2565** | **0.40+** | P2P oracle |

cdr/pharmgkb 是主要赌点 — 这两个的失败已经诊断为典型 prior shift。

## 后续可扩展

- **NPE-lite p_target estimator**: 当 p_target 未知,用 LoRA ensemble 在 OOD 上估
- **Per-doc adjustment**: 文档级 prior 估,处理 doc-level shift
- **Conformal Long-Tail PAS**: arXiv 2507.06867 的 prevalence-adjusted softmax + FUZZY CP
- **Decoupled score-then-abstain**: 先 P2P,再 R5 的 set-valued prediction 给 BC8 弃权
