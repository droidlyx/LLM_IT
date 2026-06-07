# Phase 1 Smoke Test Findings (2026-06-07)

Variant D + posthoc calibration,每个数据集只跑 5 docs,但**三条核心假设全部直接验证**。

## 结果汇总(5-doc smoke)

| 数据集 | n_pairs | Baseline F1 | Best F1 | Δ | Best method |
|---|---|---|---|---|---|
| cdr (BC5CDR) | 58 (dedup) | **0.3077** | **0.4800** | **+17.2 pt** | P2P uniform |
| disgenet | 33 | 0.5556 | 0.5556 | +0.0 | — (已饱和,recall 100%) |
| BioRED dev (processed_test) | ~50 pairs | 0.5941 | **0.6494** | **+5.5 pt** | TECP α=0.10 |
| BC8 test | ~200 pairs | 0.6067 | **0.6871** | **+8.0 pt** | P2P oracle |

## 关键观察

### 1. cdr — 教科书级 prior shift 验证

```
gold p   : [0.87, 0.13]     # 5 docs 抽样得到的 gold 分布
p_eff    : [0.85, 0.15]     # 模型对自身预测的有效先验
```

但 baseline argmax 给出的 TP=2 FP=4 FN=5,只识别了 28% 的 CID,**完全是 D variant 在全量 cdr 上 0.42 F1 的复制**。

P2P uniform(没有 oracle 也能用)→ TP=6 FP=12 FN=1,F1=0.48(+17pt)。
P2P oracle 反而退步,因为 oracle prior 把 None prior 拉得过低,过度修正。

**结论**: 在 prior shift 严重的 OOD 集上,**P2P uniform 是无 oracle 时的安全 fallback**,与文献 L1 P2P 的 effective prior 概念吻合。

### 2. BC8 — P2P oracle 大胜

```
adjustment vector (BC8 8-class):
  None                  : +0.07     # 几乎不动
  Association          : -0.43     # 减
  Positive_Correlation : +0.11     # 加
  Negative_Correlation : -0.40     # 减
  Comparison           : +1.97     # 大加
  Bind                 : -1.97     # 大减
  Conversion           : +4.37     # 极大加
  Cotreatment          : +7.37     # 极大加
```

模型 p_eff 中 Comparison/Conversion/Cotreatment 接近 0,而 BC8 gold 里这些 rare class 占小但非零。oracle 把它们拉起来后:

- F1: 0.61 → 0.69 (+8 pt)
- precision: 0.55 → 0.67 (+12 pt)
- recall: 0.68 → 0.70 (+2 pt)

Comparison 类 F1 从 0 → 0.5(LA τ=0.5)。

### 3. BioRED dev — TECP 弃权救了 F1

```
TECP α=0.10, threshold 0.725 (entropy)
  abstain rate: 16.7%
  F1: 0.59 → 0.65 (+5.5 pt)
  precision: 0.56 → 0.83 (+27 pt)
  recall: 0.64 → 0.53 (-11 pt)
```

放弃了高熵的 16.7% 预测换来 P 飞涨 R 微跌,净 F1 +5.5 pt。说明 BioRED dev 上的 baseline 错误集中在高熵区。

### 4. 长尾 rare class

- BioRED dev 上 Positive_Correlation: 0.25 → **0.58**(LA τ=1.0)
- BC8 上 Comparison: 0 → **0.50**(LA τ=0.5)

但 LA τ 过大会过校,τ=2.0 把 BC8 F1 干到 0.26,τ=1.0 也会让 BC8 F1 跌到 0.46(因为 BC8 上 P2P oracle 路线更优)。**LA 不是无脑套高 τ**。

## 反思 / 风险

1. **样本量小(5 doc smoke)**: cdr 上 dedup 后只有 58 pair,分布估计噪声大。**等全量跑完才能下定论**。
2. **disgenet 没动**: 5 docs 抽样下 baseline 已 100% recall,只能靠 P 提升,而合成数据上 LA 反而拖 P 下来。可能全量 disgenet 会不同。
3. **P2P oracle vs uniform 不一定哪个赢**: cdr 上 uniform 胜,BC8 上 oracle 胜。需要全量结果决定何时用哪个。
4. **TECP 在 cdr 上不太工作**: 因为 cdr 的错误是 FN 主导(under-predict),弃权只能修 FP。

## 下一步

- ✅ 全量跑(GPU0 BioRED+test, GPU1 OOD 三个)目前后台进行
- 等结果出来对照 smoke 数字
- 把最强方法(预计 cdr P2P uniform, BC8 P2P oracle, BioRED dev TECP+P2P)固化到一个 "best combination" pipeline 里
- 准备 Phase 2:NPE-lite 估 p_target,免去 oracle 依赖
