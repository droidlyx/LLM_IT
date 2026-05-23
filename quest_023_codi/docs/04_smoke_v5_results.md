# Smoke v5 Results (CODI-Bi, 2000 steps, 11K synth, k=4)

**Run**：2026-05-20 00:00-00:25
**Adapter**：`/home/ds/quest_023_codi/artifacts/smoke_v5/`（未committed）
**Config**：lora_r=16, batch=4, lr=2e-4, τ=0.07, α=0.5, β=1.0, γ=0.5, warmup=200, ramp=200

---

## 1. 训练曲线

```
steps     avg_loss  acc_S  acc_T  备注
  0-199   3.733     0.374  0.436  warmup, β=γ=0
200-399   2.634     0.509  0.696  ramp期，β γ线性涨
400-599   2.617     0.539  0.762  全开后
600-799   2.095     0.620  0.833
800-999   2.010     0.639  0.835  ← acc_T peak
1000-1199 2.028     0.635  0.833
1200-1399 2.222     0.620  0.830
1400-1599 2.533     0.610  0.795
1600-1799 2.709     0.590  0.745  ← acc_T降5个点
1800-1999 2.694     0.647  0.772  最终
```

- 训练时间：3476s（58 min）
- 峰值显存：13.1 GB
- align_gap：1.129（step 0）→ 0.40（end）

**关键观察**：
- Teacher acc在step 800达0.835 peak，之后**mild overfitting**降到0.77
- Student acc 0.65 plateau，**始终低于teacher** 0.83
- L_align 数值显示为0.000，是scale artifact（2560-d normalized vec的MSE量级是1e-4），从align_gap曲线看alignment实际工作

## 2. Embedding diagnostic (v5 vs v4)

```
Training labels (n=27) inter-cosine:
  mean: +0.546  (v4 baseline: +0.401)   ← WORSE
  max:  +0.928  (v4 baseline: +0.936)   ← marginal improvement

Top-5 closest pairs:
  +0.928  ACTIVATOR ↔ Positive_Correlation
  +0.909  Association ↔ no_relation               ← 警告
  +0.902  PRODUCT-OF ↔ SUBSTRATE_PRODUCT-OF
  +0.900  INDIRECT-UPREGULATOR ↔ INDIRECT-DOWNREGULATOR
  +0.899  INHIBITOR ↔ Negative_Correlation

GATE: INDIRECT-UPREG ↔ INDIRECT-DOWNREG = +0.900
  v5 acceptance: < +0.7  →  FAIL
```

**结论**：multi-token pool + k=4 learnable latents alone**没有解决**directional discrimination和crowding。crowding甚至加重。

## 3. 设计回顾：哪些机制work，哪些不work

| 设计要素 | 预期 | 实际 |
|---|---|---|
| `<think>`+`<answer>`+4 latents结构 | 提供"思考空间" | 结构正确，forward跑通 ✓ |
| Joint teacher/student forward | 避免post-hoc distill的forgetting | acc_S确实涨到0.65 ✓ |
| 4-loss with warmup | 稳定收敛 | 无NaN，无divergence ✓ |
| Multi-position pool (k+1=5 tokens) | 缓解single-token bottleneck | acc_S仍plateau远低于teacher ✗ |
| Label encoding via same template | label/judgment同一representation space | crowding mean +0.546 反更糟 ✗ |
| Learnable latents (4 nn.Parameters) | 多样化"思考模式" | 怀疑collapsed to shared mode（待verify） ✗ |

## 4. F1 evals 缺失

容器迁移前未跑完。计划的命令（在新容器执行）：

```bash
# BioRED full-RE (注意路径是 biored_test_full.jsonl 不是 biored_full_test.jsonl)
PYTHONPATH=/root/shared-nvme/ds-workspace/llm_it_env/lib/python3.12/site-packages \
python scripts/eval_codi.py \
  --data /home/ds/cross_eval/data/biored_test_full.jsonl \
  --label_dict /tmp/synth_smoke/training_v4/label_dict.json \
  --adapter_dir artifacts/smoke_v5/lora_adapter \
  --tok_dir artifacts/smoke_v5/tokenizer \
  --latents_pt artifacts/smoke_v5/latents.pt \
  --special_ids artifacts/smoke_v5/special_ids.json \
  --mean_pt artifacts/smoke_v5/mean.pt \
  --out artifacts/smoke_v5/eval_biored --name biored_full --tau_cal 0.0

# BioTriplex
python scripts/eval_codi.py --data /home/ds/cross_eval/data/biotriplex_test.jsonl \
  --label_dict /home/ds/cross_eval/label_dicts/biotriplex.json \
  --out artifacts/smoke_v5/eval_biotriplex --name biotriplex --tau_cal 0.0 \
  (其余参数同上)

# DrugProt subsample 5K
head -5000 /home/ds/cross_eval/data/drugprot_dev_full.jsonl > /tmp/drugprot_5k.jsonl
python scripts/eval_codi.py --data /tmp/drugprot_5k.jsonl ...
```

预测（基于embedding crowding gate已经fail）：
- BioRED macro F1 ≤ v4的0.201（很可能持平或略低）
- BioTriplex ≤ v4的0.024
- 真要见效需要走 next_steps.md 的其它改造

## 5. Diagnostic value of v5

即便F1不上，v5提供了重要信号：
1. **Joint CODI training pipeline完整work**（acc涨、loss降、无NaN、显存13GB）
2. **Multi-token pool不够**：4个learnable latents + `<answer>` 没有相对单token明显改善embedding几何
3. **Label encoding共享template没解决根本问题**：label embeddings也走同样5-token pool，理论上应该完全对齐，但crowding反而更严重 → 说明问题不在label/judgment representation不一致，而在**模型本身缺乏discriminate biomedical relation方向的内在能力**
4. **acc_S vs acc_T gap (0.65 vs 0.83) 是关键诊断信号**：student在relevant方向上确实学不到teacher的精确判别 → compression失败

## 6. 完整命令历史 (reproducibility)

```bash
cd /home/ds/quest_023_codi

# Setup
mkdir -p src tests scripts artifacts
# ... (各 src/codi_*.py via TDD)

# Compute global mean
PYTHONPATH=$LLMIT python scripts/compute_mean.py  # mean norm 134.87

# Tiny smoke
PYTHONPATH=$LLMIT python scripts/smoke_tiny.py
# 50 steps, 1.6s/step, peak 13GB

# Full smoke v5
PYTHONPATH=$LLMIT python scripts/train_codi.py 2>&1 | tee artifacts/smoke_v5_train.log
# 2000 steps, 58min

# Diagnostic
PYTHONPATH=$LLMIT python scripts/embedding_diag_v5.py | tee artifacts/smoke_v5/diag.log
# GATE FAILED
```

其中 `$LLMIT=/root/shared-nvme/ds-workspace/llm_it_env/lib/python3.12/site-packages`
