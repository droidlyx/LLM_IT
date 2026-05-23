# Next Steps after Smoke v5 GATE FAIL

按改动成本从低到高排列。建议先A+B+C组合（无breaking changes），再决定要不要D/E/F。

---

## A. Per-label latents（label-conditional reasoning）

**改动**：`LearnableLatents(k, d_model)` → `LearnableLatents(k, n_labels, d_model)`。训练时根据gold label index选对应的latent slice。推理时**遍历所有n_labels个slice，每个label得到一个judgment vec → cosine with对应label_emb → argmax**。

**预期**：每个label有自己的"思考方向"，避免4个共享latent collapse到shared mode。

**代价**：n_labels × k个参数（27 × 4 × 2560 ≈ 276K 仅向量），推理变n_labels倍forward（27倍）。可以batch化但额外开销。

**实现**：
- 修改`src/codi_latents.py`：增加n_labels维度
- 修改`src/codi_inputs.py:build_student_inputs`：接受label_idx，slice正确的latent
- 推理：每个input N次forward（每个label一次），取best cosine

---

## B. Drop `<answer>` from pool（消除聚合主导）

**改动**：pool from `{L_0..L_{k-1}, <answer>}` (5 positions) → `{L_0..L_{k-1}}` only (4 positions)。

**Rationale**：`<answer>`位置在causal attention下看到所有之前内容（包括k个latents），它的hidden state本质上是latents的"summary"。Mean pool把summary和原始latents 1:k混合，summary作用约占1/5 → 几乎等价于single-token pool。

**预期**：去掉`<answer>`，强迫k个latent**各自**承载discriminative信号。

**代价**：只改`build_student_inputs`和`build_teacher_inputs`中pool_idx计算（教师对应是last k reasoning tokens）。1行修改。

---

## C. Cosine-loss替代MSE for L_align

**改动**：
```python
# 现在
L_align = F.mse_loss(s_centered_norm, t_centered_norm)  # ~1e-4量级

# 改为
L_align = 1.0 - (s_centered_norm * t_centered_norm.detach()).sum(dim=-1).mean()  # ~1e-1量级
```

**预期**：梯度magnitude放大100×，align信号在4-loss中真正发挥作用（目前β=1.0 × 1e-4 vs L_cls=0.5 → align贡献仅0.02%）。

**代价**：1行修改。

---

## D. Coconut-style hidden-state recursion

**改动**：latent embeddings不再是static `nn.Parameter`，而是**第一个latent**从`<think>`的hidden state得到，**第k个latent**的input = 第k-1个latent的output hidden state。需要分k次forward。

**预期**：latent间有"reasoning chain"，类似Coconut的连续思考。

**代价**：训练forward变k+1次/sample（teacher也需要类似改造）。代码改造大（custom forward loop）。可能跟LoRA + gradient checkpointing 配合复杂。

---

## E. Bidi attention on latent block

**改动**：在`<think>...<answer>`之间的k个latent positions之间用bidirectional attention，外部仍causal。

**预期**：参考NV-Embed latent attention head，k个latent能互相refine信息。

**代价**：要改attention mask构造（HuggingFace自定义4D mask）。可能跟flash_attention_2不兼容（需切回sdpa）。

---

## F. Generative pivot

**改动**：抛弃bi-encoder的cosine head，让模型直接generate label name token序列（e.g. "INDIRECT-UPREGULATOR"）。Loss = teacher-forced NLL on label tokens after `<answer>`。

**预期**：
- Embedding几何不再是bottleneck，因为不再用单vector判别
- 推理需要beam search或constrained decoding（限制到label vocabulary）
- 失去bi-encoder高效推理 + label transferability（新schema需要至少几个示例）

**代价**：method-level rewrite，paper story从"label-template-based zero-shot"转向"generative few-shot"。

---

## 推荐组合实验：smoke v6

按低成本顺序：

### v6a (A+B+C)
- Per-label latents (A)
- Drop `<answer>` from pool (B)
- Cosine-loss align (C)
- 在现有11K synth上跑2000步

**判断**：
- 如果mean inter-cosine < +0.40 且 UPREG↔DOWNREG < 0.7 → **成功**，进v7做k sweep [2, 8, 16]
- 如果crowding改善但F1没涨 → 走D（recursion）
- 如果crowding没改善 → 直接走F（generative）

### v6b（fallback）
- B+C only（不要A），看看是否`<answer>`聚合主导是主因
- 如果v6a和v6b同样crowded，说明A不是关键

---

## 数据层面的可能改善

如果方法改造都不够，回到数据：

1. **Longer reasoning**：当前teacher reasoning只30 char ≈ ~30 token，对compression任务无意义。重生成synth，要求Sonnet写200+ token reasoning。
2. **Direction-explicit contrast**：在synth pipeline中**显式对比** UPREG vs DOWNREG的样本对，加contrastive loss强迫推开。
3. **30K scale**：当前11K可能不够，按原计划扩展到30K。但若structural cap存在，scale不解决根本问题。

---

## 工程层面的TODO

- 修复 `tie_word_embeddings` 警告：当`modules_to_save=["embed_tokens"]`时，peft会warn需要`ensure_weight_tying`。不影响bi-encoder cosine eval，但若以后做generative需要保存model时要解决。
- Latent的device/dtype：当前latents和model.parameters()不在同一optimizer group更清晰：`optimizer = AdamW([{'params': model.parameters()}, {'params': latents.parameters(), 'lr': lr*5}])`，给latents更大lr。
- L_align显示为0.000的format问题：改成`{L_align:.6f}`，避免误以为loss不工作。
