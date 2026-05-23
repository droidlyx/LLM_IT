# Synth training data (Quest 023 v1)

由Claude Sonnet（via Claude Code Agent dispatch）在PubMed未标注passage上生成的多schema biomedical RE合成数据。**生成成本不可忽视，避免重复生成**。

## 三轮数据规模

| 版本 | 样本数 | 用途 | 解压后 |
|---|---|---|---|
| training_v2 | 1,666 | 早期smoke验证pipeline | ~3 MB |
| training_v3 | 5,442 | smoke v3（500步训练，BioRED full-RE 0.099 raw / 0.158+τ_cal） | ~10 MB |
| training_v4 | 11,265 | smoke v4（1000步 → BioRED 0.201，BioTriplex 0.024）和 smoke v5（CODI 2000步） | ~21 MB |

## 文件说明

每个目录包含：
- `synth_train.jsonl.gz` — gzipped合成训练数据。每行JSON记录字段：
  - `pmid`, `doc` (passage), `e1_text/e1_type/e1_id`, `e2_text/e2_type/e2_id`, `label`, `reasoning`
- `label_dict.json` — 27个统一标签 × 5个template/label
- `mean.pt` — **仅适用于Quest 022 v4单token架构**的global centering mean。CODI-Bi (Quest 023 v2 / v5) 需要用 `scripts/compute_mean.py` 重新计算，因为pool位置变了

## 解压使用

```bash
cd quest_023_codi/data/training_v4
gunzip -k synth_train.jsonl.gz   # -k保留.gz文件
# 或不解压，直接读：
python -c "import gzip, json; [print(json.loads(l)['label']) for l in gzip.open('synth_train.jsonl.gz', 'rt')]" | head
```

训练脚本（`scripts/train_codi.py`、`scripts/smoke_tiny.py`）当前从 `/tmp/synth_smoke/training_v4/synth_train.jsonl` 读，新容器需要解压到该路径，或修改`DATA`常量指向branch内位置。

## 标签分布（v4）

27个标签覆盖6个schema：
- DrugProt/ChemProt（13）：INHIBITOR, DIRECT-REGULATOR, SUBSTRATE, ACTIVATOR, INDIRECT-UPREGULATOR, INDIRECT-DOWNREGULATOR, ANTAGONIST, PRODUCT-OF, PART-OF, AGONIST, SUBSTRATE_PRODUCT-OF, AGONIST-INHIBITOR, AGONIST-ACTIVATOR (note: 实际是13个)
- BioRED（8）：Association, Positive_Correlation, Negative_Correlation, Bind, Cotreatment, Comparison, Drug_Interaction, Conversion
- DDI（4）：effect, mechanism, advise, int
- GDA（3）：Biomarker, Therapeutic, *Negative→已映射到no_relation*
- CDR（1）：CID
- PharmGKB（1）：PharmGKB_Association
- + no_relation

## Teacher quality caveat

诊断显示Sonnet对负样本的FP率约39%（用100条真no_relation + 113条positive）。real F1（不区分positive/negative）只有~0.523。也就是说teacher本身有相当的噪音 — 学生模型F1能否超越teacher是开放问题。
