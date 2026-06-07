#!/bin/bash
# Phase 1: post-hoc calibration (P2P / LA / PAS / TECP) on variant D.
#
# Stage 1 (GPU): score every (pair, label) on dev + test + 3 OOD sets.
#                outputs posthoc/results/<dataset>_scores.json
# Stage 2 (CPU): apply 5 methods, print before/after F1, save report JSON.
#                outputs posthoc/results/<dataset>_adjusted.json
#
# Usage:
#   bash posthoc/run_phase1.sh                  # full pipeline (needs GPU for stage 1)
#   bash posthoc/run_phase1.sh score_only       # skip eval
#   bash posthoc/run_phase1.sh eval_only        # skip scoring, use existing JSONs
#   bash posthoc/run_phase1.sh smoke            # 5-doc smoke test on dev

set -e
cd "$(dirname "$0")/.."

VARIANT_DIR=${VARIANT_DIR:-results/biored_finetune/D/checkpoint}
MODEL_PATH=${MODEL_PATH:-base_models/Qwen3-8B-Base}
OUT_DIR=${OUT_DIR:-posthoc/results}
DATA_DIR=${DATA_DIR:-./dataset/Biomedical/processed}
OOD_FILES="./dataset/Biomedical/cdr.pubtator,./dataset/Biomedical/disgenet.pubtator,./dataset/Biomedical/pharmgkb.pubtator"

mkdir -p "$OUT_DIR"

MODE=${1:-full}
LIMIT_DOCS=0
[ "$MODE" = "smoke" ] && LIMIT_DOCS=5

source .venv/bin/activate

if [ "$MODE" != "eval_only" ]; then
  echo "=========================================="
  echo "Stage 1: Score pairs (variant D, vLLM)"
  echo "  variant   : $VARIANT_DIR"
  echo "  output    : $OUT_DIR"
  echo "  limit_docs: $LIMIT_DOCS"
  echo "=========================================="

  EXTRA_ARGS=""
  [ "$LIMIT_DOCS" -gt 0 ] && EXTRA_ARGS="--limit_docs $LIMIT_DOCS"

  # Dev + test (BioRED)
  python posthoc/score_pairs.py \
    --data_dir "$DATA_DIR" \
    --dev_file processed_train_dev.json \
    --test_file processed_test.json \
    --model_name_or_path "$MODEL_PATH" \
    --variant_dir "$VARIANT_DIR" \
    --output_dir "$OUT_DIR" \
    $EXTRA_ARGS

  # OOD datasets (cdr/disgenet/pharmgkb) — each uses its native rel_list
  python posthoc/score_pairs.py \
    --data_dir "$DATA_DIR" \
    --ood_test_files "$OOD_FILES" \
    --model_name_or_path "$MODEL_PATH" \
    --variant_dir "$VARIANT_DIR" \
    --output_dir "$OUT_DIR" \
    $EXTRA_ARGS
fi

if [ "$MODE" != "score_only" ]; then
  echo
  echo "=========================================="
  echo "Stage 2: Post-hoc calibration (P2P / LA / PAS / TECP)"
  echo "=========================================="

  for json in "$OUT_DIR"/*_scores.json; do
    [ -f "$json" ] || continue
    echo
    echo "----- $(basename "$json") -----"
    python posthoc/eval_adjusted.py \
      --scores_json "$json" \
      --report_dir "$OUT_DIR"
  done

  echo
  echo "=========================================="
  echo "Summary table"
  echo "=========================================="
  python -c "
import json, glob, os
from pathlib import Path
rows = []
for f in sorted(glob.glob('$OUT_DIR/*_adjusted.json')):
  d = json.load(open(f))
  name = Path(f).stem.replace('_adjusted', '')
  base_f1 = d['baseline']['micro']['f1']
  best_method, best_f1 = 'baseline', base_f1
  for m, v in d.items():
    if m == 'baseline': continue
    if 'micro' not in v: continue
    if v['micro']['f1'] > best_f1:
      best_method, best_f1 = m, v['micro']['f1']
  delta = best_f1 - base_f1
  rows.append((name, base_f1, best_f1, best_method, delta))
print(f\"{'dataset':<32} {'base F1':>10} {'best F1':>10} {'method':<18} {'Δ':>8}\")
print('-' * 84)
for r in rows:
  print(f'{r[0]:<32} {r[1]:>10.4f} {r[2]:>10.4f} {r[3]:<18} {r[4]:>+8.4f}')
"
fi
