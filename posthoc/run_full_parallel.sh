#!/bin/bash
# Run two GPU instances in parallel:
#   GPU 0: BioRED dev + test (variant D's training distribution)
#   GPU 1: 3 OOD pubtator files (cdr, disgenet, pharmgkb)
#
# Each vLLM instance loads its own copy of Qwen3-8B-Base + variant D LoRA.
# Memory: 2 x ~17GB → fits 2x 24GB cards comfortably.
#
# After both finish, run stage 2 eval on all 5 score JSONs.

set -e
cd "$(dirname "$0")/.."

VARIANT_DIR=${VARIANT_DIR:-results/biored_finetune/D/checkpoint}
MODEL_PATH=${MODEL_PATH:-/root/gpufree-data/Qwen3-8B-Base}
OUT_DIR=${OUT_DIR:-posthoc/results}
DATA_DIR=${DATA_DIR:-./dataset/biored}
DEV_FILE=${DEV_FILE:-processed_test.pubtator}
TEST_FILE=${TEST_FILE:-processed_bc8_test.pubtator}
OOD_FILES="./dataset/Biomedical/processed/cdr.pubtator,./dataset/Biomedical/processed/disgenet.pubtator,./dataset/Biomedical/processed/pharmgkb.pubtator"

mkdir -p "$OUT_DIR"
mkdir -p posthoc/logs

source .venv/bin/activate

LIMIT_DOCS=${LIMIT_DOCS:-0}
EXTRA_ARGS=""
[ "$LIMIT_DOCS" -gt 0 ] && EXTRA_ARGS="--limit_docs $LIMIT_DOCS"

echo "=========================================="
echo "Stage 1 PARALLEL: GPU0 → BioRED, GPU1 → OOD"
echo "  variant: $VARIANT_DIR"
echo "  output:  $OUT_DIR"
echo "  limit:   $LIMIT_DOCS"
echo "=========================================="
echo

# --- GPU 0: BioRED dev + test ---
echo "[GPU 0] start: dev=$DEV_FILE  test=$TEST_FILE"
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
nohup python posthoc/score_pairs.py \
  --data_dir "$DATA_DIR" \
  --dev_file "$DEV_FILE" \
  --test_file "$TEST_FILE" \
  --model_name_or_path "$MODEL_PATH" \
  --variant_dir "$VARIANT_DIR" \
  --output_dir "$OUT_DIR" \
  $EXTRA_ARGS \
  > posthoc/logs/gpu0_biored.log 2>&1 &
PID0=$!
echo "[GPU 0] PID=$PID0  log=posthoc/logs/gpu0_biored.log"

# --- GPU 1: OOD ---
echo "[GPU 1] start: $OOD_FILES"
CUDA_VISIBLE_DEVICES=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
nohup python posthoc/score_pairs.py \
  --data_dir "$DATA_DIR" \
  --ood_test_files "$OOD_FILES" \
  --model_name_or_path "$MODEL_PATH" \
  --variant_dir "$VARIANT_DIR" \
  --output_dir "$OUT_DIR" \
  $EXTRA_ARGS \
  > posthoc/logs/gpu1_ood.log 2>&1 &
PID1=$!
echo "[GPU 1] PID=$PID1  log=posthoc/logs/gpu1_ood.log"

echo
echo "Waiting for both jobs..."
wait $PID0
RC0=$?
wait $PID1
RC1=$?
echo "[GPU 0] exited rc=$RC0"
echo "[GPU 1] exited rc=$RC1"

if [ $RC0 -ne 0 ] || [ $RC1 -ne 0 ]; then
  echo "ERROR: at least one stage-1 job failed"
  exit 1
fi

echo
echo "=========================================="
echo "Stage 2: Post-hoc calibration"
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
import json, glob
from pathlib import Path
rows = []
for f in sorted(glob.glob('$OUT_DIR/*_adjusted.json')):
  d = json.load(open(f))
  name = Path(f).stem.replace('_adjusted', '')
  base_f1 = d['baseline']['micro']['f1']
  best_m, best_f1 = 'baseline', base_f1
  for m, v in d.items():
    if m == 'baseline' or 'micro' not in v: continue
    if v['micro']['f1'] > best_f1:
      best_m, best_f1 = m, v['micro']['f1']
  rows.append((name, base_f1, best_f1, best_m, best_f1 - base_f1))
print(f\"{'dataset':<34} {'base F1':>10} {'best F1':>10} {'method':<18} {'Δ':>8}\")
print('-' * 86)
for r in rows:
  print(f'{r[0]:<34} {r[1]:>10.4f} {r[2]:>10.4f} {r[3]:<18} {r[4]:>+8.4f}')
"
