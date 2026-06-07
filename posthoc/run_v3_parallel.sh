#!/bin/bash
# v3: multi-pair logprob scoring — matches deployment distribution.
set -e
cd "$(dirname "$0")/.."

VARIANT_DIR=${VARIANT_DIR:-results/biored_finetune/D/checkpoint}
MODEL_PATH=${MODEL_PATH:-/root/gpufree-data/Qwen3-8B-Base}
OUT_DIR=${OUT_DIR:-posthoc/v3_results}
DATA_DIR=${DATA_DIR:-./dataset/biored}
DEV_FILE=${DEV_FILE:-processed_test.pubtator}
TEST_FILE=${TEST_FILE:-processed_bc8_test.pubtator}
OOD_FILES="./dataset/Biomedical/processed/cdr.pubtator,./dataset/Biomedical/processed/disgenet.pubtator,./dataset/Biomedical/processed/pharmgkb.pubtator"

mkdir -p "$OUT_DIR" posthoc/logs
source .venv/bin/activate

LIMIT_DOCS=${LIMIT_DOCS:-0}
EXTRA_ARGS=""
[ "$LIMIT_DOCS" -gt 0 ] && EXTRA_ARGS="--limit_docs $LIMIT_DOCS"

echo "============================================"
echo "v3 parallel multi-pair: GPU0 BioRED, GPU1 OOD"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
nohup python posthoc/score_pairs_v3.py \
  --data_dir "$DATA_DIR" \
  --dev_file "$DEV_FILE" \
  --test_file "$TEST_FILE" \
  --model_name_or_path "$MODEL_PATH" \
  --variant_dir "$VARIANT_DIR" \
  --output_dir "$OUT_DIR" \
  $EXTRA_ARGS \
  > posthoc/logs/v3_gpu0_biored.log 2>&1 &
PID0=$!
echo "[GPU 0] PID=$PID0"

CUDA_VISIBLE_DEVICES=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
nohup python posthoc/score_pairs_v3.py \
  --data_dir "$DATA_DIR" \
  --ood_test_files "$OOD_FILES" \
  --model_name_or_path "$MODEL_PATH" \
  --variant_dir "$VARIANT_DIR" \
  --output_dir "$OUT_DIR" \
  $EXTRA_ARGS \
  > posthoc/logs/v3_gpu1_ood.log 2>&1 &
PID1=$!
echo "[GPU 1] PID=$PID1"

wait $PID0; RC0=$?
wait $PID1; RC1=$?
echo "[GPU 0] rc=$RC0"
echo "[GPU 1] rc=$RC1"

if [ $RC0 -ne 0 ] || [ $RC1 -ne 0 ]; then
  echo "ERROR"; exit 1
fi

echo
echo "==== Stage 2: eval_adjusted ===="
for json in "$OUT_DIR"/*_scores.json; do
  [ -f "$json" ] || continue
  echo "--- $(basename "$json") ---"
  python posthoc/eval_adjusted.py --scores_json "$json" --report_dir "$OUT_DIR"
done

echo
echo "==== Stage 3: SUMMARY.md ===="
python posthoc/summarize_results.py --results_dir "$OUT_DIR" --out "$OUT_DIR/SUMMARY.md"
