#!/bin/bash
# BioRED Ch3 ablation runner.
#
# Runs four variants serially (each uses both GPUs via DDP), then shuts the
# machine down to save GPU rental cost. Each variant logs to its own file
# under results/biored_finetune/<tag>/run.log.
#
# Usage:
#   bash scripts/llm_biored_ablation.sh             # full matrix + shutdown
#   AUTO_SHUTDOWN=False bash scripts/llm_biored_ablation.sh   # skip shutdown
#   RUN_ONLY=A bash scripts/llm_biored_ablation.sh  # only variant A
#
# Tip: launch in tmux/nohup so an SSH drop doesn't abort the matrix.

set -u  # error on undefined vars; intentionally NOT set -e so one bad
        # variant doesn't skip the rest (and the shutdown)

CUDA_DEVICE=${CUDA_DEVICE:-0,1}
DATA_DIR=${DATA_DIR:-./dataset/biored}
MODEL_PATH=${MODEL_PATH:-/root/gpufree-data/Qwen3-8B-Base}
PREPRO_TOKENIZER_PATH=${PREPRO_TOKENIZER_PATH:-bert-base-uncased}
TRAIN_FILE=${TRAIN_FILE:-processed_train_dev.pubtator}
DEV_FILE=${DEV_FILE:-processed_test.pubtator}
TEST_FILE=${TEST_FILE:-processed_bc8_test.pubtator}
SEED=${SEED:-66}
EXTRA_DATASETS=${EXTRA_DATASETS:-drugprot,ddi}
AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-True}
RUN_ONLY=${RUN_ONLY:-}  # "" = all; or e.g. "A" / "B" / "B,C" to restrict
RESULT_ROOT=${RESULT_ROOT:-./results/biored_finetune}

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
NUM_GPUS=$(echo $CUDA_DEVICE | tr ',' '\n' | wc -l)

# --- variant definitions ---
# Each variant: tag, extra-flags-for-train
declare -A VARIANTS=(
  [A]=""
  [B]="--use_extra_training_datasets --extra_datasets ${EXTRA_DATASETS}"
  [C]="--use_extra_training_datasets --extra_datasets ${EXTRA_DATASETS} --loss_reweight"
  [D]="--loss_reweight"
)
declare -A VARIANT_DESC=(
  [A]="baseline (BioRED only, new prompt, no reweight)"
  [B]="+multi-dataset (BioRED + DrugProt + DDI)"
  [C]="+multi-dataset +loss_reweight"
  [D]="baseline +loss_reweight only"
)
ORDER=(A B C D)

# Filter by RUN_ONLY
if [ -n "$RUN_ONLY" ]; then
  IFS=',' read -ra FILTER <<< "$RUN_ONLY"
  NEW_ORDER=()
  for v in "${ORDER[@]}"; do
    for f in "${FILTER[@]}"; do
      if [ "$v" = "$f" ]; then NEW_ORDER+=("$v"); fi
    done
  done
  ORDER=("${NEW_ORDER[@]}")
fi

echo "================================================================"
echo "BioRED Ch3 ablation matrix"
echo "GPUs:        $CUDA_DEVICE ($NUM_GPUS devices)"
echo "Model:       $MODEL_PATH"
echo "Variants:    ${ORDER[*]}"
echo "Auto-shutdown after run: $AUTO_SHUTDOWN"
echo "================================================================"

run_variant() {
  local tag=$1
  local extra=$2
  local desc="${VARIANT_DESC[$tag]}"
  local out_dir="${RESULT_ROOT}/${tag}"
  mkdir -p "$out_dir"
  local logf="${out_dir}/run.log"

  echo ""
  echo "----------------------------------------------------------------"
  echo "VARIANT $tag: $desc"
  echo "  extra flags: $extra"
  echo "  output:      $out_dir"
  echo "  log:         $logf"
  echo "  start time:  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "----------------------------------------------------------------"

  # Activate venv inside the subshell so torchrun picks the right python
  source .venv/bin/activate

  # --- TRAIN ---
  torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    train_llm.py \
    --data_dir $DATA_DIR \
    --model_name_or_path $MODEL_PATH \
    --prepro_tokenizer_path $PREPRO_TOKENIZER_PATH \
    --train_file $TRAIN_FILE \
    --dev_file $DEV_FILE \
    --test_file $TEST_FILE \
    --seed $SEED \
    --result_save_path $out_dir \
    $extra \
    --phase 1 2>&1 | tee -a "$logf"
  local train_ec=${PIPESTATUS[0]}

  if [ $train_ec -ne 0 ]; then
    echo "VARIANT $tag TRAIN FAILED (exit $train_ec), skipping eval" | tee -a "$logf"
    return $train_ec
  fi

  # --- TEST on BioRED dev + BC8 ---
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python test_llm.py \
    --data_dir $DATA_DIR \
    --model_name_or_path $MODEL_PATH \
    --prepro_tokenizer_path $PREPRO_TOKENIZER_PATH \
    --train_file $TRAIN_FILE \
    --dev_file $DEV_FILE \
    --test_file $TEST_FILE \
    --seed $SEED \
    --result_save_path $out_dir \
    --phase 1 2>&1 | tee -a "$logf"

  echo "VARIANT $tag finished at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$logf"
}

START_TS=$(date +%s)
for variant in "${ORDER[@]}"; do
  run_variant "$variant" "${VARIANTS[$variant]}"
done
END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))

echo ""
echo "================================================================"
echo "All variants done. Total wall time: $((DURATION/3600))h $(((DURATION%3600)/60))m"
echo "Per-variant logs under: $RESULT_ROOT/<tag>/run.log"
echo "================================================================"

if [ "$AUTO_SHUTDOWN" = "True" ]; then
  echo "Auto-shutdown in 60s. Press Ctrl-C to cancel."
  sleep 60
  /usr/sbin/poweroff
fi
