#!/bin/bash
# BioRED LLM training with DDP for full multi-GPU utilization
# Automatically detects available GPUs and launches with torchrun

CUDA_DEVICE=0,1
DATA_DIR="./dataset/biored"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Default to local Qwen3-8B-Base. Override by setting MODEL_PATH before invoking.
MODEL_PATH="${MODEL_PATH:-/root/gpufree-data/Qwen3-8B-Base}"
# Wordpiece tokenizer for entity-span construction (originally BiomedBERT).
# Fall back to bert-base-uncased from HF cache if BiomedBERT isn't local.
PREPRO_TOKENIZER_PATH="${PREPRO_TOKENIZER_PATH:-bert-base-uncased}"
TRAIN_FILE="processed_train_dev.pubtator"
DEV_FILE="processed_test.pubtator"
TEST_FILE="processed_bc8_test.pubtator"
SEED=66
USE_DIRECTION=False
USE_AUGMENTED_TRAINING=False
USE_EXTRA_TRAINING_DATASETS=False
LOSS_REWEIGHT=False
EXTRA_DATASETS="drugprot,ddi"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"  # 2048 fits 98% of BioRED queries; safe on RTX 4090
RESULT_PATH="${RESULT_PATH:-./results/biored_finetune/llm_no_direction}"

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Count available GPUs
NUM_GPUS=$(echo $CUDA_DEVICE | tr ',' '\n' | wc -l)

echo "============================================="
echo "BioRED LLM Training"
echo "GPUs: $CUDA_DEVICE (${NUM_GPUS} devices)"
echo "Result path: $RESULT_PATH"
echo "============================================="

# Training with torchrun for DDP
torchrun \
--nproc_per_node=${NUM_GPUS} \
--master_port=29500 \
train_llm.py \
--data_dir $DATA_DIR \
--model_name_or_path $MODEL_PATH \
--prepro_tokenizer_path $PREPRO_TOKENIZER_PATH \
--max_seq_length $MAX_SEQ_LENGTH \
--train_file $TRAIN_FILE \
--dev_file $DEV_FILE \
--test_file $TEST_FILE \
--seed $SEED \
$(if [ "$USE_DIRECTION" = "True" ]; then echo "--use_direction"; fi) \
$(if [ "$USE_AUGMENTED_TRAINING" = "True" ]; then echo "--use_augmented_training"; fi) \
--result_save_path $RESULT_PATH \
$(if [ "$USE_EXTRA_TRAINING_DATASETS" = "True" ]; then echo "--use_extra_training_datasets --extra_datasets ${EXTRA_DATASETS}"; fi) \
$(if [ "$LOSS_REWEIGHT" = "True" ]; then echo "--loss_reweight"; fi) \
--phase 1

# Testing
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python test_llm.py \
--data_dir $DATA_DIR \
--model_name_or_path $MODEL_PATH \
--prepro_tokenizer_path $PREPRO_TOKENIZER_PATH \
--max_seq_length $MAX_SEQ_LENGTH \
--train_file $TRAIN_FILE \
--dev_file $DEV_FILE \
--test_file $TEST_FILE \
--seed $SEED \
$(if [ "$USE_DIRECTION" = "True" ]; then echo "--use_direction"; fi) \
--result_save_path $RESULT_PATH \
--phase 1