#!/bin/bash
# BioRED LLM training with DDP for full multi-GPU utilization
# Automatically detects available GPUs and launches with torchrun

CUDA_DEVICE=0,1
DATA_DIR="./dataset/biored"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="$(realpath "${SCRIPT_DIR}/../../base_models/Meta-Llama-3.1-8B-Instruct")"
TRAIN_FILE="processed_train_dev.pubtator"
DEV_FILE="processed_test.pubtator"
TEST_FILE="processed_bc8_test.pubtator"
SEED=66
USE_DIRECTION=False
USE_AUGMENTED_TRAINING=False
USE_EXTRA_TRAINING_DATASETS=False
RESULT_PATH="./results/biored_finetune/llm_no_direction"

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
--train_file $TRAIN_FILE \
--dev_file $DEV_FILE \
--test_file $TEST_FILE \
--seed $SEED \
$(if [ "$USE_DIRECTION" = "True" ]; then echo "--use_direction"; fi) \
$(if [ "$USE_AUGMENTED_TRAINING" = "True" ]; then echo "--use_augmented_training"; fi) \
--result_save_path $RESULT_PATH \
$(if [ "$USE_EXTRA_TRAINING_DATASETS" = "True" ]; then echo "--use_extra_training_datasets"; fi) \
--phase 1

# Testing
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python test_llm.py \
--data_dir $DATA_DIR \
--model_name_or_path $MODEL_PATH \
--train_file $TRAIN_FILE \
--dev_file $DEV_FILE \
--test_file $TEST_FILE \
--seed $SEED \
$(if [ "$USE_DIRECTION" = "True" ]; then echo "--use_direction"; fi) \
--result_save_path $RESULT_PATH \
--phase 1