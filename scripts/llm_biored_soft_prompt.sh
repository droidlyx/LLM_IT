CUDA_DEVICE=2
DATA_DIR="./dataset/biored"
MODEL_PATH="../../base_models/Meta-Llama-3.1-8B-Instruct"
TRAIN_FILE="bioredirect_train_dev.pubtator"
DEV_FILE="bioredirect_test.pubtator"
TEST_FILE="bioredirect_bc8_test.pubtator"
SEED=66
USE_DIRECTION=False
RESULT_PATH="./results/biored_finetune/llm_no_direction_soft_prompt_first_lora"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_llm_soft_prompt.py \
--data_dir $DATA_DIR \
--model_name_or_path $MODEL_PATH \
--train_file $TRAIN_FILE \
--dev_file $DEV_FILE \
--test_file $TEST_FILE \
--seed $SEED \
--use_direction $USE_DIRECTION \
--result_save_path $RESULT_PATH \
--use_soft_prompt True \
--use_lora False \
--num_train_epochs 1.0 \

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_llm_soft_prompt.py \
--data_dir $DATA_DIR \
--model_name_or_path $MODEL_PATH \
--train_file $TRAIN_FILE \
--dev_file $DEV_FILE \
--test_file $TEST_FILE \
--seed $SEED \
--use_direction $USE_DIRECTION \
--result_save_path $RESULT_PATH \
--use_soft_prompt True \
--use_lora True \
--num_train_epochs 2.0 \

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python test_llm_soft_prompt.py \
--data_dir $DATA_DIR \
--model_name_or_path $MODEL_PATH \
--train_file $TRAIN_FILE \
--dev_file $DEV_FILE \
--test_file $TEST_FILE \
--seed $SEED \
--use_direction $USE_DIRECTION \
--result_save_path $RESULT_PATH \
--use_soft_prompt True \