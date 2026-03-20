CUDA_DEVICE=0
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
USE_WMSS=False
RESULT_PATH="./results/biored_finetune/llm_no_direction_all_datasets"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_llm.py \
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
$(if [ "$USE_WMSS" = "True" ]; then echo "--use_wmss"; fi) \
--phase 1

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