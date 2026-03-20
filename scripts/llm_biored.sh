CUDA_DEVICE=1
DATA_DIR="./dataset/biored"
MODEL_PATH="../../base_models/Meta-Llama-3.1-8B-Instruct"
TRAIN_FILE="processed_train_dev.pubtator"
DEV_FILE="processed_test.pubtator"
TEST_FILE="processed_bc8_test.pubtator"
SEED=66
USE_DIRECTION=False
USE_AUGMENTED_TRAINING=False
USE_EXTRA_TRAINING_DATASETS=True
RESULT_PATH="./results/biored_finetune/llm_no_direction_all_datasets"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_llm.py \
--data_dir $DATA_DIR \
--model_name_or_path $MODEL_PATH \
--train_file $TRAIN_FILE \
--dev_file $DEV_FILE \
--test_file $TEST_FILE \
--seed $SEED \
--use_direction $USE_DIRECTION \
--use_augmented_training $USE_AUGMENTED_TRAINING \
--result_save_path $RESULT_PATH \
--use_extra_training_datasets $USE_EXTRA_TRAINING_DATASETS \
--phase 1

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python test_llm.py \
--data_dir $DATA_DIR \
--model_name_or_path $MODEL_PATH \
--train_file $TRAIN_FILE \
--dev_file $DEV_FILE \
--test_file $TEST_FILE \
--seed $SEED \
--use_direction $USE_DIRECTION \
--result_save_path $RESULT_PATH \
--phase 1