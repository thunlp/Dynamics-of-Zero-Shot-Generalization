#! /bin/bash
export CUDA_VISIBLE_DEVICES=1 # TODO

MASTER_ADDR=localhost
MASTER_PORT=12361 # TODO
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="/data"

SEED=0 # TODO
DATASET_SETTING="NFT" # TODO: NFT / FFT / RT / round_robin / cluster / random
DATASET_NAME="ultrachat"    # TODO: p3   / flan / ni / ultrachat
MIDDLE_STEPS=940     # TODO: 890  / 900  / 900 / 940
MAX_STEPS=1880       # TODO: 1780 / 1800 / 1800 / 1880

PROJECT_PATH="${BASE_PATH}/dynamics_of_zero_shot_generalization"
# TODO
CKPT_DIRECTORY="${PROJECT_PATH}/ckpts/${DATASET_NAME}/checkpoints_${DATASET_NAME}_${DATASET_SETTING}"
CKPT_STRIDE=10  # the stride for saving ckpts

# TODO
for ((global_step = 0; global_step < MIDDLE_STEPS; global_step += CKPT_STRIDE)); do

    subdirectory_name="step_${global_step}"
    echo ${subdirectory_name}

    OPTS=""

    OPTS+=" --seed ${SEED}"

    # dataset config
    # TODO
    OPTS+=" --data_dir ${BASE_PATH}/datasets/ultrachat_optimal_weighted"
    # OPTS+=" --data_dir ${BASE_PATH}/datasets/flan_mini_${DATASET_SETTING}"  # with task name
    # OPTS+=" --data_dir ${BASE_PATH}/datasets/P3_small"
    # OPTS+=" --data_dir ${BASE_PATH}/datasets/natural-instructions/splits/default"  # with task name

    OPTS+=" --dataset_name ${DATASET_NAME}"

    # for super ni dataset
    OPTS+=" --task_dir ${BASE_PATH}/datasets/natural-instructions/tasks"
    OPTS+=" --max_num_instances_per_task 100"
    OPTS+=" --max_num_instances_per_eval_task 100"
    OPTS+=" --num_pos_examples 2"
    OPTS+=" --num_neg_examples 0"
    OPTS+=" --add_task_definition"

    OPTS+=" --max_source_length 1024"
    OPTS+=" --max_target_length 128"
    OPTS+=" --max_test_samples 120" # TODO (generalist)
    OPTS+=" --max_test_samples_per_task 5" # TODO | flan: 100 * 5; ni: 119 * 5 (specialist)

    # model config
    OPTS+=" --model_name_or_path ${BASE_PATH}/model_weights/llama-2-7b"
    # TODO generalist or specialist
    OPTS+=" --output_dir ${PROJECT_PATH}/results/loss/generalist/results_${DATASET_NAME}_${DATASET_SETTING}_${SEED}/${subdirectory_name}"
    OPTS+=" --batch_size_per_device 8"
    OPTS+=" --cache_dir ${PROJECT_PATH}/cache/"

    OPTS+=" --tensorboard ${PROJECT_PATH}/tensorboard_eval_by_loss/"`date +"%Y%m%d%H%M%S"`

    OPTS+=" --load_ckpt ${CKPT_DIRECTORY}/${subdirectory_name}"

    CMD="torchrun ${DISTRIBUTED_ARGS} ${PROJECT_PATH}/src/eval_by_loss.py ${OPTS}"

    echo "-------final CMD is------"
    echo "${CMD}"
    echo "-------final CMD end------"

    ${CMD} 2>&1 | tee ${PROJECT_PATH}/logs/eval_by_loss_on_${DATASET_NAME}_${DATASET_SETTING}_${SEED}.log

done
