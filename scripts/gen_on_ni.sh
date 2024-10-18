#! /bin/bash
# only one gpu!
export CUDA_VISIBLE_DEVICES=2  # TODO

MASTER_ADDR=localhost
MASTER_PORT=12346   # TODO
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

DATASET_SETTING="random" # TODO: random / round_robin / cluster / optimal
BASE_PATH="/data"
PROJECT_PATH="${BASE_PATH}/dynamics_of_zero_shot_generalization"
CKPT_DIRECTORY="${PROJECT_PATH}/ckpts/ni/checkpoints_ni_${DATASET_SETTING}"
CKPT_STRIDE=10  # the stride for saving ckpts

# Use a for loop to iterate through subdirectories
# GPU-0: [0, 990]    port: 12346
# GPU-1: [1000, 1990] port: 12347
# TODO
for ((global_step = 180; global_step <= 180; global_step += CKPT_STRIDE)); do

    subdirectory_name="step_${global_step}"
    echo ${subdirectory_name}
    OPTS=""
    # dataset config
    OPTS+=" --data_dir ${BASE_PATH}/datasets/natural-instructions/splits/default"
    OPTS+=" --task_dir ${BASE_PATH}/datasets/natural-instructions/tasks"
    OPTS+=" --max_num_instances_per_task 100"
    OPTS+=" --max_num_instances_per_eval_task 100"
    OPTS+=" --num_pos_examples 2"
    OPTS+=" --num_neg_examples 0"
    OPTS+=" --add_task_definition"
    OPTS+=" --max_source_length 1024"
    OPTS+=" --max_target_length 128"
    OPTS+=" --max_train_samples 0"
    OPTS+=" --max_test_samples 120" # TODO
    # model config
    OPTS+=" --model_name_or_path ${BASE_PATH}/model_weights/llama-2-7b"
    OPTS+=" --load_ckpt ${CKPT_DIRECTORY}/${subdirectory_name}"
    # other config
    OPTS+=" --batch_size_per_device 8"
    # OPTS+=" --tensorboard ${PROJECT_PATH}/tensorboard_eval/${subdirectory_name}"`date +"%Y%m%d%H%M%S"`
    OPTS+=" --cache_dir ${PROJECT_PATH}/cache/"
    OPTS+=" --output_dir ${PROJECT_PATH}/results/rm/results_ni/${subdirectory_name}"

    CMD="torchrun ${DISTRIBUTED_ARGS} ${PROJECT_PATH}/src/gen_on_ni.py ${OPTS}"

    echo "-------final CMD is------"
    echo "${CMD}"
    echo "-------final CMD end------"

    ${CMD} 2>&1 | tee ${PROJECT_PATH}/logs/$subdirectory_name-gen_on_ni.log
done