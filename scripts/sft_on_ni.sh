#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1

MASTER_ADDR=localhost
MASTER_PORT=12345
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=2

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="/data"
PROJECT_PATH="${BASE_PATH}/dynamics_of_zero_shot_generalization"

OPTS=""
# dataset config

DATASET_NAME="ni"
DATASET_SETTING="random" # TODO: random / round_robin / cluster / optimal
OPTS+=" --data_dir ${BASE_PATH}/datasets/natural-instructions/splits/default"

OPTS+=" --dataset_name ${DATASET_NAME}"
OPTS+=" --dataset_setting ${DATASET_SETTING}"

# for super ni dataset
OPTS+=" --task_dir ${BASE_PATH}/datasets/natural-instructions/tasks"
OPTS+=" --max_num_instances_per_task 100"
OPTS+=" --max_num_instances_per_eval_task 100"
OPTS+=" --num_pos_examples 2"
OPTS+=" --num_neg_examples 0"
OPTS+=" --add_task_definition"

OPTS+=" --max_source_length 1024"
OPTS+=" --max_target_length 128"
# OPTS+=" --max_train_samples 30000" # TODO, for ni
# model config
OPTS+=" --model_name_or_path ${BASE_PATH}/model_weights/llama-2-7b"
# training config
OPTS+=" --logging_step 5" 
OPTS+=" --batch_size_per_device 8"
OPTS+=" --save_step 10"
OPTS+=" --epochs 1"
OPTS+=" --lr 1e-6"
# OPTS+=" --train_iters 1000"
OPTS+=" --warmup_iters 0"
OPTS+=" --start_step 0"
OPTS+=" --loss_scale 6400"
OPTS+=" --tensorboard ${PROJECT_PATH}/tensorboard_sft/"`date +"%Y%m%d%H%M%S"`
OPTS+=" --save_dir ${PROJECT_PATH}/ckpts/ni/checkpoints_${DATASET_NAME}_${DATASET_SETTING}/" # TODO
OPTS+=" --cache_dir ${PROJECT_PATH}/cache/"

CMD="torchrun ${DISTRIBUTED_ARGS} ${PROJECT_PATH}/src/sft_on_ni.py ${OPTS}"

echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"

${CMD} 2>&1 | tee ${PROJECT_PATH}/logs/finetune_on_${DATASET_NAME}_${DATASET_SETTING}.log
