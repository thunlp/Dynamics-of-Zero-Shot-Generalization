#! /bin/bash
# only one gpu!
export CUDA_VISIBLE_DEVICES=0  # TODO

MASTER_ADDR=localhost
MASTER_PORT=12351   # TODO
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

DATASET_NAME="p3" # TODO
BASE_PATH="/data"
PROJECT_PATH="${BASE_PATH}/dynamics_of_zero_shot_generalization"
HDFS_PATH="/data"

OPTS=""
OPTS+=" --model_name_or_path ${HDFS_PATH}/model_weights/llama-2-13b"
OPTS+=" --load_ckpt ${HDFS_PATH}/model_weights/UltraRM/ultrarm_final.pt"
OPTS+=" --data_dir ${PROJECT_PATH}/results/rm/results_${DATASET_NAME}"
OPTS+=" --max_seq_length 1024"
# OPTS+=" --logging_step 1"
# OPTS+=" --batch_size_per_device 16"
OPTS+=" --seed 0"

CMD="torchrun ${DISTRIBUTED_ARGS} ${PROJECT_PATH}/src/eval_by_rm.py ${OPTS}"

echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"

$CMD