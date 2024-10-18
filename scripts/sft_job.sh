GPUS_PER_NODE=2

cd ModelCenter
pip install -r requirements.txt
pip install -e .

pip install wandb
pip install nltk
pip install datasets
pip install bmtrain --upgrade
pip install IPython
pip install transformers==4.35.2

cd ..
echo "=============================================================================="
ls

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $WORLD_SIZE \
                  --node_rank $RANK \
                  --master_addr $MASTER_ENDPOINT \
                  --master_port $MASTER_PORT"

# wiki_bio_key_content
# quoref_What_Is_The_Answer
# task1047_pib_translation_english_telugu
# duorc_SelfRC_decide_worth_it
# apps
# task050_multirc_answerability
# super_glue_multirc:1.0.2
# task103_facts2story_long_text_generation
# task851_synthetic_multiply_evens
# merge
# merge_per_task
# ultrachat
METHOD="mt" # TODO: ot / mt / weighted
TASK="ultrachat_200k" # TODO
SUBSET="min_4000"
SELECT="uniform" # TODO: polar / uniform
DATASET_NAME="ultrachat" # TODO: p3 / flan / ni / ultrachat
DATASET_SETTING="random" # TODO: NFT / FFT / RT / max-min / min-max / max / min / random

BASE_PATH="/data"
CKPT_BASE_DIRECTORY="/data/checkpoints/TWO2/${TASK}_${METHOD}" # TODO

OPTS=""
# dataset config
# OPTS+=" --data_dir /data/checkpoints/datasets2/generalist/flan_mini_${TASK}_${SUBSET}_coarse_${METHOD}"
# OPTS+=" --data_dir /data/checkpoints/datasets2/generalist/flan_mini_${TASK}_${SUBSET}_fine_${METHOD}/cos_earth${DATASET_SETTING}_30k"
OPTS+=" --data_dir /data/checkpoints/datasets2/generalist/${TASK}_optimal_${METHOD}/cos_earth_${DATASET_SETTING}_30k" # TODO
# OPTS+=" --data_dir /data/checkpoints/datasets/generalist/ultrachat_single_optimal_optimal_${METHOD}" # TODO
# OPTS+=" --data_dir /data/checkpoints/datasets/generalist/flan_mini_${TASK}_optimal_${SELECT}_weighted" # TODO
# OPTS+=" --data_dir /data/checkpoints/datasets/generalist/flan_mini_${TASK}_optimal_${METHOD}" # TODO
# OPTS+=" --data_dir /data/checkpoints/datasets/specialist/flan_mini_${TASK}_optimal_${METHOD}" # TODO
# OPTS+=" --data_dir ${BASE_PATH}/datasets/dynamic_data/flan_mini_${TASK}_optimal_weighted" # TODO
# OPTS+=" --data_dir ${BASE_PATH}/datasets/merged_P3_train_small"
# OPTS+=" --data_dir ${BASE_PATH}/datasets/natural-instructions/splits/default"

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
OPTS+=" --batch_size_per_device 8" # TODO
OPTS+=" --save_step 10" # TODO
OPTS+=" --epochs 1"
OPTS+=" --lr 1e-6"
# OPTS+=" --train_iters 1000"
OPTS+=" --warmup_iters 0"
OPTS+=" --start_step 0"
OPTS+=" --loss_scale 6400"
OPTS+=" --save_dir ${CKPT_BASE_DIRECTORY}/checkpoints_${DATASET_NAME}_${DATASET_SETTING}/" # TODO
OPTS+=" --save_name pytorch_model"

CMD="torchrun ${DISTRIBUTED_ARGS} src/sft_job.py ${OPTS}"

echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"

${CMD}

