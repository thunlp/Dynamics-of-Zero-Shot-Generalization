GPUS_PER_NODE=1

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

BASE_PATH="/data"

SEED=0 # TODO
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
# ultrachat
# merge_per_task
# merge_ot
METHOD="weighted" # TODO: ot / mt / weighted
TASK="merge"
SUBSET="min_4000"
DATASET_SETTING="NFT" # TODO: NFT / FFT / RT / max-min / min-max / max / min / _random
DATASET_NAME="flan"    # TODO: p3   / flan / ni / ultrachat / ultrachat
MIDDLE_STEPS=900     # TODO: 900  / 900  / 1000 / 940       / 960
MAX_STEPS=1800       # TODO: 1780 / 1800 / 2000 / 1880      / 1920

# TODO
CKPT_DIRECTORY="/data/checkpoints/TWO2/${TASK}_${SUBSET}_coarse_${METHOD}/checkpoints_${DATASET_NAME}_${DATASET_SETTING}"
CKPT_STRIDE=10  # TODO: the stride for saving ckpts

# TODO
for ((global_step = 1000; global_step < MAX_STEPS; global_step += CKPT_STRIDE)); do

    subdirectory_name="step_${global_step}"
    echo ${subdirectory_name}

    OPTS=""

    OPTS+=" --seed ${SEED}"

    # dataset config
    # TODO
    OPTS+=" --data_dir /data/checkpoints/datasets2/generalist/flan_mini_${TASK}_${SUBSET}_coarse_${METHOD}"
    # OPTS+=" --data_dir /data/checkpoints/datasets2/generalist/flan_mini_${TASK}_${SUBSET}_fine_${METHOD}/cos_earth${DATASET_SETTING}_30k" # TODO
    # OPTS+=" --data_dir /data/checkpoints/datasets2/generalist/${TASK}_optimal_${METHOD}/cos_earth_${DATASET_SETTING}_30k" # TODO
    # OPTS+=" --data_dir /data/checkpoints/datasets/specialist/flan_mini_${TASK}_optimal_${METHOD}" # TODO
    # OPTS+=" --data_dir /data/checkpoints/datasets/generalist/ultrachat_single_optimal_optimal_${METHOD}" # TODO
    # OPTS+=" --data_dir /data/checkpoints/datasets/generalist/flan_mini_${TASK}_optimal_${METHOD}" # TODO
    # OPTS+=" --data_dir ${BASE_PATH}/datasets/dynamic_data/flan_mini_${TASK}_optimal_weighted"  # with task name
    # OPTS+=" --data_dir ${BASE_PATH}/datasets/merged_P3_eval_small"
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
    OPTS+=" --max_test_samples 120" # TODO
    OPTS+=" --max_test_samples_per_task 5" # TODO | flan: 100 * 5; ni: 119 * 5

    # model config
    OPTS+=" --model_name_or_path ${BASE_PATH}/model_weights/llama-2-7b"
    # TODO:
    OPTS+=" --output_dir /data/checkpoints/TWO2/${TASK}_${SUBSET}_coarse_${METHOD}/loss_${DATASET_NAME}_${DATASET_SETTING}_${METHOD}_${SEED}/${subdirectory_name}"
    OPTS+=" --batch_size_per_device 16"
    # OPTS+=" --cache_dir ${PROJECT_PATH}/cache/"

    OPTS+=" --load_ckpt ${CKPT_DIRECTORY}/${subdirectory_name}"

    CMD="torchrun ${DISTRIBUTED_ARGS} src/eval_by_loss_job.py ${OPTS}"

    echo "-------final CMD is------"
    echo "${CMD}"
    echo "-------final CMD end------"

    ${CMD}

done

