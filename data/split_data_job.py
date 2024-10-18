import ot
import json
import random
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_from_disk
import numpy as np
import pandas as pd
from IPython import embed
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sentence_transformers import SentenceTransformer


def pW_cal(a, b, p=2, metric='euclidean'):
    """ Args:
            a, b: samples sets drawn from α, β respectively
            p: the coefficient in the OT cost (i.e., the p in p-Wasserstein)
            metric: the metric to compute cost matrix, 'euclidean' or 'cosine'
    """
    # cost matrix
    M = ot.dist(a, b, metric=metric)
    M = pow(M, p)

    # uniform distribution assumption
    alpha = ot.unif(len(a))
    beta = ot.unif(len(b))
    
    pW = ot.emd(alpha, beta, M, numItermax=100000)
    return np.sum(pW * M, axis=1)

    # p-Wasserstein Distance
    pW = ot.emd2(alpha, beta, M, numItermax=100000)
    pW = pow(pW, 1/p)
    
    return pW


random.seed(0)

# TODO: 保证训练集和测试集的任务不同
TRAIN_CNT_PER_TASK = 20
HELD_OUT_TASK_CNT = 225

# TODO
# wiki_bio_key_content
# quoref_What_Is_The_Answer
# task1047_pib_translation_english_telugu
# duorc_SelfRC_decide_worth_it
# apps
# task050_multirc_answerability
# super_glue/multirc:1.0.2
# task103_facts2story_long_text_generation
# task851_synthetic_multiply_evens
# merge
chosen_task = "merge"
method = "weighted" # TODO: mt / ot / weighted
DATASET_SETTING = 'fine' # TODO: coarse / fine

"""
{
    "id": 'identity_flan_583804',
    "source": 'task061_ropes_answer_generation',
    "conversations": [
        {"from": "human", "value": "Q: ...Question: ..Sandra or Michelle?\nA:"},
        {"from": "gpt", "value': "Michelle."},
        ...... 
    ]
}
"""
data_path = "path/to/flan_mini.json"
dataset = json.load(open(data_path, "r"))
task_list = {}
for data in tqdm(dataset):
    task_name = data['source']
    if task_name in task_list:
        task_list[task_name]['cnt'] += 1
        task_list[task_name]['data'].append({
                "data": [sentence['value'].strip() for sentence in data['conversations']],
                "task": data['source']
            })
    else:
        task_list[task_name] = {
            "cnt": 1,
            "data": [{
                "data": [sentence['value'].strip() for sentence in data['conversations']],
                "task": data['source']
            }]
        }


# TODO
with open('data/train_eval_key_list.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    train_key_list = data['train_key_list']
    eval_key_list = data['eval_key_list']

train_data, eval_data = [], []


for key in train_key_list:
    train_data.extend(random.sample(task_list[key]['data'], min(TRAIN_CNT_PER_TASK, len(task_list[key]['data']))))

# TODO:
if chosen_task == "merge" or chosen_task == "merge_all":
    save = "generalist"
    for key in eval_key_list:
        eval_data.extend(random.sample(task_list[key]['data'], min(5, len(task_list[key]['data']))))
else:
    save = "specialist"
    assert chosen_task in eval_key_list
    eval_data.extend(random.sample(task_list[chosen_task]['data'], min(100, len(task_list[chosen_task]['data']))))

model = SentenceTransformer('path/to/all-MiniLM-L6-v2')
model.cuda()

"""
给定某一个泛化任务的测试数据集 eval_data，找出特定的训练数据集分布，使得在这个分布的训练集上训出来的一系列 ckpt，在这个测试集上的 loss 呈现漂亮的泛化曲线
"""
# encoding the train / test dataset
train_encoded = [model.encode(' '.join(data['data'])) for data in tqdm(train_data)]
eval_encoded = [model.encode(' '.join(data['data'])) for data in tqdm(eval_data)]

if method == "ot":
    distances = pW_cal(np.array(train_encoded), np.array(eval_encoded), p=1)
    # distances = [pW_cal(np.array([d]), np.array(eval_encoded)) for d in train_encoded]

elif method == "weighted":
    distances = -cosine_similarity(np.array(train_encoded), np.array(eval_encoded)) # (28751, TEST_CNT_PER_TASK)
    print("distances shape: ", distances.shape)
    min_ = np.min(distances, axis=1) # (28751,)
    avg_ = np.mean(distances, axis=1) # (28751,)

    # ===================================== hyper-parameter ==================================
    min_weight = 1
    # TODO
    # avg_weight = 0
    avg_weight = min_.mean() / avg_.mean()
    # ===================================== hyper-parameter ==================================

    distances = min_weight * min_ + avg_weight * avg_

else:
    raise ValueError(f"Unsupported method {method}")

sorted_indices = np.argsort(distances)

assert len(train_data) == sorted_indices.shape[0]
train_data = [train_data[i] for i in sorted_indices] # reordered based on the distance to test data

dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame(train_data)),
    'test': Dataset.from_pandas(pd.DataFrame(eval_data))
})

dataset_dict.save_to_disk(f'/data/checkpoints/datasets2/{save}/flan_mini_{chosen_task.replace("/", "_")}_{DATASET_SETTING}_{method}')
new_dataset = load_from_disk(f'/data/checkpoints/datasets2/{save}/flan_mini_{chosen_task.replace("/", "_")}_{DATASET_SETTING}_{method}')

print("First...")
print(new_dataset['train'][0])

print("Last...")
print(new_dataset['train'][-1])
