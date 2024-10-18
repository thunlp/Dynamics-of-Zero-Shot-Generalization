import json
import random
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_from_disk
import os
import numpy as np
import pandas as pd
from IPython import embed
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sentence_transformers import SentenceTransformer

random.seed(0)
TRAIN_CNT_PER_TASK = 20
TEST_CNT_PER_TASK = 5 # TODO

# TODO
DATASET_SETTING = 'optimal' # cluster / round_robin / optimal

model = SentenceTransformer('path/to/all-MiniLM-L6-v2').cuda()
data_path = "path/to/flan_mini.json"
save_prefix = "/data/checkpoints/datasets/generalist"

dataset = json.load(open(data_path, "r"))
task_list = {}
for data in tqdm(dataset):
    task_name = data['source']
    if task_name in task_list:
        # if task_list[task_name]['cnt'] >= 3: continue
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

with open('data/train_eval_key_list.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    train_key_list = data['train_key_list']
    eval_key_list = data['eval_key_list']

polar_train_dataset, uniform_train_dataset, eval_data = [], [], []

# sample 1k test set first
for key in eval_key_list:
    eval_data.extend(random.sample(task_list[key]['data'], min(TEST_CNT_PER_TASK, len(task_list[key]['data']))))

eval_encoded = [model.encode(f"{data['data'][0]} {data['data'][1]}") for data in tqdm(eval_data)]
eval_encoded = np.array(eval_encoded)


def sample(train_data):
    train_encoded = [model.encode(f"{data['data'][0]} {data['data'][1]}") for data in tqdm(train_data)]

    distances = -cosine_similarity(np.array(train_encoded), np.array(eval_encoded)) # (28751, TEST_CNT_PER_TASK)
    min_ = np.min(distances, axis=1) # (28751,)
    avg_ = np.mean(distances, axis=1) # (28751,)

    # ===================================== hyper-parameter ==================================
    min_weight = 1
    avg_weight = min_.mean() / avg_.mean()
    # ===================================== hyper-parameter ==================================

    distances = min_weight * min_ + avg_weight * avg_
    sorted_indices = np.argsort(distances)

    train_data = [train_data[i] for i in sorted_indices] # reordered based on the distance to test data
    return train_data


def sample_one_task(key):
    # print(f"Processing {key}... Length: {len(task_list[key]['data'])}")

    train_encoded = [model.encode(f"{data['data'][0]} {data['data'][1]}") for data in tqdm(task_list[key]['data'])]
    
    distances = -cosine_similarity(np.array(train_encoded), eval_encoded) # (length_of_task_data, 1k)
    # print("distances shape: ", distances.shape)
    
    min_ = np.min(distances, axis=1)
    avg_ = np.mean(distances, axis=1)
    
    # ===================================== hyper-parameter ==================================
    min_weight = 1
    # TODO
    # avg_weight = 0
    avg_weight = min_.mean() / avg_.mean()
    # ===================================== hyper-parameter ==================================
    
    distances = min_weight * min_ + avg_weight * avg_
    sorted_indices = np.argsort(distances)
    
    train_data = [task_list[key]['data'][i] for i in sorted_indices]
    
    if len(train_data) < TRAIN_CNT_PER_TASK: return train_data, train_data
    
    # ============================================== polar ===============================================

    half = TRAIN_CNT_PER_TASK // 2
    polar_train_data = train_data[:half] + train_data[-half:]

    # ============================================== uniform ===============================================

    interval_size = len(train_data) // TRAIN_CNT_PER_TASK
    uniform_train_data = [train_data[i] for i in range(0, len(train_data), interval_size)]
    
    return polar_train_data, uniform_train_data


print(f"length eval set: {len(eval_data)}")

for key in tqdm(train_key_list):
    polar_train_data, uniform_train_data = sample_one_task(key)
    print(f"{key} | Polar: {len(polar_train_data)} | Uniform: {len(uniform_train_data)}")
    
    polar_train_dataset.extend(polar_train_data)
    uniform_train_dataset.extend(uniform_train_data)

print(f"length polar training set: {len(polar_train_dataset)}")
print(f"length uniform training set: {len(uniform_train_dataset)}")
print(f"length eval set: {len(eval_data)}")

polar_train_dataset = sample(list(polar_train_dataset))
uniform_train_dataset = sample(list(uniform_train_dataset))

polar_dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame(polar_train_dataset)),
    'test': Dataset.from_pandas(pd.DataFrame(eval_data))
})

uniform_dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame(uniform_train_dataset)),
    'test': Dataset.from_pandas(pd.DataFrame(eval_data))
})

# TODO
polar_dataset_dict.save_to_disk(os.path.join(save_prefix, f'flan_mini_merge_per_task_{DATASET_SETTING}_polar_weighted'))
uniform_dataset_dict.save_to_disk(os.path.join(save_prefix, f'flan_mini_merge_per_task_{DATASET_SETTING}_uniform_weighted'))

new_polar_dataset = load_from_disk(os.path.join(save_prefix, f'flan_mini_merge_per_task_{DATASET_SETTING}_polar_weighted'))
new_uniform_dataset = load_from_disk(os.path.join(save_prefix, f'flan_mini_merge_per_task_{DATASET_SETTING}_uniform_weighted'))

print(f"Polar First...")
print(new_polar_dataset['train'][0])

print(f"Polar Last...")
print(new_polar_dataset['train'][-1])

print(f"Uniform First...")
print(new_uniform_dataset['train'][0])

print(f"Uniform Last...")
print(new_uniform_dataset['train'][-1])
