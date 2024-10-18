import os
import json
import numpy as np
from tqdm import tqdm
from IPython import embed
import matplotlib.pyplot as plt
from datasets import load_from_disk
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sentence_transformers import SentenceTransformer

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def get_dataset(setting: str):
    assert setting in ['random', 'round_robin', 'cluster']
    data_dir = f"path/to/flan_mini_{setting}"
    dataset = load_from_disk(data_dir)
    return dataset['train'], dataset['test']


"""
{
    "data": [Q, A],
    "task": "task123_conala_sort_dictionary"
}
"""
print("Loading dataset...")
round_robin_train_dataset, all_test_dataset = get_dataset('round_robin')
random_train_dataset, _ = get_dataset('random')
cluster_train_dataset, _ = get_dataset('cluster')

# TODO: TMP
# round_robin_train_dataset = round_robin_train_dataset.select(range(500))
# cluster_train_dataset = cluster_train_dataset.select(range(500))
# random_train_dataset = random_train_dataset.select(range(500))

# TODO: TMP
# steps = [i for i in range(10, 40, 10)]
steps = [i for i in range(10, 1800, 10)]

task_cnt_dict = {}
test_dataset = {}
CALC_METHOD = "cosine" # TODO: euclidean / cosine
max_test_samples_per_task = 5
SEED = 0 # TODO: 0 / 1 / 2
all_test_dataset = all_test_dataset.shuffle(seed=SEED)

for data in all_test_dataset:
    if data['task'] not in task_cnt_dict:
        test_dataset[data['task']] = []
        task_cnt_dict[data['task']] = 0
    if task_cnt_dict[data['task']] < max_test_samples_per_task:
        task_cnt_dict[data['task']] += 1
        test_dataset[data['task']].append(data)


metrics = {}

model = SentenceTransformer('path/to/all-MiniLM-L6-v2')
model.cuda()

print("Processing test dataset...")
for key in tqdm(test_dataset):
    test_dataset[key] = [f"{data['data'][0]} {data['data'][1]}" for data in test_dataset[key]]
    test_dataset[key] = [model.encode(data) for data in test_dataset[key]]
    test_dataset[key] = np.array(test_dataset[key])


def calc(train_dataset, setting: str):
    global test_dataset

    # step-1: concatenate
    train_dataset = [f"{data['data'][0]} {data['data'][1]}" for data in train_dataset]
    
    # step-2: get embedding of all data points (sentence transformers)
    print(f"[{setting}] Generating embeddings...")
    train_dataset = [model.encode(data) for data in tqdm(train_dataset)]
    train_dataset = np.array(train_dataset)
    
    # step-3: accumlate to calculate similarity score, 1ckpt ~ 10steps ~ 160 samples
    print(f"[{setting}] Calculating scores...")
    
    for task in tqdm(test_dataset):

        if CALC_METHOD == "euclidean":
            distances = euclidean_distances(train_dataset, test_dataset[task])
        elif CALC_METHOD == "cosine":
            distances = -cosine_similarity(train_dataset, test_dataset[task])
        else:
            raise ValueError(f"Unsupported calculate method {CALC_METHOD}")

        len_train, len_test = len(train_dataset), len(test_dataset[task])

        # metrics
        distances_scores = {"avg": [], "min": [], "max": [], "centroid": []}

        for i in range(160, len_train, 160):
            centroid_train = np.mean(train_dataset[:i], axis=0)
            centroid_test = np.mean(test_dataset[task], axis=0)
            
            if CALC_METHOD == "euclidean":
                centroid_score = euclidean(centroid_train, centroid_test)
            elif CALC_METHOD == "cosine":
                centroid_train = centroid_train.reshape(1, -1)  # Reshape to make it a 2D array
                centroid_test = centroid_test.reshape(1, -1)
                centroid_score = -cosine_similarity(centroid_train, centroid_test)[0, 0]
            else:
                raise ValueError(f"Unsupported calculate method {CALC_METHOD}")
            # TODO:
            distances_scores['centroid'].append(float(centroid_score))
            distances_scores['max'].append(float(np.max(distances[:i])))
            distances_scores['avg'].append(float(np.mean(distances[:i])))
            distances_scores['min'].append(float(np.min(distances[:i])))
            

        if task not in metrics:
            metrics[task] = {}
        for metric in distances_scores:
            if metric not in metrics[task]:
                metrics[task][metric] = {}
            metrics[task][metric][setting] = distances_scores[metric]


calc(round_robin_train_dataset, "round_robin")
calc(random_train_dataset, "random")
calc(cluster_train_dataset, "cluster")


similarity_save_dir = f'path/to/similarity_{SEED}/{CALC_METHOD}'

colors = ["#82B0D2", "#FFBE7A", "#FA7F6F"]

for task in metrics:
    save_path = os.path.join(similarity_save_dir, task.replace("/", "_"))
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, 'distances.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(metrics[task], indent=4))

    for metric in metrics[task]:
        settings_scores = metrics[task][metric]
        plt.figure(figsize=(12, 6))
        for i, (setting, distances) in enumerate(settings_scores.items()):
            plt.plot(steps, distances, label=setting, color=colors[i])
        plt.xlabel('Steps')
        plt.ylabel('Similarity Distances')
        plt.legend()
        plt.title(f"Similarity Distance ({CALC_METHOD}-{metric}) on Unseen Task: {task}")
        plt.grid()
        plt.xticks(range(0, max(steps) + 101, 100), rotation=45)
        # Save the plot as an image
        plt.savefig(os.path.join(save_path, f'{metric}.png'))
