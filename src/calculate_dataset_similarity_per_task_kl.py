import os
import json
import numpy as np
from tqdm import tqdm
from IPython import embed
from scipy.special import rel_entr
from collections import Counter
import matplotlib.pyplot as plt
from datasets import load_from_disk


def get_dataset(setting: str):
    assert setting in ['random', 'round_robin', 'cluster']
    data_dir = f"path/to/flan_mini_{setting}"
    dataset = load_from_disk(data_dir)
    return dataset['train'], dataset['test']


def get_ngrams(sentence, n):
    words = sentence.split()
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def compute_ngram_distribution(dataset, n):
    ngrams = []
    for sentence in dataset:
        ngrams.extend(get_ngrams(sentence, n))
    ngram_counts = Counter(ngrams)
    total_ngrams = sum(ngram_counts.values())
    ngram_distribution = {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
    return total_ngrams, ngram_distribution


def kl_divergence(p, q, epsilon=1e-10):
    # Add epsilon to each probability in q to avoid zero values
    q_smoothed = q + epsilon
    kl_div = np.sum(p * np.log(np.clip(p / q_smoothed, epsilon, None)))
    return kl_div


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
CALC_METHOD = 'kl'
max_test_samples_per_task = 5
SEED = 2 # TODO: 0 / 1 / 2
all_test_dataset = all_test_dataset.shuffle(seed=SEED)

for data in all_test_dataset:
    if data['task'] not in task_cnt_dict:
        test_dataset[data['task']] = []
        task_cnt_dict[data['task']] = 0
    if task_cnt_dict[data['task']] < max_test_samples_per_task:
        task_cnt_dict[data['task']] += 1
        test_dataset[data['task']].append(data)


metrics = {}
# Define the n-gram order
n = 2

print("Processing test dataset...")
for task in tqdm(test_dataset):
    test_dataset[task] = [f"{data['data'][0]} {data['data'][1]}" for data in test_dataset[task]]


def calc(train_dataset, setting: str):
    global test_dataset

    # step-1: concatenate
    train_dataset = [f"{data['data'][0]} {data['data'][1]}" for data in train_dataset]
    
    # step-2: get all-ngrams
    print(f"[{setting}] Getting n-grams...")
    all_ngrams = []
    for sentence in tqdm(train_dataset):
        all_ngrams.extend(get_ngrams(sentence, n))
    for task in tqdm(test_dataset):
        for sentence in test_dataset[task]:
            all_ngrams.extend(get_ngrams(sentence, n))
    
    # step-3: preprocess test dataset distributions
    print(f"[{setting}] Preprocessing test dataset distributions...")
    q = {}
    for task in tqdm(test_dataset):
        if task not in metrics:
            metrics[task] = {"kl": {}}
        if setting not in metrics[task]['kl']:
            metrics[task]['kl'][setting] = []
        _, tmp = compute_ngram_distribution(test_dataset[task], n)
        q[task] = np.array([tmp.get(ngram, 0) for ngram in all_ngrams])
    
    # step-4: accumlate to calculate similarity score, 1ckpt ~ 10steps ~ 160 samples
    print(f"[{setting}] Calculating scores...")
    train_distribution = None
    len_train = len(train_dataset)
    for i in tqdm(range(160, len_train, 160)):
        _, train_distribution = compute_ngram_distribution(train_dataset[:i], n)
        # if train_distribution is None:
        #     total_ngrams, train_distribution = compute_ngram_distribution(train_dataset[:160], n)
        # else:
        #     total_ngrams, train_distribution = update_ngram_distribution(train_distribution, total_ngrams, train_dataset[i - 160:i], n)

        p = np.array([train_distribution.get(ngram, 0) for ngram in all_ngrams])
        
        for task in test_dataset:
            kl_dv = float(kl_divergence(p, q[task]))
            metrics[task]['kl'][setting].append(kl_dv)


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
        plt.title(f"Similarity Distance (KL Divergence) on Unseen Task: {task}")
        plt.grid()
        plt.xticks(range(0, max(steps) + 101, 100), rotation=45)
        # Save the plot as an image
        plt.savefig(os.path.join(save_path, f'{metric}.png'))