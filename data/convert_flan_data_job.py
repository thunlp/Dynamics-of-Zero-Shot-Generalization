from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
# from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from IPython import embed

random.seed(0)
TRAIN_CNT_PER_TASK = 20

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
chosen_task = "merge_all"
method = "mt" # TODO: ot / mt / weighted
DATASET_SETTING = 'fine' # TODO: coarse / fine

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

with open('data/train_eval_key_list.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    train_key_list = data['train_key_list']
    eval_key_list = data['eval_key_list']

train_dataset, eval_dataset = [], []

# TODO:
if chosen_task == "merge" or chosen_task == "merge_all":
    save = "generalist"
    for key in train_key_list:
        # TODO:
        # train_data.extend(random.sample(task_list[key]['data'], min(TRAIN_CNT_PER_TASK, len(task_list[key]['data']))))
        train_dataset.extend(task_list[key]['data'])
    for key in eval_key_list:
        eval_dataset.extend(random.sample(task_list[key]['data'], min(5, len(task_list[key]['data']))))
else:
    save = "specialist"
    for key in train_key_list:
        train_dataset.extend(random.sample(task_list[key]['data'], min(TRAIN_CNT_PER_TASK, len(task_list[key]['data']))))
    assert chosen_task in eval_key_list
    eval_dataset.extend(random.sample(task_list[chosen_task]['data'], min(100, len(task_list[chosen_task]['data']))))


def trim_data(example):
    # This function will be applied to each example in the dataset
    example['data'] = example['data'][:2]  # Keep only the first two items
    return example

for data in train_dataset:
    data = trim_data(data)

for data in eval_dataset:
    data = trim_data(data)


# TODO: embedding begin

embed_model = SentenceTransformer('path/to/all-MiniLM-L6-v2').cuda()
print("begin train embedding")
train_encoded = embed_model.encode([f"{data['data'][0]} {data['data'][1]}" for data in tqdm(train_dataset)])
print("begin eval embedding")
eval_encoded = embed_model.encode([f"{data['data'][0]} {data['data'][1]}" for data in tqdm(eval_dataset)])
# cos_distances = -cosine_similarity(np.array(train_encoded), np.array(eval_encoded))

print("begin calculating similarity")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embeddings_a = torch.tensor(train_encoded).to(device)
embeddings_b = torch.tensor(eval_encoded).to(device)
# 计算余弦相似度
cos_sim = torch.zeros((embeddings_a.shape[0], embeddings_b.shape[0]), device=device)
for i, emb_a in tqdm(enumerate(embeddings_a)):
    cos_sim[i] = F.cosine_similarity(emb_a.unsqueeze(0), embeddings_b)
cos_distances = -cos_sim.cpu().numpy()

save_prefix = f'/data/checkpoints/datasets2/{save}/flan_mini_{chosen_task.replace("/", "_")}_{DATASET_SETTING}_{method}'
# np.save(f"{save_prefix}/flan_{chosen_task}_minus_cos_distance", cos_distances)

# TODO: embedding end

def select_min_data(distances, num):
    indexes = []
    max_value = np.max(distances) + 1
    round = 0
    while len(indexes) < min(num, distances.shape[0]):
        min_indices = np.argmin(distances, axis=0)
        unique_min_indices = np.unique(min_indices)
        indexes.extend(unique_min_indices)
        distances[unique_min_indices, :] = max_value
        round += 1
        print(round, len(indexes))
    return indexes

def select_max_data(distances, num):
    indexes = []
    min_value = np.min(distances) - 1
    round = 0
    while len(indexes) < min(num, distances.shape[0]):
        min_indices = np.argmax(distances, axis=0)
        unique_min_indices = np.unique(min_indices)
        indexes.extend(unique_min_indices)
        distances[unique_min_indices, :] = min_value
        round += 1
        print(round, len(indexes))
    return indexes


distances: np.array = cos_distances
print(f"Distances shape: {distances.shape}")

# min_values = distances.min(axis=1)
# avg_values = distances.mean(axis=1)
# avg_weight = min_values.mean() / avg_values.mean()
# weighted_value = min_values + avg_weight * avg_values
# indices = min_values.argsort()
# df = dataset.to_pandas()
# df = df.reindex(indices)
# dataset = Dataset.from_pandas(df)  
# dataset = dataset.remove_columns(['__index_level_0__'])
# ids = []
# for i in range(len(dataset)):
#     if i < 15000 or i >= len(dataset)-15000:
#         ids.append(i)

ori_distances = distances.copy()
max_ids = select_max_data(ori_distances, 30000)
new_ids = []
for id in max_ids:
    if not id in new_ids:
        new_ids.append(id)
print(f"for cos_earthmax_30k, length of new ids: {len(new_ids)}")
sampled_dataset = [train_dataset[id] for id in new_ids]
dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame(sampled_dataset)),
    'test': Dataset.from_pandas(pd.DataFrame(eval_dataset))
})
dataset_dict.save_to_disk(f'{save_prefix}/cos_earth_max_30k')
print("for cos_earthmax_30k, First...")
print(dataset_dict['train'][0])

print("for cos_earthmax_30k, Last...")
print(dataset_dict['train'][-1])

print("*" * 50)

ori_distances = distances.copy()
min_ids = select_min_data(ori_distances, 30000)
new_ids = []
for id in min_ids:
    if not id in new_ids:
        new_ids.append(id)
print(f"for cos_earthmin_30k, length of new ids: {len(new_ids)}")
sampled_dataset = [train_dataset[id] for id in new_ids]
dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame(sampled_dataset)),
    'test': Dataset.from_pandas(pd.DataFrame(eval_dataset))
})
dataset_dict.save_to_disk(f'{save_prefix}/cos_earth_min_30k')
print("for cos_earthmin_30k, First...")
print(dataset_dict['train'][0])

print("for cos_earthmin_30k, Last...")
print(dataset_dict['train'][-1])

print("*" * 50)


ori_distances = distances.copy()
max_ids = select_max_data(ori_distances, 15000)
ori_distances = distances.copy()
min_ids = select_min_data(ori_distances, 15000)
ori_max_ids = max_ids.copy()
ori_min_ids = min_ids.copy()

max_ids.extend(min_ids[::-1])
new_ids = []
for id in max_ids:
    if not id in new_ids:
        new_ids.append(id)
print(f"for cos_earthmax-min_30k, length of new ids: {len(new_ids)}")
sampled_dataset = [train_dataset[id] for id in new_ids]
dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame(sampled_dataset)),
    'test': Dataset.from_pandas(pd.DataFrame(eval_dataset))
})
dataset_dict.save_to_disk(f'{save_prefix}/cos_earth_max-min_30k')
print("for cos_earthmax-min_30k, First...")
print(dataset_dict['train'][0])

print("for cos_earthmax-min_30k, Last...")
print(dataset_dict['train'][-1])

print("*" * 50)

ori_min_ids.extend(ori_max_ids[::-1])
new_ids = []
for id in ori_min_ids:
    if not id in new_ids:
        new_ids.append(id)
print(f"for cos_earthmin-max_30k, length of new ids: {len(new_ids)}")
sampled_dataset = [train_dataset[id] for id in new_ids]
dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame(sampled_dataset)),
    'test': Dataset.from_pandas(pd.DataFrame(eval_dataset))
})
dataset_dict.save_to_disk(f'{save_prefix}/cos_earth_min-max_30k')
print("for cos_earthmin-max_30k, First...")
print(dataset_dict['train'][0])

print("for cos_earthmin-max_30k, Last...")
print(dataset_dict['train'][-1])

print("*" * 50)

random.seed(0)
random.shuffle(new_ids)
print(f"for cos_earth_random_30k, length of new ids: {len(new_ids)}")
sampled_dataset = [train_dataset[id] for id in new_ids]
dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame(sampled_dataset)),
    'test': Dataset.from_pandas(pd.DataFrame(eval_dataset))
})
dataset_dict.save_to_disk(f'{save_prefix}/cos_earth_random_30k')
print("for cos_earth_random_30k, First...")
print(dataset_dict['train'][0])

print("for cos_earth_random_30k, Last...")
print(dataset_dict['train'][-1])
