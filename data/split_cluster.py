import json
import random
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_from_disk
import pandas as pd
from IPython import embed

random.seed(0)

# TODO: 保证训练集和测试集的任务不同
TRAIN_CNT_PER_TASK = 20
TEST_CNT_PER_TASK = 5
DATASET_SETTING = 'random' # cluster / round_robin / random

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
dataset = json.load(open('path/to/flan_mini.json', "r"))
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

cnt = 0
for k, v in task_list.items():
    if v['cnt'] < TRAIN_CNT_PER_TASK:
        cnt += 1

print("cnt: %d" % cnt)

with open('train_eval_key_list.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    train_key_list = data['train_key_list']
    eval_key_list = data['eval_key_list']

train_data, eval_data = [], []

if DATASET_SETTING == 'round_robin':
    for _ in range(TRAIN_CNT_PER_TASK):
        for key in train_key_list:
            task_info = task_list[key]
            if task_info['cnt'] > 0:
                train_data.append(task_info['data'].pop(0))
                task_info['cnt'] -= 1
    
    for key in eval_key_list:
        eval_data.extend(task_list[key]['data'])

elif DATASET_SETTING == 'cluster':
    for key in train_key_list:
        train_data.extend(task_list[key]['data'][:TRAIN_CNT_PER_TASK])
        
    for key in eval_key_list:
        eval_data.extend(task_list[key]['data'])

elif DATASET_SETTING == 'random':
    for key in train_key_list:
        train_data.extend(task_list[key]['data'][:TRAIN_CNT_PER_TASK])
        
    for key in eval_key_list:
        eval_data.extend(task_list[key]['data'])
    
    random.shuffle(train_data)

else:
    raise ValueError(f"Unsupported dataset setting {DATASET_SETTING}")


print(f"Training dataset size: {len(train_data)}")
print(f"Testing dataset size: {len(eval_data)}")

dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame(train_data)),
    'test': Dataset.from_pandas(pd.DataFrame(eval_data))
})
dataset_dict.save_to_disk(f'path/to/save/flan_mini_{DATASET_SETTING}')

new_dataset = load_from_disk(f'path/to/save/flan_mini_{DATASET_SETTING}')

embed()
