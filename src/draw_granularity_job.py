import os
import json
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import embed

# for rouge score and RM score
# TODO
seed = 0
dataset_name = 'flan'
dataset_settings = ['_coarse', '']

Task = "merge"
subset = "min_4000"
method = "weighted"
STRIDE = 10
save_pic_path = f'/data/checkpoints/TWO2/{Task}_{subset}_coarse_{method}/'

MAX_STEPS = 1800

x = []

test_loss_scores = {}
test_loss_task_scores = {}

# TODO
for step in tqdm(range(0, MAX_STEPS, STRIDE)):

    x.append(step)
    
    for dataset_setting in dataset_settings:
        metrics_dir_path = f'/data/checkpoints/TWO2/{Task}_{subset}{dataset_setting}_{method}/loss_{dataset_name}'

        step_dir = os.path.join(f"{metrics_dir_path}_NFT_{method}_{seed}", f'step_{step}') # TODO
        with open(os.path.join(step_dir, 'test_results.json'), 'r', encoding='utf-8') as f:
            test_results_scores_data = json.load(f)

        test_loss_score = 0.0
        test_loss_task_score = {}
        if dataset_setting not in test_loss_task_scores:
            test_loss_task_scores[dataset_setting] = {}
            test_loss_scores[dataset_setting] = []
        for data in test_results_scores_data['data']:
            # strip NAN
            if math.isnan(data['loss']):
                data['loss'] = 0
            task = data['task']

            if task not in test_loss_task_score:
                test_loss_task_score[task] = {"loss": 0.0, "cnt": 0}
            test_loss_task_score[task]['loss'] += data['loss']
            test_loss_task_score[task]['cnt'] += 1

            test_loss_score += data['loss']
        
        test_loss_score /= len(test_results_scores_data['data'])
        test_loss_scores[dataset_setting].append(test_loss_score)

        for k, v in test_loss_task_score.items():
            if k not in test_loss_task_scores[dataset_setting]:
                test_loss_task_scores[dataset_setting][k] = []
            test_loss_task_scores[dataset_setting][k].append(v['loss'] / v['cnt'])


#BEB8DC
colors = ["#82B0D2", "#FFBE7A", "#FA7F6F", "#8ECFC9", "#BEB8DC"]
maps = {"": "fine-grained", "_coarse": "coarse-grained"}

plt.figure(figsize=(12, 6))
for idx, (dataset_setting, v) in enumerate(test_loss_scores.items()):
    plt.plot(x, v, label=maps[dataset_setting], color=colors[idx])
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.title(f"Averaged Test Loss (Seed = {seed}) on All Unseen Tasks")
plt.grid()
plt.xticks(range(0, max(x) + 101, 100), rotation=45)
# Save the plot as an image
save_path = os.path.join(save_pic_path, f'loss_pics_per_task_optimal_{method}_{seed}')
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, 'loss_merge.png'))

for k in test_loss_task_scores[dataset_settings[0]].keys():
    plt.figure(figsize=(12, 6))
    for idx, (dataset_setting, v) in enumerate(test_loss_task_scores.items()):
        v = v[k]
        plt.plot(x, v, label=maps[dataset_setting], color=colors[idx])
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f"Test Loss (Seed = {seed}) on Unseen Task: {k}")
    plt.grid()
    plt.xticks(range(0, max(x) + 101, 100), rotation=45)
    # Save the plot as an image
    save_path = os.path.join(save_pic_path, f'loss_pics_per_task_optimal_{method}_{seed}')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'loss_{k.replace("/", "_")}.png'))

