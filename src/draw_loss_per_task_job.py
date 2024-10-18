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
dataset_setting = 'optimal' # round_robin / cluster / random
# task050_multirc_answerability
# task103_facts2story_long_text_generation
# apps
# duorc_SelfRC_decide_worth_it
# quoref_What_Is_The_Answer
# wiki_bio_key_content
task = "task050_multirc_answerability"
STRIDE = 5
metrics_dir_path = f'/data/checkpoints/loss_{task}/results_{dataset_name}_{dataset_setting}_{seed}'
save_pic_path = f'/data/checkpoints/loss_{task}/results_{dataset_name}_{dataset_setting}_{seed}/'

MAX_STEPS = 0
if dataset_name == 'flan':
    MAX_STEPS = 900
elif dataset_name == 'p3':
    MAX_STEPS = 1780
elif dataset_name == 'ni':
    MAX_STEPS = 2000
elif dataset_name == 'ultrachat':
    MAX_STEPS = 1880


def get_num_of_examples():
    path = os.path.join(metrics_dir_path, 'step_0/test_results.json')
    with open(path, 'r', encoding='utf-8') as f:
        test_results_scores_data = json.load(f)
    return len(test_results_scores_data['data'])


x = []

num_of_examples = get_num_of_examples()
test_loss_task_scores = {}
test_loss_scores = []

# TODO
for step in tqdm(range(0, MAX_STEPS, STRIDE)):
    step_dir = os.path.join(metrics_dir_path, f'step_{step}')
    if not os.path.exists(os.path.join(step_dir, 'test_results.json')): continue
    x.append(step)

    with open(os.path.join(step_dir, 'test_results.json'), 'r', encoding='utf-8') as f:
        test_results_scores_data = json.load(f)
    
    test_loss_score = 0.0
    test_loss_task_score = {}
    for idx, data in enumerate(test_results_scores_data['data']):
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
    test_loss_scores.append(test_loss_score)
    
    for k, v in test_loss_task_score.items():
        if k not in test_loss_task_scores:
            test_loss_task_scores[k] = []
        test_loss_task_scores[k].append(v['loss'] / v['cnt'])


def plot_abs_and_delta(x, data, title: str=""):
    delta_data = [
        data[i + 1] - data[i] 
        for i in range(len(data) - 1)
    ]
    # TODO: without label
    # plt.plot(x[:-1], delta_data)
    plt.plot(x, data)
    # TODO: with label
    # plt.plot(x[:-1], delta_data, label=f'Delta Test {title} Score')
    # plt.plot(x, data, label=f'{title}')


for k, v in test_loss_task_scores.items():
    plt.figure(figsize=(10, 5))
    plot_abs_and_delta(x, v)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f"TASK: {k}")
    plt.grid()
    # Save the plot as an image
    save_path = os.path.join(save_pic_path, f'loss_pics_per_task_{STRIDE}')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'loss_{k.replace("/", "_")}_{dataset_setting}.png'))
