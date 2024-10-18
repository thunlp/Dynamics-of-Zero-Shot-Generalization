import argparse
import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
import bmtrain as bmt
from functools import partial
import time
import os
import json
from model_center.model import Llama
from model_center.tokenizer import LlamaTokenizer
from functools import partial
from dataset_wrapper import PromptIterableDataset, collator
from compute_metrics import compute_metrics, compute_grouped_metrics
import wandb
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import logging
import shutil
import numpy as np
import math
from IPython import embed
from datasets import load_dataset, load_from_disk
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_tokenizer(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return tokenizer


def get_model(args):
    model = Llama.from_pretrained(args.model_name_or_path)
    if args.load_ckpt is not None:
        logger.info(f"loading model from {args.load_ckpt}")
        bmt.load(model, os.path.join(args.load_ckpt, "pytorch_model.pt"), strict=False)

    return model


def setup_model_and_tokenizer(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args)
    return tokenizer, model


def initialize():
    # get arguments
    parser = argparse.ArgumentParser("")
    
    # model training arguments
    parser.add_argument("--model_name_or_path", default='path/to/llama-2-7b')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--batch_size_per_device", default=2, type=int)
    parser.add_argument("--wandb", action="store_true")

    # data parameters
    parser.add_argument('--dataset_name', type=str, help='The dataset name, p3 or flan or ni')
    parser.add_argument('--data_dir', type=str, help='The directory for saving the dataset')
    parser.add_argument('--max_source_length', type=int, help='The maximum total input sequence length after tokenization.')
    parser.add_argument('--max_target_length', type=int, help='The maximum total sequence length for target text after tokenization.')
    parser.add_argument('--max_test_samples', type=int, help='The maximum number of testing samples')
    parser.add_argument('--max_test_samples_per_task', type=int, help='The maximum number of testing samples per task')

    parser.add_argument('--task_dir', type=str, help='The directory for saving the NaturalInstructions tasks json files.')
    parser.add_argument('--tk_instruct', action='store_true', help='whether to use instruction tokenization.')
    parser.add_argument('--add_task_name', action='store_true', help='whether to preappend task name before the task input.')
    parser.add_argument('--add_task_definition', action='store_true', help='whether to preappend task definition before the task input.')
    parser.add_argument('--num_pos_examples', type=int, help='number of in-context positive examples.')
    parser.add_argument('--num_neg_examples', type=int, help='number of in-context negative examples.')
    parser.add_argument('--add_explanation', action='store_true', help='whether to add explanation for both the postive examples and negtive examples.')
    parser.add_argument('--max_num_instances_per_task', type=int, help='The maximum number of instances we will consider for each training task.')
    parser.add_argument('--max_num_instances_per_eval_task', type=int, help='The maximum number of instances we will consider for each validation/test task.')

    parser.add_argument('--cache_dir', type=str, help='The directory for cache')
    parser.add_argument('--output_dir', type=str, help='The directory for output')

    parser.add_argument("--tensorboard", type=str, default=None, help="whether using tb")
    parser.add_argument("--load_ckpt", type=str, default=None, help="resumed ckpt")

    args = parser.parse_args()
    # init bmt 
    bmt.init_distributed(seed=args.seed, zero_level=3)
    
    # wandb
    if args.wandb and bmt.rank() == 0:
        wandb.init()

    return args


def load_raw_dataset(args):
    """ Get the dataset """
    # TODO:
    task_cnt_dict = {}
    chosen_dataset = []

    if args.dataset_name == "ni":
        dataset = load_dataset(
            "src/ni_dataset.py", 
            data_dir=args.data_dir, 
            task_dir=args.task_dir, 
            cache_dir=args.cache_dir,
            max_num_instances_per_task=args.max_num_instances_per_task,
            max_num_instances_per_eval_task=args.max_num_instances_per_eval_task
        )
        dataset = dataset['test']
        
        for data in dataset:
            if data['Task'] not in task_cnt_dict:
                task_cnt_dict[data['Task']] = 0
            if task_cnt_dict[data['Task']] < args.max_test_samples_per_task:
                task_cnt_dict[data['Task']] += 1
                chosen_dataset.append(data)

    elif args.dataset_name == "p3":
        dataset = load_from_disk(args.data_dir)
        dataset = dataset['train'] # eval dataset, different tasks with label

    elif args.dataset_name == "flan" or args.dataset_name == "ultrachat":
        dataset = load_from_disk(args.data_dir)
        dataset = dataset['test']
        # ======================== eval optimal setting: one task only ============================
        for data in dataset:
            chosen_dataset.append(data)
        # ======================== eval optimal setting end: one task only ============================

        # dataset = dataset.shuffle(seed=args.seed)
        
        # sample_task_list_path = '/data/temporal-analysis/results/loss/sample_task_list.txt'
        # with open(sample_task_list_path, 'r', encoding='utf-8') as f:
        #     sample_task_list = [line.strip() for line in f]

        # for data in dataset:
        #     # TODO:
        #     if data['task'] not in sample_task_list: continue
        #     # TODO END
        #     if data['task'] not in task_cnt_dict:
        #         task_cnt_dict[data['task']] = 0
        #     if task_cnt_dict[data['task']] < args.max_test_samples_per_task:
        #         task_cnt_dict[data['task']] += 1
        #         chosen_dataset.append(data)

    # if args.max_test_samples is not None:
    #     dataset = dataset.shuffle().select(range(args.max_test_samples))

    return chosen_dataset


def evaluate(args, tokenizer, model, dataset):
    # tensorboard
    if args.tensorboard is not None and bmt.rank() == 0:
        from torch.utils.tensorboard import SummaryWriter
        import distutils.version  # noqa: F401

        if not os.path.exists(args.tensorboard):
            os.makedirs(args.tensorboard)
        writer = SummaryWriter(log_dir=args.tensorboard)
    
    logger.info(f"total testing instance number: {len(dataset)}")

    # loss function
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    avg_time_recorder = bmt.utils.AverageRecorder()
    avg_loss_recorder = bmt.utils.AverageRecorder()
    train_start_time = time.time()
    global_step = 0

    dataset = PromptIterableDataset(
        dataset, 
        tokenizer=tokenizer, 
        max_seq_length=args.max_seq_length, 
        dataset_name=args.dataset_name,
        teacher_forcing=True, 
        truncate_method="tail",
        # SuperNI config
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        add_task_name=args.add_task_name,
        add_task_definition=args.add_task_definition,
        num_pos_examples=args.num_pos_examples,
        num_neg_examples=args.num_neg_examples,
        add_explanation=args.add_explanation,
        tk_instruct=args.tk_instruct
    )

    logger.info("wrapping up data")
    bs = args.batch_size_per_device
    dataloader = DataLoader(dataset, batch_size=bs, collate_fn=partial(collator, tokenizer))

    progress_bar = tqdm(range(len(dataloader)), disable=not bmt.rank()==0)
    test_results = []
    
    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            st = time.time()

            with bmt.inspect.inspect_tensor() as inspector:
                for k in inputs:
                    inputs[k] = inputs[k].cuda()
                labels = inputs.pop("labels")
                
                logits = model(**inputs).logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                # shift_logits = shift_logits.view(-1, len(tokenizer))
                # shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                # loss = loss_func(shift_logits, shift_labels)

                predicted_indices = torch.argmax(shift_logits, dim=-1)
                predicted_indices[shift_labels == -100] = 1
                preds = tokenizer.batch_decode(predicted_indices.tolist(), skip_special_tokens=True)

                prompts = inputs['input_ids']
                prompts[labels != -100] = 1
                prompts = tokenizer.batch_decode(prompts.tolist(), skip_special_tokens=True)

                labels[labels == -100] = 1
                labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

                losses = []
                for i in range(shift_logits.shape[0]):
                    loss_i = loss_func(shift_logits[i], shift_labels[i]).item()
                    losses.append(loss_i)
                if args.dataset_name == "ni":
                    tasks = [data['Task'] for data in dataset.raw_dataset[bs * step: bs * step + shift_logits.shape[0]]]
                elif args.dataset_name == "flan":
                    tasks = [data['task'] for data in dataset.raw_dataset[bs * step: bs * step + shift_logits.shape[0]]]
                elif args.dataset_name == "ultrachat":
                    tasks = ["ultrachat" for _ in dataset.raw_dataset[bs * step: bs * step + shift_logits.shape[0]]]
                test_results.extend([{
                    "prompt": prompt,
                    "pred": pred,
                    "label": label,
                    "loss": loss,
                    "task": task,
                } for prompt, pred, label, loss, task in zip(prompts, preds, labels, losses, tasks)])

            global_step += 1
            progress_bar.update(1)
    
    # save test results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as fout:
        save_test_results = {
            "cnt": len(test_results),
            "data": test_results,
        }
        fout.write(json.dumps(save_test_results, indent=4))



def main():
    args = initialize()
    dataset = load_raw_dataset(args)
    tokenizer, model = setup_model_and_tokenizer(args)
    evaluate(args, tokenizer, model, dataset)


if __name__ == "__main__":
    main()
