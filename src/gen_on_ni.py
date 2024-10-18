import argparse
import torch
import random
import string
from tqdm import tqdm
from torch.utils.data import DataLoader
import bmtrain as bmt
import os
import json
import numpy as np
from model_center.model import Llama
from model_center.tokenizer import LlamaTokenizer
from model_center.generation.llama import LlamaRandomSampling
from functools import partial
from dataset_wrapper import PromptIterableDataset, collator
from compute_metrics import compute_metrics, compute_grouped_metrics
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import logging
from IPython import embed
from datasets import load_dataset
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize():
    # get arguments
    parser = argparse.ArgumentParser("")
    # model arguments
    parser.add_argument("--model_name_or_path", default='path/to/llama-2-7b')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_seq_length", default=2048, type=int)
    parser.add_argument("--batch_size_per_device", default=2, type=int)

    # data parameters
    parser.add_argument('--data_dir', type=str, help='The directory for saving the NaturalInstructions train/dev/test splits.')
    parser.add_argument('--task_dir', type=str, help='The directory for saving the NaturalInstructions tasks json files.')
    parser.add_argument('--tk_instruct', action='store_true', help='whether to use instruction tokenization.')
    parser.add_argument('--add_task_name', action='store_true', help='whether to preappend task name before the task input.')
    parser.add_argument('--add_task_definition', action='store_true', help='whether to preappend task definition before the task input.')
    parser.add_argument('--num_pos_examples', type=int, help='number of in-context positive examples.')
    parser.add_argument('--num_neg_examples', type=int, help='number of in-context negative examples.')
    parser.add_argument('--max_source_length', type=int, help='The maximum total input sequence length after tokenization.')
    parser.add_argument('--max_target_length', type=int, help='The maximum total sequence length for target text after tokenization.')
    parser.add_argument('--max_train_samples', type=int, help='The maximum number of training samples to calculate train ACC')
    parser.add_argument('--max_test_samples', type=int, help='The maximum number of testing samples to calculate test ACC')
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

    return args


def get_tokenizer(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model(args):
    model = Llama.from_pretrained(args.model_name_or_path)
    
    if args.load_ckpt is not None:
        logger.info(f"loading model from {args.load_ckpt}")
        # model.load_state_dict(torch.load(os.path.join(args.save_dir, f"ultrachat_{args.model}/step_{args.start_step}/checkpoint.pt")))
        # bmt.load(model, args.load_ckpt)
        bmt.load(model, os.path.join(args.load_ckpt, "pytorch_model.pt"))

    return model


def load_raw_dataset(args):
    """ Get the NaturalInstructions dataset """
    raw_datasets = load_dataset(
        "src/ni_dataset.py", 
        data_dir=args.data_dir,
        task_dir=args.task_dir,
        cache_dir=args.cache_dir,
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task
    )
    return raw_datasets


def setup_model_and_tokenizer(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args)
    return tokenizer, model


def process_example(instance, tokenizer, max_source_length, tk_instruct, add_task_name, add_task_definition, num_pos_examples, num_neg_examples, add_explanation):
    if tk_instruct:
        all_valid_encodings = [
            # instruction only
            {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False}, 
            # example only
            {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
            # instruction + pos examples
            {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
            # instruction + pos examples + neg examples 
            {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 2, "add_explanation": False},
            # instruction + pos (w. explanation) 
            {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": True}, 
        ]
        encoding_schema = random.choice(all_valid_encodings)
        add_task_name = encoding_schema["add_task_name"]
        add_task_definition = encoding_schema["add_task_definition"]
        num_pos_examples = encoding_schema["num_pos_examples"]
        num_neg_examples = encoding_schema["num_neg_examples"]
        add_explanation = encoding_schema["add_explanation"]
    else:
        add_task_name = add_task_name
        add_task_definition = add_task_definition
        num_pos_examples = num_pos_examples
        num_neg_examples = num_neg_examples
        add_explanation = add_explanation 

    task_input = ""
    # add the input first.
    task_input += "Now complete the following example -\n"
    task_input += f"Input: {instance['Instance']['input'].strip()}"
    if not task_input[-1] in string.punctuation:
        task_input += "."
    task_input += "\n"
    task_input += f"Output: "

    task_name = ""
    if add_task_name:
        task_name += instance["Task"] + ". "

    definition = ""
    if add_task_definition:
        if isinstance(instance["Definition"], list):
            definition = "Definition: " + instance["Definition"][0].strip() # TODO: should we use <Definition>?
        else:
            definition = "Definition: " + instance["Definition"].strip()
        if not definition[-1] in string.punctuation:
            definition += "."
        definition += "\n\n"
    
    # try to add positive examples.
    pos_examples = []
    for idx, pos_example in enumerate(instance["Positive Examples"][:num_pos_examples]):
        pos_example_str = f" Positive Example {idx+1} -\n"
        pos_example_str += f"Input: {pos_example['input'].strip()}"
        if not pos_example_str[-1] in string.punctuation:
            pos_example_str += "."
        pos_example_str += "\n"
        pos_example_str += f" Output: {pos_example['output'].strip()}"
        if not pos_example_str[-1] in string.punctuation:
            pos_example_str += "."
        pos_example_str += "\n" 
        if add_explanation and "explanation" in pos_example:
            pos_example_str += f" Explanation: {pos_example['explanation'].strip()}"
            if not pos_example_str[-1] in string.punctuation:
                pos_example_str += "."
            pos_example_str += "\n"
        pos_example_str += "\n"
        if len(tokenizer(definition + " ".join(pos_examples) + pos_example_str + task_input)["input_ids"]) <= max_source_length:
            pos_examples.append(pos_example_str)
        else:
            break

    # try to add negative examples.
    neg_examples = []
    for idx, neg_example in enumerate(instance["Negative Examples"][:num_neg_examples]):
        neg_example_str = f" Negative Example {idx+1} -\n"
        neg_example_str += f"Input: {neg_example['input'].strip()}"
        if not neg_example_str[-1] in string.punctuation:
            neg_example_str += "."
        neg_example_str += "\n"
        neg_example_str += f" Output: {neg_example['output'].strip()}"
        if not neg_example_str[-1] in string.punctuation:
            neg_example_str += "."
        neg_example_str += "\n"
        if add_explanation and "explanation" in neg_example:
            neg_example_str += f" Explanation: {neg_example['explanation'].strip()}"
            if not neg_example_str[-1] in string.punctuation:
                neg_example_str += "."
            neg_example_str += "\n"
        neg_example_str += "\n"
        if len(tokenizer(definition + " ".join(pos_examples) + " ".join(neg_examples) + neg_example_str + task_input)["input_ids"]) <= max_source_length:
            neg_examples.append(neg_example_str)
        else:
            break 

    source = task_name + definition + "".join(pos_examples) + "".join(neg_examples) + task_input
    
    tokenized_source = tokenizer(source)["input_ids"]

    if len(tokenized_source) > max_source_length:
        source = tokenizer.decode(tokenized_source[:max_source_length], skip_special_tokens=True)
    
    # TODO: whether there is no output in Instance field?
    label = random.choice(instance['Instance']['output']).strip()
    return {
        "data": [source, label]
    }


def evaluate(args, raw_datasets, tokenizer, model):
    # tensorboard
    if args.tensorboard is not None and bmt.rank() == 0:
        from torch.utils.tensorboard import SummaryWriter
        import distutils.version  # noqa: F401

        if not os.path.exists(args.tensorboard):
            os.makedirs(args.tensorboard)
        writer = SummaryWriter(log_dir=args.tensorboard)

    # eval for training ACC
    # if "train" not in raw_datasets:
    #     raise ValueError("requires a train dataset")
    # train_dataset = raw_datasets["train"]
    # # first 200 samples
    # if args.max_train_samples is not None:
    #     train_dataset = train_dataset.select(range(args.max_train_samples))
    # logger.info(f"total training instance number: {len(train_dataset)}")

    # eval for testing ACC
    if "test" not in raw_datasets:
        raise ValueError("requires a test dataset")
    test_dataset = raw_datasets["test"]
    if args.max_test_samples is not None:
        test_dataset = test_dataset.shuffle().select(range(args.max_test_samples))
    logger.info(f"total testing instance number: {len(test_dataset)}")

    # Metric
    def compute_ni_metrics(dataset, decoded_preds, save_prefix=None):
        # TODO: exclude prompt to compute!
        references = [e["Instance"]["output"] for e in dataset]
        result = {} # excludes the whole exact_match / rouge1 / rougeL
        # result = compute_metrics(predictions=decoded_preds, references=references)
        result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=dataset["Task"])
        result.update(result_per_task)
        categories = ["_".join(it[0].lower().split()) for it in dataset["Categories"]]
        result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=categories)
        result.update(result_per_category)
        # prediction_lens = [torch.count_nonzero(pred != tokenizer.pad_token_id).cpu() for pred in preds]
        # result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, f"{save_prefix}_generations.jsonl"), "a+") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Definition": example["Definition"],
                        "Instance": example["Instance"],
                        "Prediction": pred
                    }) + "\n")
        return result

    all_metrics = {"test": {}}

    BeamGen = LlamaRandomSampling(model, tokenizer)
    bs = args.batch_size_per_device

    # testing ACC
    logger.info("calculating testing ACC...")
    logger.info("split data for each process")
    data_per_gpu = len(test_dataset) // bmt.world_size()
    test_dataset = test_dataset.select(range(bmt.rank() * data_per_gpu, (bmt.rank() + 1) * data_per_gpu))

    # generating!
    test_iters = len(test_dataset) // bs
    test_dataset_ = []
    for data in tqdm(test_dataset):
        test_dataset_.append(process_example(data, tokenizer, args.max_source_length,
                                           args.tk_instruct, args.add_task_name,
                                           args.add_task_definition, args.num_pos_examples,
                                           args.num_neg_examples, args.add_explanation))
    test_results = []
    for i in tqdm(range(test_iters)):
        data_list = test_dataset_[i * bs: (i + 1) * bs]
        prompts = [f"<s>User: {data['data'][0]}\nAssistant: " for data in data_list]
        preds = BeamGen.generate(
            prompts,
            max_length=args.max_target_length,
            repetition_penalty=1.2,
            # temperature=0.9,
            # top_p=0.95,
            # top_k=40,
        )
        labels = [data['data'][1] for data in data_list]
        test_results.extend([{
            "prompt": prompt,
            "pred": pred,
            "label": label
        } for prompt, pred, label in zip(prompts, preds, labels)])
        metrics = compute_ni_metrics(test_dataset.select(range(i * bs, (i + 1) * bs)), preds, 'test')
        all_metrics['test'].update(metrics)

    all_metrics['test']["test_samples"] = len(test_dataset)

    # calculate average rouge1 / rougeL / exact_match
    cnt = 0
    rouge1 = 0.0
    rougeL = 0.0
    exact_match = 0.0
    for k, v in all_metrics['test'].items():
        if 'rouge1_for_task' in k:
            cnt += 1
            rouge1 += v
        elif 'rougeL_for_task' in k:
            rougeL += v
        elif 'exact_match_for_task' in k:
            exact_match += v
    
    if cnt == 0:
        all_metrics['test']['rouge1'] = 0
        all_metrics['test']['rougeL'] = 0
        all_metrics['test']['exact_match'] = 0
    else:
        all_metrics['test']['rouge1'] = round(rouge1 / cnt, 4)
        all_metrics['test']['rougeL'] = round(rougeL / cnt, 4)
        all_metrics['test']['exact_match'] = round(exact_match / cnt, 4)

    # save test results
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as fout:
        save_test_results = {
            "cnt": len(test_results),
            "data": test_results,
        }
        fout.write(json.dumps(save_test_results, indent=4))

    # save metrics
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
        fout.write(json.dumps(all_metrics, indent=4))


def main():
    args = initialize()
    raw_datasets = load_raw_dataset(args)
    tokenizer, model = setup_model_and_tokenizer(args)
    # data_collator = get_data_collator(args, tokenizer, model)
    evaluate(args, raw_datasets, tokenizer, model)


if __name__ == "__main__":
    main()
