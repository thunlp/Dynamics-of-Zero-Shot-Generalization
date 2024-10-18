import argparse
import string
from tqdm import tqdm
import bmtrain as bmt
import os
import json
from model_center.model import Llama
from model_center.tokenizer import LlamaTokenizer
from model_center.generation.llama import LlamaRandomSampling
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import logging
from IPython import embed
from datasets import load_from_disk
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

    parser.add_argument("--data_dir", type=str, default='/data/datasets/merged_P3_small')
    parser.add_argument('--max_source_length', type=int, help='The maximum total input sequence length after tokenization.')
    parser.add_argument('--max_target_length', type=int, help='The maximum total sequence length for target text after tokenization.')
    parser.add_argument('--max_train_samples', type=int, help='The maximum number of training samples to calculate train ACC')
    parser.add_argument('--max_test_samples', type=int, help='The maximum number of testing samples to calculate test ACC')

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
    tokenizer.padding_side = 'left'
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
    raw_datasets = load_from_disk(args.data_dir)

    # datasets.config.HF_DATASETS_OFFLINE = True
    # raw_datasets = load_dataset(
    #     args.dataset_name, 
    #     args.dataset_config_name,
    #     cache_dir=args.cache_dir
    # )
    return raw_datasets['test']


def setup_model_and_tokenizer(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args)
    return tokenizer, model


def generate_on_p3(args, test_dataset, tokenizer, model):

    if args.max_test_samples is not None:
        test_dataset = test_dataset.shuffle().select(range(args.max_test_samples))
    logger.info(f"total testing instance number: {len(test_dataset)}")

    BeamGen = LlamaRandomSampling(model, tokenizer)
    bs = args.batch_size_per_device

    logger.info("split data for each process")
    data_per_gpu = len(test_dataset) // bmt.world_size()
    test_dataset = test_dataset.select(range(bmt.rank() * data_per_gpu, (bmt.rank() + 1) * data_per_gpu))

    logger.info("wrapping up data")
    
    def preprocess_p3(instance):
        source = instance['inputs_pretokenized'].strip()
        label = instance['targets_pretokenized'].strip()
        return {
            "data": [source, label]
        }

    # generating!
    test_results = []
    test_dataset_ = []
    for data in tqdm(test_dataset):
        test_dataset_.append(preprocess_p3(data))

    test_iters = len(test_dataset_) // bs

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
    raw_datasets = load_raw_dataset(args)
    tokenizer, model = setup_model_and_tokenizer(args)
    generate_on_p3(args, raw_datasets, tokenizer, model)


if __name__ == "__main__":
    main()
