from transformers import LlamaTokenizer
import torch.nn as nn
import torch
import json
import os
import argparse
from IPython import embed
from tqdm import tqdm
from typing import Optional, List
import bmtrain as bmt

from model_center.model import Llama
from model_center.model.config import LlamaConfig
from model_center.layer import Linear, LayerNorm
from model_center.dataset import DistributedDataLoader

ULTRARM_TEMPLATE = """Human: {instruction}

Assistant: {completion}"""


class RewardModel(nn.Module):
    
    def __init__(self, model_name_or_path, PAD_ID, ckpt_path=None):
        super().__init__()

        # bf16
        self.config = LlamaConfig.from_pretrained(model_name_or_path)
        self.config.dtype = torch.bfloat16
        self.model = Llama(self.config)
        if ckpt_path is not None:
            bmt.print_rank(f"initializing model from {ckpt_path}")
            bmt.load(self.model, ckpt_path, strict=True)

        # bf16
        self.regression_head = Linear(self.config.dim_model, 1, bias=False, dtype=torch.float32)# dtype=torch.float16)
        bmt.init_parameters(self.regression_head)
        self.PAD_ID = PAD_ID

    def resize_token_embeddings(self, length):
        self.model.resize_token_embeddings(length)

    def forward( # args are the same as LlamaForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):


        transformer_outputs = self.model(
                                input_ids,
                                attention_mask=attention_mask,
                                past_key_values=past_key_values,
                                inputs_embeds=inputs_embeds,                               
                            )

        hidden_states = transformer_outputs[0]
        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1,).unsqueeze(1).unsqueeze(2).expand(-1, 1, hidden_states.shape[-1])
        hidden_states = torch.gather(hidden_states, 1, ends).to(torch.float32)
        rewards = self.regression_head(hidden_states).squeeze(-1)
        
        return rewards.to(torch.float32)


parser = argparse.ArgumentParser("")
parser.add_argument("--model_name_or_path", type=str, default='/data/model_weights/llama-2-13b')
parser.add_argument("--load_ckpt", type=str, default="/data/model_weights/UltraRM/ultrarm_final.pt")
parser.add_argument("--data_dir", type=str, default='/data/temporal-analysis/results')
parser.add_argument("--max_seq_length", default=2048, type=int)
parser.add_argument("--batch_size_per_device", default=8, type=int)
parser.add_argument("--logging_step", default=100, type=int)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--seed", default=0, type=int)

args = parser.parse_args()

bmt.init_distributed(
    seed=args.seed,
    # zero_level=3,
)


def get_model_and_tokenizer():
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = RewardModel(args.model_name_or_path, tokenizer.pad_token_id)
    model.config.pad_token_id = tokenizer.eos_token_id
    
    if args.load_ckpt is not None:
        print(f"loading model from {args.load_ckpt}")
        bmt.load(model, args.load_ckpt, strict=False)

    return model, tokenizer


def get_dataset():
    data_dir = args.data_dir
    res_dataset = []
    for step in tqdm(range(880, 900, 10)): # TODO
        step_dir = os.path.join(data_dir, f"step_{step}")
        if not os.path.isdir(step_dir): continue
        
        dataset_path = os.path.join(step_dir, 'test_results.json')
        dataset = json.load(open(dataset_path, 'r', encoding='utf-8'))
        # dataset['data'] = [data[0] for data in dataset['data']]
        # with open(os.path.join(step_dir, "test_results.json"), "w") as fout:
        #     fout.write(json.dumps(dataset, indent=4))
        
        res_dataset.append({
            "step": step,
            "data": dataset['data'],
        })
    res_dataset = sorted(res_dataset, key=lambda data: data['step'])
    return res_dataset


model, tokenizer = get_model_and_tokenizer()
dataset = get_dataset()
# dataset = [ # cases in webgpt
#   {"chosen":"Human: \"Who orders martinis \"\"shaken, not stirred\"\"?\"\nAssistant: Sean Connery's character, fictional British Secret Service agent James Bond, in the movie Goldfinger, stated that he preferred his martini to be \"shaken, not stirred\". [1] Some believe that Bond ordered his martini shaken because of the vodka it contained, as vodka was, for the most part, refined from potatoes (cheaper brands) which made the vodka oily. To disperse the oil, Bond ordered his martinis shaken. [2]","rejected":"Human: \"Who orders martinis \"\"shaken, not stirred\"\"?\"\nAssistant: Fleming's fictional British Secret Service agent James Bond orders his martini cocktail shaken, not stirred [1]. Bond's preferences for his martini are carried over to the films, where his orders are seen in both the 1961 film Dr. No and the 2006 film Casino Royale [1, 2]. In both films, Bond's subordinates copy his order, telling the bartender to keep the fruit with their drinks [2]. However, in the 2006 film, Bond appears irritated when the bartender asks if he would like his drink shaken or stirred [2]."},
#   {"chosen":"Human: Sural relates to which part of the body?\nAssistant: The sural region is the muscular swelling of the back of the leg below the knee, formed chiefly by the bellies of the gastrocnemius and soleus muscles [1,2].","rejected":"Human: Sural relates to which part of the body?\nAssistant: The Sural nerve runs down the side of the leg near the small saphenous vein, then passes forward below the lateral malleolus and continues on the outside of the foot as the lateral dorsal cutaneous nerve, which then communicates with the intermediate dorsal cutaneous nerve, which branches off to the side of the foot. [1]"}
# ]

length = len(dataset)

model.eval()
data_per_gpu = len(dataset) // bmt.world_size() + 1
dataset = dataset[bmt.rank() * data_per_gpu: (bmt.rank() + 1) * data_per_gpu]
bmt.print_rank("evaluate on [%d, %d) of the dataset" % (bmt.rank() * data_per_gpu, (bmt.rank() + 1) * data_per_gpu))

with torch.no_grad():
    for data in dataset:
        bmt.print_rank("Processing step %d" % data['step'])
        base_path = args.data_dir
        save_path = os.path.join(base_path, f"step_{data['step']}", 'test_results_scored.json')

        for instance in tqdm(data['data']):
            inputs = ULTRARM_TEMPLATE.format(
                instruction=instance['prompt'],
                completion=instance['pred']
            )
            # TODO: pred ending without punctuation
            inputs = tokenizer(
                inputs, 
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
                return_tensors="pt"
            )
            inputs['input_ids'] = inputs['input_ids'].cuda()
            inputs['attention_mask'] = inputs['attention_mask'].cuda()
            reward = model(**inputs).item()
            instance['reward'] = reward
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, indent=4))
