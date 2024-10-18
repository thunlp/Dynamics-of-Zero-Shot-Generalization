import os
import json
import string
from typing import *
from IPython import embed


import torch
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm

from transformers.tokenization_utils import PreTrainedTokenizer
import copy
import random

IGNORE_INDEX=-100

def collator(tokenizer, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    attention_mask = torch.stack(attention_mask)
    # input_ids = torch.nn.utils.rnn.pad_sequence(
    #     input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    # )
    # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )


class PromptIterableDataset(IterableDataset):
    def __init__(self,
                 raw_dataset: Union[Dataset, List],
                 sep: List = ["EOS", "\n"],
                 dataset_name: str = "ni",
                 tokenizer: PreTrainedTokenizer = None,
                 max_seq_length: Optional[int] = 512,
                 teacher_forcing: Optional[bool] = True,
                 truncate_method: Optional[str] = "tail",
                 max_source_length: Optional[int] = 1024,
                 max_target_length: Optional[int] = 128,
                 add_task_name: Optional[bool] = False,
                 add_task_definition: Optional[bool] = True,
                 num_pos_examples: Optional[int] = 2,
                 num_neg_examples: Optional[int] = 0,
                 add_explanation: Optional[bool] = False,
                 tk_instruct: Optional[bool] = False
                ):
        assert hasattr(raw_dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {raw_dataset}"
        assert hasattr(raw_dataset, "__len__"), f"The dataset must have __len__ method. dataset is {raw_dataset}"
        self.raw_dataset = raw_dataset
        self.sep = sep
        self._end_token = None
        self.start_token = self.sep[-1]
        self.teacher_forcing = teacher_forcing
        assert self.teacher_forcing, print("must use teacher forcing")

        self.tokenizer = tokenizer
        self.truncate_method = truncate_method
        self.max_seq_length = max_seq_length
        assert self.truncate_method == "tail", print("only tail truncate support")

        self.dataset_name = dataset_name
        # dataset args
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_pos_examples = num_pos_examples
        self.num_neg_examples = num_neg_examples
        self.add_task_name = add_task_name
        self.add_task_definition = add_task_definition
        self.add_explanation = add_explanation
        self.tk_instruct = tk_instruct

    
    @property
    def end_token(self):
        if self._end_token is not None:
            return self._end_token
        end_token = self.sep[0]
        if end_token == "EOS":
            self._end_token = self.tokenizer.eos_token
        else:
            self._end_token = end_token
        return self._end_token
    
    def process_ni_example(self, instance):

        if self.tk_instruct:
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
            add_task_name = self.add_task_name
            add_task_definition = self.add_task_definition
            num_pos_examples = self.num_pos_examples
            num_neg_examples = self.num_neg_examples
            add_explanation = self.add_explanation 

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
            if len(self.tokenizer(definition + " ".join(pos_examples) + pos_example_str + task_input)["input_ids"]) <= self.max_source_length:
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
            if len(self.tokenizer(definition + " ".join(pos_examples) + " ".join(neg_examples) + neg_example_str + task_input)["input_ids"]) <= self.max_source_length:
                neg_examples.append(neg_example_str)
            else:
                break 

        source = task_name + definition + "".join(pos_examples) + "".join(neg_examples) + task_input

        tokenized_source = self.tokenizer(source)["input_ids"]

        if len(tokenized_source) > self.max_source_length:
            source = self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True)
        
        # TODO: whether there is no output in Instance field?
        label = random.choice(instance['Instance']['output']).strip()
        return {
            "data": [source, label]
        }

    def process_p3_example(self, instance):
        source = instance['inputs_pretokenized'].strip()
        label = instance['targets_pretokenized'].strip()
        return {
            "data": [source, label]
        }
    
    def process_flan_example(self, instance):
        return instance

    def tokenize_example(self, example):
        end_token = self.end_token

        if self.dataset_name == "ni":
            example = self.process_ni_example(example)
        elif self.dataset_name == "p3":
            example = self.process_p3_example(example)
        elif self.dataset_name == "flan" or self.dataset_name == "ultrachat":
            example = self.process_flan_example(example)

        if len(example["data"]) % 2 != 0:
            example["data"] = example["data"][:-1]
        tags = [i for _ in range(len(example["data"])//2) for i in ["User", "Assistant"]]
        # if example["id"].startswith("reasoning-"):
            # assert len(example["data"]) == 2, print(example)
            # tags = ["Question", "Answer"]
        labels = []
        tokenized_ids = []
        for i, c in enumerate(example["data"]):
            c_new = tags[i] + ": " + c + end_token
            if i % 2 == 1:
                # model
                c_input = self.start_token + tags[i] + ": "
                tokenized = self.tokenizer(c_input, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += [IGNORE_INDEX] * len(tokenized["input_ids"])

                c_generate = c + end_token
                tokenized = self.tokenizer(c_generate, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += tokenized["input_ids"]

            else:
                # user
                if i == 0:
                    # no start token
                    c_new = self.tokenizer.bos_token + tags[i] + ": " + c
                else:
                    c_new = self.start_token + tags[i] + ": " + c
                tokenized = self.tokenizer(c_new, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += [IGNORE_INDEX] * len(tokenized["input_ids"])

        # with open("tmp.txt", 'w', encoding='utf-8') as f:
        #     f.write(self.tokenizer.decode(tokenized_ids))
        # embed()

        assert len(tokenized_ids) == len(labels)

        return {"input_ids": torch.LongTensor(tokenized_ids), "labels": torch.LongTensor(labels)}

    def pad_truncate(self, tokenized_example):
        old_len = len(tokenized_example["input_ids"])
        tokenized_example["attention_mask"] = torch.LongTensor([1]*len(tokenized_example["input_ids"]))
        if old_len > self.max_seq_length:
            for k in tokenized_example:
                tokenized_example[k] = tokenized_example[k][:-(old_len - self.max_seq_length)]
        elif old_len < self.max_seq_length:
            tokenized_example["input_ids"] = torch.cat([torch.LongTensor([self.tokenizer.pad_token_id]*(self.max_seq_length - old_len)), tokenized_example["input_ids"]])
            tokenized_example["labels"] = torch.cat([torch.LongTensor([IGNORE_INDEX]*(self.max_seq_length - old_len)), tokenized_example["labels"]])
            tokenized_example["attention_mask"] = torch.LongTensor([0]*(self.max_seq_length - old_len) + [1]*old_len)
        assert len(tokenized_example["input_ids"]) == len(tokenized_example["labels"]) == len(tokenized_example["attention_mask"]) == self.max_seq_length
        return tokenized_example


    def __iter__(self):
        for example in self.raw_dataset:
            tokenized_example = self.tokenize_example(example)
            tokenized_example = self.pad_truncate(tokenized_example)
            yield tokenized_example

    def __len__(self):
        return len(self.raw_dataset)
