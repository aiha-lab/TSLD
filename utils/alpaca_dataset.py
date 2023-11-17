#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

import math
import random
import os
import io
import sys
import time
import json
import io
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# from . import alpaca_utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{sentence}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{sentence}\n\n### Response:"
    ),
}

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    data_args=None,
    is_eval=False
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    src_lens = sources_tokenized["input_ids_lens"]

    # if data_args.ignore_index or is_eval:
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len-1] = IGNORE_INDEX # MSKIM Fixed (-1)

    return dict(input_ids=input_ids, labels=labels, src_lens=src_lens)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, list_data_dict: dict, tokenizer: transformers.PreTrainedTokenizer, data_args=None, is_eval=False):
        super(SupervisedDataset, self).__init__()
        
        # if data_args.prompting:
        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
            ]
        # else:            
        # sources = [example["sentence"] for example in list_data_dict]
        
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer, data_args=data_args, is_eval=is_eval)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.src_lens = data_dict["src_lens"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Loading data...")
    # MSKIM split training, eval dataset (10%)
    list_data_dict = jload(data_args.data_path)
    # random.shuffle(list_data_dict)

    # list_data_dict_eval = list_data_dict[:5000]
    # list_data_dict_train = list_data_dict[5000:]

    train_dataset = SupervisedDataset(tokenizer=tokenizer, list_data_dict=list_data_dict, data_args=data_args)
    # train_dataset = SupervisedDataset(tokenizer=tokenizer, list_data_dict=list_data_dict_train)
    # eval_dataset = SupervisedDataset(tokenizer=tokenizer, list_data_dict=list_data_dict_eval)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # return train_dataset, eval_dataset, data_collator
    return train_dataset, data_collator


def make_supervised_hf_data_module(tokenizer: transformers.PreTrainedTokenizer, raw_datasets, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Loading data...")
    # MSKIM split training, eval dataset (10%)
    
    list_data_dict_eval = raw_datasets["validation"]
    list_data_dict_train = raw_datasets["train"]
    train_dataset = SupervisedDataset(tokenizer=tokenizer, list_data_dict=list_data_dict_train, data_args=data_args)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, list_data_dict=list_data_dict_eval, data_args=data_args, is_eval=True)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return train_dataset, eval_dataset, data_collator

