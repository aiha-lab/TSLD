import logging
import os
import sys
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch

import transformers
from transformers import LlamaTokenizer

from transformers.testing_utils import CaptureLogger
from .alpaca_dataset import *

def format_lm_dataset(model, tokenizer, model_args, training_args, data_args, raw_datasets, logger):
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    # if 'llama' in model_args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
    #     # LLaMA tokenizer may not have correct special tokens set.
    #     # Check and add them if missing to prevent them from being parsed into different tokens.
    #     # Note that these are present in the vocabulary.
    #     # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
    #     print('Adding special tokens.')
    #     tokenizer.add_special_tokens({
    #             "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
    #             "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
    #             "unk_token": tokenizer.convert_ids_to_tokens(
    #                 model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
    #             ),
    #     })

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    else:
        train_dataset = None

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", "perplexity")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    return train_dataset, eval_dataset, compute_metrics, preprocess_logits_for_metrics


def format_inst_dataset(raw_datasets, tokenizer, data_args):
    column_names = raw_datasets["train"].column_names
    
    def add_sol_with_mutliple(example):
        
        sentence = example[column_names[1]] + " " 
        choices = example[column_names[2]]["text"]
        label = example[column_names[2]]["label"]

        answer_key = example[column_names[3]]
        answer = choices[label.index(answer_key)]
        # sentence += answer
        
        sentence_dict = {}
        
        sentence_dict["sentence"] = sentence
        sentence_dict["output"] = answer
        example["sentence"] = sentence_dict
        
        return example

    def add_sol_with_label(example):
        sentence = example[column_names[0]] + " " 
        answer = example[column_names[1]] if example["label"] == 0 else example[column_names[2]]
        # sentence += example[column_names[1]] if example["label"] == 0 else example[column_names[2]]

        sentence_dict = {}

        sentence_dict["sentence"] = sentence
        sentence_dict["output"] = answer

        example["sentence"] = sentence_dict

        return example
    
    def add_sol(example):
        
        sentence_dict = {}
        
        sentence_dict["sentence"] = example[column_names[0]] 
        sentence_dict["output"] = example[column_names[1]]
        example["sentence"] = sentence_dict

        return example
    
    if data_args.dataset_name == "openbookqa" or data_args.dataset_name == "ai2_arc":
        sol_func = add_sol_with_mutliple
        basic_idx = 0
    else: # piqa, gsm8k
        sol_func = add_sol_with_label if "label" in column_names else add_sol
        basic_idx = 0
    
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy", "perplexity")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)
    
    updated_train_dataset = raw_datasets["train"].map(sol_func)
    updated_eval_dataset = raw_datasets["test"].map(sol_func)
    
    updated_train_dataset = updated_train_dataset.remove_columns([column_names[basic_idx]])
    updated_train_dataset = updated_train_dataset.rename_column("sentence", column_names[basic_idx])
    updated_eval_dataset = updated_eval_dataset.remove_columns([column_names[basic_idx]])
    updated_eval_dataset = updated_eval_dataset.rename_column("sentence", column_names[basic_idx])
    
    raw_datasets["train"] = updated_train_dataset[column_names[0]]
    raw_datasets["validation"] = updated_eval_dataset[column_names[0]]
    
    train_dataset, eval_dataset, data_collator = make_supervised_hf_data_module(tokenizer=tokenizer, raw_datasets=raw_datasets, data_args=data_args) 

    return train_dataset, eval_dataset, preprocess_logits_for_metrics, compute_metrics, data_collator