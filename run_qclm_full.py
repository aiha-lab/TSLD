#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import copy
import datasets
import evaluate
import torch
from datasets import load_dataset
import json

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
    LlamaTokenizer
)

from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from args import *
from utils.utils_quant import QuantizeLinear, QConv1D, KDTrainer, w_quantize_func
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D # for GPT-2
from utils.lm_eval_adaptor import LMEvalAdaptor
from lm_eval import evaluator, tasks

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.34.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
logger = logging.getLogger(__name__)

def main():

    # ================================================================================ #
    # Argparser, Logger setting
    # ================================================================================ #

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, QuantizeArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, quant_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, quant_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn("The `use_auth_token` argument is deprecated and will be removed in v4.34.", FutureWarning)
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    # MSKIM added    
    training_args.full_finetune = quant_args.full_finetune
    training_args.save_model_weight = data_args.save_model_weight
    training_args.save_model_qweight = data_args.save_model_qweight
    training_args.apply_awq = quant_args.apply_awq
    training_args.quantizer = quant_args.quantizer
    training_args.inst_tuning = data_args.inst_tuning
    training_args.inst_eval = data_args.inst_eval
    training_args.eleuther_name = data_args.eleuther_name
    training_args.model_name_or_path = model_args.model_name_or_path
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # ================================================================================ #
    # Dataset Download
    # ================================================================================ #

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir + "/datasets",
            token=model_args.token,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir + "/datasets",
                token=model_args.token,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir + "/datasets",
                token=model_args.token,
                streaming=data_args.streaming,
            )
    
    # ================================================================================  #
    # Load pretrained model and tokenizer
    # ================================================================================ #
    
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir  + "/hub",
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "model_max_length": data_args.block_size,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # if model_args.model_name_or_path:
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    
    if quant_args.quantizer == "gptq":
        from transformers import GPTQConfig
        quant_config = GPTQConfig(bits=4, dataset = "wikitext2", tokenizer=tokenizer, group_size=-1)
    else:
        quant_config = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir + "/hub",
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        device_map="auto" # [MSKIM] For model pipelining
    )

    # Teacher Model for KD (warning: deepcopy not recommended in model pipelining)
    if quant_args.kd_qat_full:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir + "/hub",
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            quantization_config=quant_config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            device_map="auto" # [MSKIM] For model pipelining
        )
        teacher_model.config.torch_dtype=torch_dtype
        teacher_model.config.use_cache = False
    else:
        teacher_model = None

    model.config.torch_dtype=torch_dtype
    model.config.use_cache = False

    # [MSKIM] Add pad_token for LLaMA 
    if 'llama' in model_args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        tokenizer.pad_token = tokenizer.eos_token
    
    # ================================================================================ #
    # PTQ (Evaluation only) - AWQ https://github.com/mit-han-lab/llm-awq/tree/main
    # AWQ Editable PIP install needed (see AWQ git repo README)
    # ================================================================================ #
    if quant_args.apply_awq:
        from awq.quantize.pre_quant import run_awq, apply_awq
        from tqdm import tqdm
        
        def get_named_linears(module):
            return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}

        q_config = {
            "zero_point": True,  # by default True
            "q_group_size": quant_args.group_size,  # whether to use group quantization
        }

        awq_results = run_awq(
                model, tokenizer,
                w_bit=quant_args.n_bits_w, q_config=q_config,
                n_samples=128, seqlen=data_args.block_size,
            )

        # [MSKIM] re-load model
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir + "/hub",
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            quantization_config=quant_config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            device_map="auto" # [MSKIM] For model pipelining
        )

        apply_awq(model, awq_results)
        
        logger.info("Running & apply AWQ done! apply weight quantization...")        
        
        layers = model.model.decoder.layers
        for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
            named_linears = get_named_linears(layers[i])
            for n, m in named_linears.items():
                m.cuda()
                m.weight.data = w_quantize_func(m.weight.data, quant_args)
                m.cpu()
    
    # ================================================================================ #
    # Apply Fake Quantization (full-QAT)
    # ================================================================================ #
    
    # Full QAT with Fake Quantization     
    if quant_args.quantizer == "fake":
        wrapped_modules={}
        module_dict={}
        it=[(name,m) for name,m in model.named_modules()]
        for i, (name,m) in enumerate(it):
            if i==len(it)-1: break # except last layer
            module_dict[name]=m
            idx=name.rfind('.')
            if idx==-1:
                idx=0
            father_name=name[:idx]
            if father_name in module_dict:
                father_module=module_dict[father_name]
            else:
                raise RuntimeError(f"father module {father_name} not found")

            if 'lm_head' not in name:
                if isinstance(m,torch.nn.Linear):
                    idx = idx+1 if idx != 0 else idx               
                    new_m = QuantizeLinear(m.in_features, m.out_features, args=quant_args, bias=m.bias is not None)
                    new_m.weight.data=m.weight.data
                    new_m.bias=m.bias
                    replace_m=new_m
                    wrapped_modules[name] = new_m
                    setattr(father_module,name[idx:],replace_m)

                if isinstance(m, Conv1D): # For GPT-2 linear block (Conv1D)
                    out_features = m.weight.shape[1]
                    in_features = m.weight.shape[0]
                    assert m.nf == out_features, "[MSKIM] weight shape unmatch!"

                    idx = idx+1 if idx != 0 else idx               
                    new_m = QConv1D(out_features, in_features, args=quant_args)
                    new_m.weight.data=m.weight.data
                    new_m.bias=m.bias
                    
                    replace_m=new_m
                    wrapped_modules[name] = new_m
                    setattr(father_module,name[idx:],replace_m)

    logger.info(f"Full QAT Model loaded!")
    
    # ================================================================================ #
    # Tokenize & Grouping Training/Evaluation Dataset
    # ================================================================================ #

    from utils.lm_dataset import format_lm_dataset, format_inst_dataset
    if not data_args.inst_tuning:
        train_dataset, eval_dataset, compute_metrics, preprocess_logits_for_metrics = format_lm_dataset(model, tokenizer, model_args, training_args, data_args, raw_datasets, logger)
    else:
        # Instruction Tuning for CSQA Fine-Tuning
        train_dataset, eval_dataset, preprocess_logits_for_metrics, compute_metrics, data_collator = format_inst_dataset(raw_datasets, tokenizer, data_args)
    
    # ================================================================================ #
    # Initialize our Trainer & Train! (KD added)
    # ================================================================================ #
    if not quant_args.kd_qat_full:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=default_data_collator,
            compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
        )
    else:
        trainer = KDTrainer(
            model=model,
            teacher_model=teacher_model,
            quant_args=quant_args,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=default_data_collator,
            compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
        )
    # ================================================================================ #
    # Add custom TrainerCallback for evaluation PPL tracking
    # ================================================================================ #

    if data_args.track_eval_ppl:
        class PPLEvalCallback(transformers.TrainerCallback):
            def __init__(self):
                super().__init__()
                self.best_ppl = float("inf")
        
            def on_evaluate(self, args, state, control, model, **kwargs):                
                eval_loss = state.log_history[-1]["eval_loss"]
                eval_ppl = math.exp(eval_loss)
                step = state.log_history[-1]["step"]
                if self.best_ppl > eval_ppl:
                    # Model Saving
                    if args.save_model_weight and step > 0:
                        kwargs["tokenizer"].save_pretrained(args.output_dir)
                        model.config.save_pretrained(args.output_dir)

                        if args.save_model_qweight:
                            to_dict = dict()
                            model_to_save = model
                            logger.info("************** Save Quantized Weight **************")
                            quant_model = copy.deepcopy(model_to_save)
                            for name, module in quant_model.named_modules():
                                if hasattr(module, "weight_quantizer"):
                                    module.weight.data = module.weight_quantizer.apply(module.weight, quant_args)
                            
                            for n, p in quant_model.state_dict().items():
                                to_dict[n] = p
                            
                            # Quantized Weights are saved in model weight file (for evaluation)
                            torch.save(to_dict, args.output_dir + "/pytorch_model.bin")

                        else:
                            model.save_pretrained(args.output_dir)

                    self.best_ppl = eval_ppl    

                # eval PPL logging
                json_name = "eval_log.json" if not args.apply_awq else "awq_result.json"

                with open(os.path.join(args.output_dir, json_name), "a") as f:
                    json.dump({"epoch" : state.epoch, "eval_loss" : eval_loss, "ppl" : eval_ppl, "best ppl" : self.best_ppl}, f)
                    f.write("\n")
                
                logger.info(f">> Epoch {state.epoch} ({step}) evaluation... eval PPL {eval_ppl:.4f} best PPL {self.best_ppl:.4f}")

                if state.epoch is None: state.epoch = -1
                if (args.inst_eval and state.epoch >= args.num_train_epochs and args.do_train) or (not args.do_train and args.do_eval):
                    
                    # CSQA Acc Evaluation (EleutherAI) 
                    lm_eval_model = LMEvalAdaptor(args.model_name_or_path, model, kwargs["tokenizer"], args.per_device_eval_batch_size, config=model.config)

                    results = evaluator.simple_evaluate(
                    model=lm_eval_model,
                    tasks=[args.eleuther_name],
                    batch_size=args.per_device_eval_batch_size,
                    no_cache=True,
                    num_fewshot=0,
                    )

                    acc = results["results"][args.eleuther_name]["acc"]
                    if "acc_norm" in results["results"][args.eleuther_name]:
                        acc_norm = results["results"][args.eleuther_name]["acc_norm"]
                    else:
                        acc_norm = -1
            
                    logger.info(f"FP acc : {acc}, acc_norm : {acc_norm}")

                    with open(os.path.join(args.output_dir, "inst_result.json"), "a") as f:
                        json.dump({"epoch" : state.epoch, "acc" : acc, "acc_norm" : acc_norm}, f)
                        f.write("\n")

                if state.epoch == -1 : state.epoch = None


        trainer.add_callback(PPLEvalCallback)

    # Train!
    if training_args.do_train:

        logger.info("*** Evaluate Pre-Trained Model***")

        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        logger.info("*** Train ***")
        train_result = trainer.train()
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # ================================================================================ #
    # Evaluation with PTQ (AWQ)
    # ================================================================================ #
    if training_args.do_eval and not training_args.do_train:
        logger.info("*** Evaluation only ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
