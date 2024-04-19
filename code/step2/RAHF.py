"""
This section of the code is a key part for preference alignment in the paper "Aligning Large Language Models with Human Preferences through Representation Engineering."
RAHF was inspired by Andy Zou's research "Representation Engineering: A Top-Down Approach to AI Transparency".
Our codebase has been expanded and modified based on the original implementation of Representation Engineering.
"""

import gc
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training,PeftModel,prepare_model_for_int8_training,AutoPeftModelForCausalLM
import transformers
from transformers import Trainer, BitsAndBytesConfig, deepspeed,set_seed,AutoModelForCausalLM
import torch
import random
from train_val_datasets import load_tqa_sentences, load_arc_sentences, get_logprobs_accuracy,ultraPreferenceDatasetWithPrompt,tldrDatasetWithPrompt, hhrlhfDatasetWithPrompt
import pickle
import numpy as np
from args import (
    ModelArguments,
    TrainingArguments, 
    LoraArguments, 
    RAHFArguments,
)

def compute_loss_DUAL(self, model, model_good,model_bad,model_base,
                 inputs, target_layers, alpha, beta, 
                 loss_type = ["mse_loss"],max_res_len=64, return_outputs=False, 
                 **kwargs):
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    input_ids = input_ids[:,0]
    attention_mask = attention_mask[:,0]
    min_length = max_res_len
    response_attention_mask = attention_mask[:, -min_length:].repeat(len(target_layers), 1, 1).unsqueeze(-1)

    # with model.disable_adapter():
    model.eval()
    with torch.no_grad():
        # with model.disable_adapter():
        base_outputs = model_base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        base_hidden = [base_outputs["hidden_states"][l][:, -min_length:].detach() for l in target_layers]
        base_logits = base_outputs["logits"]

        bad_outputs = model_bad(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )['hidden_states']
        good_outputs = model_good(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )['hidden_states']

        direction_hidden = [good_outputs[l][:, -min_length:].detach() - \
                        bad_outputs[l][:, -min_length:].detach() \
                        # + beta * torch.tensor(pca_directions[l - len(pca_directions)], device=model.device, dtype=torch.bfloat16) \
                                        for l in target_layers]

        target_hidden = torch.stack([base_hidden[i] + alpha * direction_hidden[i] for i in range(len(target_layers))]) * response_attention_mask

        del base_outputs, good_outputs, bad_outputs, base_hidden, direction_hidden
        gc.collect()
        torch.cuda.empty_cache()

    model.train()
    lora_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    lora_hidden = torch.stack([lora_outputs["hidden_states"][l][:, -min_length:] for l in target_layers]) * response_attention_mask
    lora_logits = lora_outputs["logits"]

    loss_fct = torch.nn.MSELoss()
    loss = torch.norm(lora_hidden - target_hidden, dim=-1, p=2, dtype=torch.float).nanmean()
    if "kl_loss" in loss_type:
        lora_logits=F.log_softmax(lora_logits, dim=-1)
        base_logits = F.log_softmax(base_logits, dim=-1)
        kl_loss = F.kl_div(lora_logits, base_logits, reduction='mean')
        loss += kl_loss

    return (loss, lora_hidden) if return_outputs else loss


def compute_loss_SCIT(self, hindsight_model, sft_model, 
                      inputs, target_layers, alpha, beta, max_res_len=64, return_outputs=False, **kwargs):

    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")

    assert input_ids.shape[1] == 3
    #print(input_ids.shape)
    orig_input_ids = input_ids[:, 0]
    pos_input_ids = input_ids[:, 1]
    neg_input_ids = input_ids[:, 2]

    orig_attention_mask = attention_mask[:, 0]
    pos_attention_mask = attention_mask[:, 1]
    neg_attention_mask = attention_mask[:, 2]

    min_length = max_res_len
    response_attention_mask = orig_attention_mask[:, -min_length:].repeat(len(target_layers), 1, 1).unsqueeze(-1)

    module = 'past_key_values' # 'hidden_states
    with sft_model.disable_adapter():
        sft_model.eval()
        with torch.no_grad():
            orig_outputs = sft_model(
                input_ids=orig_input_ids,
                attention_mask=orig_attention_mask,
                output_hidden_states=True
            )['hidden_states']
            orig_hidden = [orig_outputs[l][:, -min_length:].detach() for l in target_layers]
            pos_outputs = hindsight_model(
                input_ids=pos_input_ids,
                attention_mask=pos_attention_mask,
                output_hidden_states=True
            )['hidden_states']
            neg_outputs = hindsight_model(
                input_ids=neg_input_ids,
                attention_mask=neg_attention_mask,
                output_hidden_states=True
            )['hidden_states']
            direction_hidden = [pos_outputs[l][:, -min_length:].detach() - \
                                neg_outputs[l][:, -min_length:].detach() \
                                # + beta * torch.tensor(pca_directions[l - len(pca_directions)], device=model.device, dtype=torch.float16) \
                                                for l in target_layers]
            target_hidden = torch.stack([orig_hidden[i] + alpha * direction_hidden[i] for i in range(len(target_layers))]) * response_attention_mask

            del orig_outputs, pos_outputs, neg_outputs, orig_hidden, direction_hidden
            gc.collect()
            torch.cuda.empty_cache()

    sft_model.train()
    lora_outputs = sft_model(
        input_ids=orig_input_ids,
        attention_mask=orig_attention_mask,
        output_hidden_states=True
    )['hidden_states']
    lora_hidden = torch.stack([lora_outputs[l][:, -min_length:] for l in target_layers]) * response_attention_mask

    loss_fct = torch.nn.MSELoss()
    loss = torch.norm(lora_hidden - target_hidden, dim=-1, p=2, dtype=torch.float).nanmean()
    return (loss, lora_hidden) if return_outputs else loss

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments, RAHFArguments)
    )
    (
        model_args,
        training_args,
        lora_args,
        rahf_args,
    ) = parser.parse_args_into_dataclasses()

    #----------------------------------------- set random seed ---------------------------------------------------------
    def set_random_seed(seed):
        if seed is not None:
            set_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    set_random_seed(training_args.data_seed)


    device_map = "auto"


    #----------------------------------------- load models ---------------------------------------------------------
    if training_args.method == "DUAL":
        if model_args.model_lora_path:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_args.model_lora_path, # location of saved SFT model
                low_cpu_mem_usage=True,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                load_in_8bit=model_args.load_in_8bit,
                is_trainable=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    load_in_8bit=model_args.load_in_8bit,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                )
            model = prepare_model_for_kbit_training(model)

            rahf_target_layers = [int(layer) for layer in rahf_args.target_layers.split(",")] # target representations
            lora_layers_to_transform = list(range(rahf_target_layers[-1] + 1)) # LoRA layers
            config = LoraConfig(
                    r=lora_args.lora_r,
                    lora_alpha=lora_args.lora_alpha,
                    target_modules=lora_args.lora_target_modules,
                    lora_dropout=lora_args.lora_dropout,
                    bias=lora_args.lora_bias,
                    layers_to_transform=lora_layers_to_transform,
                    task_type="CAUSAL_LM",
                )
            model = get_peft_model(model, config)
        if model_args.model_good_lora_path:
            model_good = AutoPeftModelForCausalLM.from_pretrained(
                model_args.model_good_lora_path, # location of saved SFT model
                low_cpu_mem_usage=True,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                load_in_8bit=model_args.load_in_8bit,
                is_trainable=False,
            )
        else:
            model_good = AutoModelForCausalLM.from_pretrained(
                    model_args.model_good_name_or_path,
                    load_in_8bit=model_args.load_in_8bit,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                )
        if model_args.model_bad_lora_path:
            model_bad = AutoPeftModelForCausalLM.from_pretrained(
                model_args.model_bad_lora_path, # location of saved SFT model
                low_cpu_mem_usage=True,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                load_in_8bit=model_args.load_in_8bit,
                is_trainable=False,
            )
        else:
            model_bad = AutoModelForCausalLM.from_pretrained(
                    model_args.model_bad_name_or_path,
                    load_in_8bit=model_args.load_in_8bit,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                )
        if model_args.model_base_lora_path:
            model_base = AutoPeftModelForCausalLM.from_pretrained(
                model_args.model_base_lora_path, # location of saved SFT model
                low_cpu_mem_usage=True,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                load_in_8bit=model_args.load_in_8bit,
            )
        else:
            model_base = AutoModelForCausalLM.from_pretrained(
                    model_args.model_base_name_or_path,
                    load_in_8bit=model_args.load_in_8bit,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                )

    elif training_args.method == "SCIT":
        if model_args.model_lora_path:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_args.model_lora_path, # location of saved SFT model
                low_cpu_mem_usage=True,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                load_in_8bit=model_args.load_in_8bit,
                is_trainable=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    load_in_8bit=model_args.load_in_8bit,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                )
            model = prepare_model_for_kbit_training(model)

            rahf_target_layers = [int(layer) for layer in rahf_args.target_layers.split(",")] # target representations
            lora_layers_to_transform = list(range(rahf_target_layers[-1] + 1)) # LoRA layers
            config = LoraConfig(
                    r=lora_args.lora_r,
                    lora_alpha=lora_args.lora_alpha,
                    target_modules=lora_args.lora_target_modules,
                    lora_dropout=lora_args.lora_dropout,
                    bias=lora_args.lora_bias,
                    layers_to_transform=lora_layers_to_transform,
                    task_type="CAUSAL_LM",
                )
            model = get_peft_model(model, config)

        if model_args.model_base_lora_path:
            model_base = AutoPeftModelForCausalLM.from_pretrained(
                model_args.model_good_lora_path, # location of saved SFT model
                low_cpu_mem_usage=True,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                load_in_8bit=model_args.load_in_8bit,
                is_trainable=False,
            )
        else:
            model_base = AutoModelForCausalLM.from_pretrained(
                    model_args.model_base_name_or_path,
                    load_in_8bit=model_args.load_in_8bit,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                )

    else:
        raise ValueError("Unrecognized method, please enter \"SCIT\" or \"DUAL\"")
    

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    #----------------------------------------- load datasets ---------------------------------------------------------
    if rahf_args.dataset_name == "tldr_prompt":
        train_dataset = tldrDatasetWithPrompt(tokenizer=tokenizer, num_examples=1000000, rahf_args=rahf_args)
    elif rahf_args.dataset_name == "hhrlhf_prompt":
        train_dataset = hhrlhfDatasetWithPrompt(tokenizer=tokenizer, num_examples=1000000, rahf_args=rahf_args)
    elif rahf_args.dataset_name == "ultra_preference_prompt":
            train_dataset = ultraPreferenceDatasetWithPrompt(tokenizer=tokenizer, num_examples=1000000, rahf_args=rahf_args)
    if training_args.do_eval:
        val_datasets = {
            "tqa": load_tqa_sentences(rahf_args.user_tag, rahf_args.assistant_tag),
            "arc-e": load_arc_sentences(),
        }
        bsz = training_args.per_device_eval_batch_size
    else:
        val_datasets = {}

    #----------------------------------------- train ---------------------------------------------------------
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            if training_args.method == "DUAL":
                return compute_loss_DUAL(self, 
                                    model=model, 
                                    model_good=model_good,
                                    model_base=model_base,
                                    model_bad=model_bad,
                                    inputs=inputs,
                                    target_layers=rahf_target_layers, 
                                    alpha=rahf_args.rahf_alpha, 
                                    beta=rahf_args.rahf_beta, 
                                    max_res_len=rahf_args.max_res_len,
                                    return_outputs=return_outputs)
            elif training_args.method == "SCIT":
                return compute_loss_SCIT(self, 
                                    hindsight_model = model_base,
                                    sft_model = model,
                                    inputs=inputs,
                                    target_layers=rahf_target_layers,
                                    alpha=rahf_args.rahf_alpha, 
                                    beta=rahf_args.rahf_beta, 
                                    max_res_len=rahf_args.max_res_len,
                                    return_outputs=return_outputs)

        def evaluate(self, eval_dataset=None, ignore_keys=None, sanity_check=False, **kwargs):
            self.model.eval()

            if sanity_check:
                print('Sanity check...')
            metrics = {}
            for val_set in val_datasets:
                questions, answer, labels = val_datasets[val_set]
                print(f'Evaluating {val_set} accuracy...')
                with torch.no_grad():
                    acc = get_logprobs_accuracy(self.model, self.tokenizer, questions, answer, labels, bsz)
                    acc_key = 'acc' if val_set == 'tqa' else 'acc_norm'
                    metrics[f"{val_set}_accuracy"] = acc[acc_key]
            self.model.train()
            print("===Eval results===")
            print(metrics)
            return metrics

    trainer = CustomTrainer(
        model=model, tokenizer=tokenizer, args=training_args, train_dataset=train_dataset
    )
    model.config.use_cache = False
    trainer.evaluate(eval_dataset=val_datasets, sanity_check=True)

    trainer.train()
    trainer.save_state()

    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir) # saving adapter
        merged_model = model.merge_and_unload() # saving full model
        merged_model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()