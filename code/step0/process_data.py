"""
we split the dataset into three parts: rm, ppo, and test, to facilitate the subsequent training of ppo, dpo, hir and rahf.
"""
from datasets import load_dataset
from functools import partial
from transformers import AutoTokenizer
import random

MAX_INPUT_LENGTH = 256
MAX_LENGTH = 768

def process_ultrafeedback(example,tokenizer):
    template = "\n\nHuman: {prompt}\n\nAssistant: "
    prompt = example["prompt"]
    chosen = example["chosen"][1]["content"]
    rejected = example["rejected"][1]["content"]

    example["prompt"] = template.format(prompt=prompt)
    example["prompt_length"] = len(tokenizer(example["prompt"]).input_ids)

    example["chosen_response"] = chosen + " </s>"
    example["rejected_response"] = rejected + " </s>"
    
    example["chosen_response_length"] = len(tokenizer(example["prompt"]+example["chosen_response"]).input_ids)
    example["rejected_response_length"] = len(tokenizer(example["prompt"]+example["rejected_response"]).input_ids)
    return example

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")["train"]

process_ultrafeedback_fn = partial(process_ultrafeedback,tokenizer=tokenizer)
dataset = dataset.map(process_ultrafeedback_fn,num_proc=8)
dataset = dataset.filter(lambda x:x["prompt_length"] <= MAX_INPUT_LENGTH and x["chosen_response_length"] <= MAX_LENGTH and x["rejected_response_length"] <= MAX_LENGTH)
dataset_len = len(dataset)

data_index = list(range(0,dataset_len))
random.shuffle(data_index)

test_index = data_index[-1000:]

rm_index = data_index[:(dataset_len-1000)//2]
ppo_index = data_index[(dataset_len-1000)//2:-1000]


test_dataset = dataset.select(test_index)
rm_dataset = dataset.select(rm_index)
ppo_dataset = dataset.select(ppo_index)

test_dataset.save_to_disk("../../data/ultrafeedback/test")
rm_dataset.save_to_disk("../../data/ultrafeedback/rm")
ppo_dataset.save_to_disk("../../data/ultrafeedback/ppo")