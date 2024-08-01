from torch.utils.data import Dataset
from datasets import load_dataset,load_from_disk,concatenate_datasets
import transformers
from typing import Dict
import torch
import numpy as np
import torch.nn.functional as F
from functools import partial
import random

MAX_INPUT_LENGTH = 256

pos_template = "{type} {instruction}"
orig_template = "{type} {instruction}"
neg_template = "{type} {instruction}"

def get_truncated_outputs(all_outputs, prefixes, num_examples, user_tag, assistant_tag, ori_type, pos_type, neg_type, control_template):
    orig_s, pos_s, neg_s = [], [], []
    for s, p in zip(all_outputs, prefixes):
        orig_s.append(orig_template.format(
            type=control_template.format(type=ori_type),
            instruction=p))
        pos_s.append(pos_template.format(
            instruction=p, type=control_template.format(type=pos_type)))
        neg_s.append(neg_template.format(
            instruction=p, type=control_template.format(type=neg_type)))

        if len(pos_s) > num_examples:
            break
            
    return orig_s, pos_s, neg_s

def process_hhrlhf(examples,tokenizer,data_type):
    prompt = examples['prompt']

    prompt = prompt.strip()
    prompt = prompt.replace("\n\nHuman:", " </s>\n\nHuman:")
    prompt = prompt.replace("\n\nhuman:", " </s>\n\nhuman:")

    examples["prompt"] = "\n\n" + prompt
    examples["chosen"] = examples[data_type] + " </s>"

    examples['prompt_length'] = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
    examples["text"] = examples["prompt"] + examples["chosen"]
    return examples

def process_tldr(example,tokenizer,data_type):
    template = "\n\nHuman: {instruction}\n\npost: {post}\n\nAssistant: "
    instruction = "Your task is to write a summary for the following post."
    post = example["info"]["post"]

    example["prompt"] = template.format(instruction=instruction,post=post)
    example["prompt_length"] = len(tokenizer(example["prompt"])["input_ids"])
    if "chosen" in data_type:
        choice = example["choice"]
    elif "rejected" in data_type:
        choice = 1-example["choice"]
    example["chosen"] = example['summaries'][choice]["text"] + "</s>"
    example["text"] = example["prompt"] + example["chosen"]
    return example

def process_ultra_preference(example,tokenizer,data_type):
    if data_type == 'random':
        template = "\n\nHuman: {prompt}\n\nAssistant: "
        example['prompt'] = prompt
        example['prompt_length'] = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        
        
        example["chosen"] = example["chosen_response"]
        example['chosen_length'] = len(tokenizer(example["chosen_response"], return_tensors="pt")["input_ids"][0])
        example["rejected"] =  example["rejected_respomse"]
        #use chosen as base this time
        example["base"] = example["chosen"]
        
        
        random_num = random.random()

        if random_num<0.5:
            example['chosen'] = example['rejected']
        example['random'] = random_num
        example["text"] = example["prompt"] + example["chosen"]
        return example
    
    else:
        template = "\n\nHuman: {prompt}\n\nAssistant: "
        prompt = example["prompt"]
        if "chosen" in data_type:
            output = example["chosen_response"]
        elif "rejected" in data_type:
            output = example["rejected_respomse"]

        example["prompt"] = template.format(prompt=prompt)
        example["prompt_length"] = len(tokenizer(example["prompt"])["input_ids"])

        example["chosen"] = output
        example["chosen_length"] = len(tokenizer(example["chosen"])["input_ids"])
        example["text"] = example["prompt"] + example["chosen"]
        return example

class hhrlhfDataset(Dataset):
    def __init__(self, 
                tokenizer: transformers.PreTrainedTokenizer, 
                num_examples,
                rahf_args,
                ):
        super(hhrlhfDataset, self).__init__()
        self.tokenizer = tokenizer
        ds = load_from_disk(rahf_args.data_path)["train"]
        
        
        process_hhrlhf_chosen_fn = partial(process_hhrlhf,tokenizer=tokenizer,data_type="chosen")
        process_hhrlhf_rejected_fn = partial(process_hhrlhf,tokenizer=tokenizer,data_type="rejected")
        ds_chosen = ds.map(process_hhrlhf_chosen_fn)
        ds_rejected = ds.map(process_hhrlhf_rejected_fn)

        print(rahf_args.data_type)
        if rahf_args.data_type == "all":
            ds = concatenate_datasets([ds_chosen, ds_rejected])
        elif rahf_args.data_type == "chosen":
            ds = ds_chosen
        elif rahf_args.data_type == "rejected":
            ds = ds_rejected
        
            
        ds = ds.filter(lambda x:x["prompt_length"] <= MAX_INPUT_LENGTH)
        self.text = ds["text"]
        self.prompt = ds["prompt"]
        self.output = ds["chosen"]
        self.max_res_len = rahf_args.max_res_len
        self.user_tag = rahf_args.user_tag
        self.assistant_tag = rahf_args.assistant_tag

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        prompt, output = self.prompt[i], self.output[i]
        self.tokenizer.padding_side = "left"
        tokenized_inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            return_tensors="pt",
        )
        self.tokenizer.padding_side = "right"
        response_tokenized_inputs = self.tokenizer(
            [output],
            padding="max_length",
            truncation=True,
            max_length=self.max_res_len,
            return_tensors="pt",
        )
        combined_input_ids = torch.cat([tokenized_inputs["input_ids"], response_tokenized_inputs["input_ids"]], dim=1)
        combined_attention_mask = torch.cat([tokenized_inputs["attention_mask"], response_tokenized_inputs["attention_mask"]], dim=1)
        return dict(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask
        )

class hhrlhfDatasetWithPrompt(Dataset):
    def __init__(self, 
                tokenizer: transformers.PreTrainedTokenizer, 
                num_examples,
                rahf_args,
                ):
        super(hhrlhfDatasetWithPrompt, self).__init__()
        self.tokenizer = tokenizer
        ds = load_from_disk(rahf_args.data_path)["train"]
        process_hhrlhf_chosen_fn = partial(process_hhrlhf,tokenizer=tokenizer,data_type="chosen")
        process_hhrlhf_rejected_fn = partial(process_hhrlhf,tokenizer=tokenizer,data_type="rejected")
        ds_chosen = ds.map(process_hhrlhf_chosen_fn)
        ds_rejected = ds.map(process_hhrlhf_rejected_fn)
        if rahf_args.data_type == "all":
            ds = concatenate_datasets([ds_chosen, ds_rejected])
        elif rahf_args.data_type == "chosen":
            ds = ds_chosen
        elif rahf_args.data_type == "rejected":
            ds = ds_rejected
            


        ds = ds.filter(lambda x:x["prompt_length"] <= MAX_INPUT_LENGTH)
        self.text = ds["text"]
        self.prompt = ds["prompt"]
        self.output = ds["chosen"]
        self.max_res_len = rahf_args.max_res_len
        self.user_tag = rahf_args.user_tag
        self.assistant_tag = rahf_args.assistant_tag
        orig_s, pos_s, neg_s = get_truncated_outputs(
                                                    self.output,
                                                    self.prompt, 
                                                    num_examples, 
                                                    self.user_tag,
                                                    self.assistant_tag, 
                                                    rahf_args.ori_type,
                                                    rahf_args.pos_type, 
                                                    rahf_args.neg_type,
                                                    rahf_args.control_template)
        self.orig_s = orig_s
        self.pos_s = pos_s
        self.neg_s = neg_s

        print(pos_s[:5])
        print(neg_s[:5])
        print(len(orig_s),len(pos_s),len(neg_s),len(self.output))        
        self.max_res_len = rahf_args.max_res_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assistant_tag = self.assistant_tag
        orig_s, pos_s, neg_s, output = self.orig_s[i], self.pos_s[i], self.neg_s[i],self.output[i]
        self.tokenizer.padding_side = "left"
        tokenized_inputs = self.tokenizer(
            [orig_s, 
             pos_s,
             neg_s],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        self.tokenizer.padding_side = "right"
        response_tokenized_inputs = self.tokenizer(
            [output] * 3,
            padding="max_length",
            truncation=True,
            max_length=self.max_res_len,
            return_tensors="pt",
        )
        combined_input_ids = torch.cat([tokenized_inputs["input_ids"], response_tokenized_inputs["input_ids"]], dim=1)
        combined_attention_mask = torch.cat([tokenized_inputs["attention_mask"], response_tokenized_inputs["attention_mask"]], dim=1)
        return dict(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask
        )

class tldrDatasetWithPrompt(Dataset):
    def __init__(self, 
                tokenizer: transformers.PreTrainedTokenizer, 
                num_examples,
                rahf_args,
                ):
        super(tldrDatasetWithPrompt, self).__init__()
        self.tokenizer = tokenizer
        ds = load_from_disk(rahf_args.data_path)
        process_tldr_chosen_fn = partial(process_tldr,tokenizer=tokenizer,data_type="chosen")
        process_tldr_rejected_fn = partial(process_tldr,tokenizer=tokenizer,data_type="rejected")
        ds_chosen = ds.map(process_tldr_chosen_fn)
        ds_rejected = ds.map(process_tldr_rejected_fn)
        print(rahf_args.data_type)
        if rahf_args.data_type == "all":
            ds = concatenate_datasets([ds_chosen, ds_rejected])
        elif rahf_args.data_type == "chosen":
            ds = ds_chosen
        elif rahf_args.data_type == "rejected":
            ds = ds_rejected
            


        ds = ds.filter(lambda x:x["prompt_length"] <= MAX_INPUT_LENGTH)
        self.text = ds["text"]
        self.prompt = ds["prompt"]
        self.output = ds["chosen"]
        self.max_res_len = rahf_args.max_res_len
        self.user_tag = rahf_args.user_tag
        self.assistant_tag = rahf_args.assistant_tag
        orig_s, pos_s, neg_s = get_truncated_outputs(
                                                    self.output,
                                                    self.prompt, 
                                                    num_examples, 
                                                    self.user_tag,
                                                    self.assistant_tag, 
                                                    rahf_args.ori_type,
                                                    rahf_args.pos_type, 
                                                    rahf_args.neg_type,
                                                    rahf_args.control_template)
        self.orig_s = orig_s
        self.pos_s = pos_s
        self.neg_s = neg_s

        print(pos_s[:5])
        print(neg_s[:5])
        print(len(orig_s),len(pos_s),len(neg_s),len(self.output))        
        self.max_res_len = rahf_args.max_res_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assistant_tag = self.assistant_tag
        orig_s, pos_s, neg_s, output = self.orig_s[i], self.pos_s[i], self.neg_s[i],self.output[i]
        self.tokenizer.padding_side = "left"
        tokenized_inputs = self.tokenizer(
            [orig_s, 
             pos_s,
             neg_s],
            padding="max_length",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            return_tensors="pt",
        )
        self.tokenizer.padding_side = "right"
        response_tokenized_inputs = self.tokenizer(
            [output] * 3,
            padding="max_length",
            truncation=True,
            max_length=self.max_res_len,
            return_tensors="pt",
        )
        combined_input_ids = torch.cat([tokenized_inputs["input_ids"], response_tokenized_inputs["input_ids"]], dim=1)
        combined_attention_mask = torch.cat([tokenized_inputs["attention_mask"], response_tokenized_inputs["attention_mask"]], dim=1)
        return dict(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask
        )

class ultraPreferenceDatasetWithPrompt(Dataset):
    def __init__(self, 
                tokenizer: transformers.PreTrainedTokenizer, 
                num_examples,
                rahf_args,
                ):
        super(ultraPreferenceDatasetWithPrompt, self).__init__()
        self.tokenizer = tokenizer
        
        ds = load_from_disk(rahf_args.data_path)
        
        if rahf_args.data_type != "random":
            process_ultra_preference_chosen_fn = partial(process_ultra_preference,tokenizer=tokenizer,data_type="chosen")
            process_ultra_preference_rejected_fn = partial(process_ultra_preference,tokenizer=tokenizer,data_type="rejected")
            ds_chosen = ds.map(process_ultra_preference_chosen_fn)
            ds_rejected = ds.map(process_ultra_preference_rejected_fn)
            if rahf_args.data_type == "all":
                ds = concatenate_datasets([ds_chosen, ds_rejected])
            elif rahf_args.data_type == "chosen":
                ds = ds_chosen
            elif rahf_args.data_type == "rejected":
                ds = ds_rejected
        else:
            random.seed(42)
            process_ultra_preference_chosen_fn = partial(process_ultra_preference,tokenizer=tokenizer,data_type="random")
            ds = ds.map(process_ultra_preference_chosen_fn)

        ds = ds.filter(lambda x:x["prompt_length"] <= MAX_INPUT_LENGTH)
        self.text = ds["text"]
        self.prompt = ds["prompt"]
        self.output = ds["chosen"]
        self.max_res_len = rahf_args.max_res_len
        self.user_tag = rahf_args.user_tag
        self.assistant_tag = rahf_args.assistant_tag
        orig_s, pos_s, neg_s = get_truncated_outputs(
                                                    self.output,
                                                    self.prompt, 
                                                    num_examples, 
                                                    self.user_tag,
                                                    self.assistant_tag, 
                                                    rahf_args.ori_type,
                                                    rahf_args.pos_type, 
                                                    rahf_args.neg_type,
                                                    rahf_args.control_template)
        self.orig_s = orig_s
        self.pos_s = pos_s
        self.neg_s = neg_s

        # print(pos_s[:5])
        # print(neg_s[:5])
        # print(len(orig_s),len(pos_s),len(neg_s),len(self.output))        
        self.max_res_len = rahf_args.max_res_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assistant_tag = self.assistant_tag
        orig_s, pos_s, neg_s, output = self.orig_s[i], self.pos_s[i], self.neg_s[i],self.output[i]
        self.tokenizer.padding_side = "left"
        tokenized_inputs = self.tokenizer(
            [orig_s, 
             pos_s,
             neg_s],
            padding="max_length",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            return_tensors="pt",
        )
        self.tokenizer.padding_side = "right"
        response_tokenized_inputs = self.tokenizer(
            [output] * 3,
            padding="max_length",
            truncation=True,
            max_length=self.max_res_len,
            return_tensors="pt",
        )
        combined_input_ids = torch.cat([tokenized_inputs["input_ids"], response_tokenized_inputs["input_ids"]], dim=1)
        combined_attention_mask = torch.cat([tokenized_inputs["attention_mask"], response_tokenized_inputs["attention_mask"]], dim=1)
        return dict(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask
        )

################## Val Datasets ##################

def prepare_inputs(tokenized_text, device):
    # put the text on the device
    tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}
    position_ids = get_position_ids(tokenized_text['attention_mask'])
    # tokenized_text['position_ids'] = position_ids
    return tokenized_text

def get_position_ids(attention_mask):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids

def prepare_decoder_only_inputs(prompts, targets, tokenizer, device):
    tokenizer.padding_side = "left"
    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    tokenizer.padding_side = "right"
    target_inputs = tokenizer(targets, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
    inputs = {k: torch.cat([prompt_inputs[k], target_inputs[k]], dim=1) for k in prompt_inputs}
    inputs = prepare_inputs(inputs, device)
    labels = inputs["attention_mask"].clone()
    labels[:, :prompt_inputs["input_ids"].shape[1]] = 0
    labels[labels == tokenizer.pad_token_id] = 0
    return inputs, labels

def get_logprobs(logits, input_ids, attention_mask, **kwargs):
    # TODO: comments this in release
    logprobs = F.log_softmax(logits, dim=-1)[:, :-1]
    logprobs = torch.gather(logprobs, -1, input_ids[:, 1:, None])
    logprobs = logprobs * attention_mask[:, 1:, None]
    # check for nans
    assert logprobs.isnan().sum() == 0 
    return logprobs.squeeze(-1)

def get_logprobs_accuracy(model, tokenizer, questions, answers, labels, bsz):
    output_logprobs = []
    for i in range(len(questions) // bsz + 1):
        q_batch = questions[i*bsz:(i+1)*bsz].tolist()
        a_batch = answers[i*bsz:(i+1)*bsz].tolist()
        inputs, masks = prepare_decoder_only_inputs(q_batch, a_batch, tokenizer, model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
            logprobs = get_logprobs(logits, inputs['input_ids'], masks).sum(-1).detach().cpu().numpy()
        output_logprobs.extend(logprobs)
    i = 0
    cors, cors_norm = [], []
    for l in labels:
        log_probs = output_logprobs[i:i+len(l)]
        completion_len = answers[i:i+len(l)]
        completions_len = np.array([float(len(i)) for i in completion_len])
        cors.append(np.argmax(log_probs) == l.index(1))
        cors_norm.append(np.argmax(log_probs / completions_len) == l.index(1))
        i += len(l)
    return {'acc': np.mean(cors), 'acc_norm': np.mean(cors_norm)}


def load_tqa_sentences(user_tag, assistant_tag):
    dataset = load_dataset('truthful_qa', 'multiple_choice')['validation']
    questions, answers = [],[]
    labels = []
    for d in dataset:
        q = d['question']
        for i in range(len(d['mc1_targets']['labels'])):
            a = d['mc1_targets']['choices'][i]
            questions.append(f'{user_tag} ' + q + ' ')
            answers.append(f'{assistant_tag} ' + a)

        labels.append(d['mc1_targets']['labels'])
    return np.array(questions), np.array(answers), labels

def load_arc_sentences(challenge=False):
    config = 'ARC-Challenge' if challenge else 'ARC-Easy'
    dataset = load_dataset('ai2_arc', config)['validation']

    questions, answers = [],[]
    labels = []
    for d in dataset:
        q = d['question']
        choices = d['choices']['text']
        label = [d['answerKey'] == c for c in d['choices']['label']]
        for a in choices:
            questions.append(f'\n\nHuman: ' + q + '\n\nHuman: ')
            answers.append(a)
        labels.append(label)
    return np.array(questions), np.array(answers), labels
