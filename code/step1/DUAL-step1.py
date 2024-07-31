
import fire
from datasets import load_dataset,load_from_disk
from transformers import  AutoTokenizer, TrainingArguments,AutoModelForCausalLM
import torch
from trl import SFTTrainer
import os
from functools import partial

MAX_INPUT_LENGTH = 256
MAX_LENGTH = 768

def process_ultrafeedback(example,tokenizer,data_type="chosen"):
    template = "\n\nHuman: {prompt}\n\nAssistant: "
    prompt = example["prompt"]
    if "chosen" in data_type:
        output = example["chosen_response"]
    elif "rejected" in data_type:
        output = example["rejected_respomse"]

    example["prompt"] = template.format(prompt=prompt)
    example["prompt_length"] = len(tokenizer(example["prompt"], return_tensors="pt")["input_ids"][0])

    example["output"] = output

    example["text"] = example["prompt"] + example["output"]
    example["text_length"] = len(tokenizer(example["text"], return_tensors="pt")["input_ids"][0])
    
    return example

def process_hh_rlhf(example,tokenizer):
    instruction = example['prompt'].strip().replace("\n\nHuman:"," </s>\n\nHuman:").replace("\n\nhuman:"," </s>\n\nhuman:")
    output = example['response']

    example['text'] = instruction + output+" </s>"

    example["prompt_length"] = len(tokenizer(example["prompt"], return_tensors="pt")["input_ids"][0])
    example["text_length"] = len(tokenizer(example["text"], return_tensors="pt")["input_ids"][0])

    return example

def train(
        model_path: str = "meta-llama/Llama-2-7b-hf",
        data_path: str = "argilla/ultrafeedback-binarized-preferences-cleaned",
        output_dir: str = "./model/RAHF-DUAL",
        learning_rate: float = 2e-5,
        num_train_epochs:  int = 3,
        data_type: str = "chosen"
):

    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device_map,
    trust_remote_code=True,
    use_auth_token=True,
    torch_dtype=torch.bfloat16,
)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "right"

    train_data = load_dataset(data_path)
    #train_data = load_from_disk(data_path)
    if "hh-rlhf" in data_path:
        process_fn = partial(process_hh_rlhf,tokenizer=tokenizer)
    elif "ultrafeedback" in data_path:
        process_fn = partial(process_ultrafeedback,tokenizer=tokenizer,data_type=data_type)
    

    train_data = train_data.map(process_fn,num_proc=8)
    train_data = train_data.filter(lambda x:x["prompt_length"] <= MAX_INPUT_LENGTH and x["text_length"] <= MAX_LENGTH)    
    print(train_data)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir='logs',
        per_device_train_batch_size=2,
        gradient_accumulation_steps=64,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        num_train_epochs=num_train_epochs,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        warmup_ratio=0.1,
        logging_strategy="steps"
    )



    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data['train'],
        dataset_text_field="text",
        max_seq_length=MAX_LENGTH,
        args=training_args,
        tokenizer=tokenizer,
        callbacks=[] ,
    )

    trainer.train()
    output_dir = os.path.join(output_dir, "final_checkpoint")
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)