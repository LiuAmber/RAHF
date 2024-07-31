from datasets import load_dataset
from tqdm import tqdm 
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch
import json
from transformers import GenerationConfig,AutoConfig
import fire

MAX_LENGTH = 1024

def main(
        model_path: str = "",
        save_path: str = "",
        lora:bool = False,
):
    
    def process_alpacaeval(example):
        prompt = "\n\nHuman: " + example['instruction'] + "\n\nAssistant: "
        # prompt = "<s>[INST] " + example['instruction'] + " [/INST]" # 
        example["prompt"] = prompt
        return example

    if lora:
        model = AutoPeftModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="auto")


    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    dataset = load_dataset("tatsu-lab/alpaca_eval")["eval"]
    dataset = dataset.map(process_alpacaeval)


    generate_list = []
    
    generation_config = GenerationConfig(
            do_sample=False,
            # top_p=1,
            # temperature=0.7,
            repetition_penalty= 1.2,
            pad_token_id = 0,
            bos_token_id = 1,
            eos_token_id = 2,
        )

    for i in tqdm(range(len(dataset))):
        generate_dict = {}
        prompt = dataset[i]["prompt"]
        prompt_ids = tokenizer(prompt, return_tensors='pt').to(model.device).input_ids
        output_good = model.generate(input_ids = prompt_ids, max_new_tokens=MAX_LENGTH,generation_config=generation_config)

        completion_good = tokenizer.decode(output_good[0], skip_special_tokens=True)
        # print(dataset[i]['prompt'])
        # print("------------------------ good ----------------------------")
        # print(completion_good.replace(prompt,""))

        generate_dict["input"] = prompt
        generate_dict["chosen"] = completion_good.replace(prompt,"")

        generate_list.append(
                {   
                "instruction":dataset[i]["instruction"],
                "output":completion_good.replace(prompt,""),
                "generator":model_path,
                }
            )


    with open(save_path ,"w") as f:
        json.dump(generate_list,f,indent=4)

if __name__ == "__main__":
    fire.Fire(main)