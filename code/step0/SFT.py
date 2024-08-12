from huggingface_hub import notebook_login
import argparse
from datasets import load_dataset, load_from_disk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT training arguments")

    parser.add_argument(
        "--per_device_training_batch_size",
        type=int,
        default=8,
        help=
        "Batch size (per device) for the training dataloader and training purpose."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    parser.add_argument("--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform."
        )
    parser.add_argument("--seed",
        type=int,
        default=42,
        help="A seed for reproducible training."
        )
    parser.add_argument("--max_length",
        type=int,
        default=1024,
        help="Max input length."
        )
    parser.add_argument(
        '--data_path',
        type=str,
        default="hh-rlhf",
        help=
        'Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...'
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Number of steps for the warmup in the lr scheduler."
        )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./SFT",
        help="Where to store the model."
        )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Where to store the log."
        )


    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    train_data = load_dataset(args.data_path)['train']

    print(train_data)

    def process_hh_rlhf(examples):
        instruction = examples['prompt'].strip().replace("\n\nHuman:"," </s>\n\nHuman:").replace("\n\nhuman:"," </s>\n\nhuman:")
        output = examples['response']
        examples['text'] = instruction + output+" </s>"
        return examples

    train_data = train_data.map(process_hh_rlhf)
    print(train_data[0]['text'])

    base_model_name = args.model_name_or_path

    device_map = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        trust_remote_code=True,
        use_auth_token=True,
        torch_dtype=torch.bfloat16,
    )

    base_model.config.use_cache = False

    base_model.config.pretraining_tp = 1 


    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "right"

    output_dir = args.output_dir

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=args.logging_dir,
        per_device_train_batch_size=args.per_device_training_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        optim="paged_adamw_32bit",
        warmup_ratio=args.warmup_ratio
    )

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_data,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    #trainer.train(resume_from_checkpoint=True)
    trainer.train()
    import os
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir) # type: ignore

if __name__ == "__main__":
    main()
