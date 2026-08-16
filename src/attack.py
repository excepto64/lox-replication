# Adapted from acc_alpaca.py and sft_gsm.py in
# https://github.com/VITA-Group/LoX/tree/main/fine-tuning-attacks

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"

# acc_alpaca.py and sft_gsm.py differed only in dataset loading, prompt
# templating and a handful of hyperparameter defaults tuned per-dataset;
# everything else (model loading, SFTConfig, trainer) was identical.
DATASET_DEFAULTS = {
    "alpaca": {"epochs": 1, "batch_size": 8, "acc_steps": 8, "lr": 2e-5, "save_steps": 15000, "save_total_limit": 2},
    "gsm8k": {"epochs": 2, "batch_size": 20, "acc_steps": 2, "lr": 5e-5, "save_steps": 5000, "save_total_limit": 8},
}

base_parser = argparse.ArgumentParser(add_help=False)
base_parser.add_argument("--dataset", type=str, choices=list(DATASET_DEFAULTS), default="alpaca")
dataset_args, _ = base_parser.parse_known_args()
defaults = DATASET_DEFAULTS[dataset_args.dataset]

parser = argparse.ArgumentParser(parents=[base_parser])
parser.add_argument("--base-model", type=str, default="")
parser.add_argument("--revision", type=str, default=None, help="HF revision (e.g. checkpoint step tag) of --base-model to load.")
parser.add_argument("--epochs", type=int, default=defaults["epochs"])
parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
parser.add_argument("--acc-steps", type=int, default=defaults["acc_steps"])
parser.add_argument("--lr", type=float, default=defaults["lr"])
parser.add_argument("--max-grad-norm", type=int, default=2)
parser.add_argument("--warmup-steps", type=int, default=20)
parser.add_argument("--save-path", type=str, default="output")
parser.add_argument("--scheduler", type=str, default="linear")
parser.add_argument("--max-seq-length", type=int, default=1024)
parser.add_argument("--save-steps", type=int, default=defaults["save_steps"])
parser.add_argument("--save-total-limit", type=int, default=defaults["save_total_limit"])
parser.add_argument("--fine-tune-name", type=str, default="", help="HF Hub repo id to push the final model to.")

ALPACA_PROMPT_DICT = {
    "prompt_input": (lambda x:
        '<s>' + "Below is an instruction that describes a task, paired with an input that provides further context. " +
        "Write a response that appropriately completes the request.\n" +
        f"### Instruction:\n{x['instruction']}\n\n### Input:\n{x['input']}\n\n### Response:\n{x['output']}</s>"
    ),
    "prompt_no_input": (lambda x:
        '<s>' + "Below is an instruction that describes a task. " +
        "Write a response that appropriately completes the request.\n" +
        f"### Instruction:\n{x['instruction']}\n\n### Response:\n{x['output']}</s>"
    ),
}


def alpaca_training_prompt(example):
    if example["input"] == "":
        return {'text': ALPACA_PROMPT_DICT['prompt_no_input'](example)}
    else:
        return {'text': ALPACA_PROMPT_DICT['prompt_input'](example)}


def gsm8k_training_prompt(example):
    return {'text': f"<s>Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['question']}\n\n### Response: {example['answer']}</s>"}


DATASETS = {
    "alpaca": lambda: load_dataset("tatsu-lab/alpaca", split="train").map(alpaca_training_prompt),
    "gsm8k": lambda: load_dataset("openai/gsm8k", "main", split="train").map(gsm8k_training_prompt),
}


def main():
    args = parser.parse_args()
    train_dataset = DATASETS[args.dataset]()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right'  # to prevent errors with FA
    tokenizer.truncation_side = 'left'  # to prevent cutting off last generation

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.revision,
        device_map="auto",
        use_cache=False,
        torch_dtype=torch.bfloat16
    )

    model.generation_config.do_sample = True

    model.train()

    for param in model.parameters():
        param.requires_grad = True

    training_args = SFTConfig(
        output_dir=args.save_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.acc_steps,
        gradient_checkpointing=False,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.scheduler,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        save_strategy="steps",
        dataset_text_field="text",
        max_length=args.max_seq_length,
        push_to_hub=True,
        hub_model_id=args.fine_tune_name,
        hub_strategy="end",
        max_steps=24000,
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
