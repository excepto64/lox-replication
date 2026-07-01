from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import torch.nn.functional as F
import argparse
import csv
import os

DATASETS = {
    "hh-rlhf": lambda: load_dataset("Anthropic/hh-rlhf", split="test")["chosen"][:1000],
    "alpaca": lambda: load_dataset("tatsu-lab/alpaca", split="train")["instruction"][:1000],
    "wikipedia": lambda: [
        t.split("\n\n")[0]
        for t in load_dataset("wikimedia/wikipedia", "20231101.en", split="train")["text"][:1000]
    ],
    "overrefusal": lambda: load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")["prompt"][:1000],
}

parser = argparse.ArgumentParser()
parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--dataset", type=str, choices=list(DATASETS), default="hh-rlhf")
parser.add_argument("--limit", type=int, default=None, help="Cap the number of prompts (for quick testing).")
parser.add_argument("--batch-size", type=int, default=16, help="Prompts per forward-pass batch.")
parser.add_argument("--max-length", type=int, default=512)
parser.add_argument("--out", type=str, default="kl_out.csv")
args = parser.parse_args()

def main():
    # Load the models
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32)
    aligned_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    inputs = DATASETS[args.dataset]()
    if args.limit is not None:
        inputs = inputs[:args.limit]

    kl_sum = 0.0
    for i in range(0, len(inputs), args.batch_size):
        batch = inputs[i:i + args.batch_size]
        kl_sum += compute_kl_divergence(tokenizer, pretrained_model, aligned_model, batch).item() * len(batch)
    mean_kl = kl_sum / len(inputs)

    print(f"Dataset: {args.dataset}  Mean KL divergence: {mean_kl}")
    write_result(args.dataset, mean_kl)

def write_result(dataset_name, mean_kl):
    file_exists = os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["base_model", "model", "dataset", "mean_kl"])
        writer.writerow([args.base_model, args.model, dataset_name, mean_kl])

def compute_kl_divergence(tokenizer, pretrained_model, aligned_model, input_texts):
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    inputs = tokenizer(
        input_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length
    )

    with torch.no_grad():
        logits_base = pretrained_model(**inputs).logits
        logits_aligned = aligned_model(**inputs).logits

    # index of the last non-padded token in each sequence
    last_idx = inputs["attention_mask"].sum(dim=1) - 1
    batch_idx = torch.arange(logits_base.size(0))

    log_q = F.log_softmax(logits_base[batch_idx, last_idx], dim=-1)
    log_p = F.log_softmax(logits_aligned[batch_idx, last_idx], dim=-1)
    p = log_p.exp()

    kl_last = (p * (log_p - log_q)).sum(dim=-1)

    return kl_last.mean()



if __name__ == "__main__":
    main()