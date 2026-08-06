"""Compare a base model and a target model (e.g. an aligned or rank-extracted
variant) by mean KL divergence, per dataset.

For datasets with a real prompt/response split (hh-rlhf, alpaca), KL is
averaged over the response tokens only. For prompt-only datasets (wikipedia,
overrefusal, overrefusal-toxic), KL falls back to the last token, since there
is no response span to average over.

Results are appended as rows to --out (default kl_out.csv):
    base_model, model, dataset, mean_kl

Example:
    python src/kl_compare.py --base-model <base> --model <target> \\
        --dataset hh-rlhf --limit 20 --batch-size 10
"""

import argparse
import csv
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _hh_rlhf_examples():
    examples = []
    for chosen in load_dataset("Anthropic/hh-rlhf", split="test")["chosen"][:1000]:
        marker = "\n\nAssistant:"
        prompt, _, response = chosen.rpartition(marker)
        if not prompt:
            continue
        examples.append({"prompt": prompt + marker, "response": response})
    return examples

def _alpaca_examples():
    ds = load_dataset("tatsu-lab/alpaca", split="train")[:1000]
    return [
        {"prompt": text[: len(text) - len(output)], "response": output}
        for text, output in zip(ds["text"], ds["output"])
    ]

def _wikipedia_examples():
    return [
        {"prompt": t.split("\n\n")[0], "response": ""}
        for t in load_dataset("wikimedia/wikipedia", "20231101.en", split="train")["text"][:1000]
    ]

def _or_bench_examples(config):
    return [
        {"prompt": p, "response": ""}
        for p in load_dataset("bench-llm/or-bench", config, split="train")["prompt"][:1000]
    ]

DATASETS = {
    "hh-rlhf": _hh_rlhf_examples,
    "alpaca": _alpaca_examples,
    "wikipedia": _wikipedia_examples,
    "overrefusal": lambda: _or_bench_examples("or-bench-hard-1k"),
    "overrefusal-toxic": lambda: _or_bench_examples("or-bench-toxic"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--dataset", type=str, choices=list(DATASETS), default="hh-rlhf")
parser.add_argument("--limit", type=int, default=None, help="Cap the number of prompts (for quick testing).")
parser.add_argument("--batch-size", type=int, default=16, help="Prompts per forward-pass batch.")
parser.add_argument("--max-length", type=int, default=512)
parser.add_argument("--out", type=str, default="kl_out.csv")
parser.add_argument("--revision", type=str, default=None, help="HF revision (e.g. checkpoint step tag) of --model to load.")
parser.add_argument("--base-revision", type=str, default=None, help="HF revision (e.g. checkpoint step tag) of --base-model to load.")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # Load the models
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.base_revision)
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.base_model, revision=args.base_revision, dtype=torch.float32, device_map=device)
    aligned_model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.float32, device_map=device)

    inputs = DATASETS[args.dataset]()
    if args.limit is not None:
        inputs = inputs[:args.limit]

    kl_sum = 0.0
    for i in tqdm(range(0, len(inputs), args.batch_size), ncols=90):
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
            writer.writerow(["base_model", "model", "revision", "dataset", "mean_kl"])
        writer.writerow([args.base_model, args.model, args.revision or "", dataset_name, mean_kl])

def compute_kl_divergence(tokenizer, pretrained_model, aligned_model, batch):
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    full_texts = [ex["prompt"] + ex["response"] for ex in batch]
    inputs = tokenizer(
        full_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length
    )
    seq_lens = inputs["attention_mask"].sum(dim=1)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits_base = pretrained_model(**inputs).logits
        logits_aligned = aligned_model(**inputs).logits

    log_q = F.log_softmax(logits_base, dim=-1)
    log_p = F.log_softmax(logits_aligned, dim=-1)
    del logits_base, logits_aligned
    kl_per_token = (log_p.exp() * (log_p - log_q)).sum(dim=-1)  # (batch, seq)
    del log_p, log_q
    # mask selects response-token positions (or the last prompt token if there's no response)
    mask = torch.zeros_like(kl_per_token, dtype=torch.bool)
    for i, ex in enumerate(batch):
        end = seq_lens[i].item()
        if ex["response"]:
            prompt_len = len(tokenizer(ex["prompt"], truncation=True, max_length=args.max_length)["input_ids"])
            start = min(prompt_len, end - 1)
            mask[i, start:end] = True
        else:
            mask[i, end - 1] = True

    per_example_kl = (kl_per_token * mask).sum(dim=1) / mask.sum(dim=1)
    return per_example_kl.mean()



if __name__ == "__main__":
    main()