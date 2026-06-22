from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
args = parser.parse_args()

def main():
    # Load the models
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    aligned_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)

    inputs = load()

    mean_kl = compute_kl_divergence(tokenizer, pretrained_model, aligned_model, inputs)
    print("Mean KL divergence:", mean_kl.item())

def load():
    hh_rlhf = load_dataset("Anthropic/hh-rlhf", split="test")
    return hh_rlhf[:10]['chosen']

def compute_kl_divergence(tokenizer, pretrained_model, aligned_model, input_texts):
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True)

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