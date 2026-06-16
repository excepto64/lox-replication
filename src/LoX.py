"""
Modified from LoX paper.
"""

from transformers import AutoModelForCausalLM
import torch
from tqdm import tqdm 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--k", type=int, default=0) # Top-ranks to extrapolate. k=0 extrapolates full rank.
parser.add_argument("--coef", type=float, default=1.) # Extrapolation coefficient

args = parser.parse_args()
print(args)

def main():
    k = args.k
    aligned_path = args.model

    # Load the models
    #tokenizer = AutoTokenizer.from_pretrained(aligned_path)
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    aligned_model = AutoModelForCausalLM.from_pretrained(aligned_path, torch_dtype=torch.bfloat16)

    print(pretrained_model)

    remove = ["model.embed_tokens.weight", "input_layernorm.weight", "post_attention_layernorm.weight", "model.norm.weight", "lm_head.weight"]

    # Take their weights and compute difference.
    W_aligned = aligned_model.state_dict()
    W_base = pretrained_model.state_dict()

    for layer in list(W_base.keys()):
        if layer in remove:
            del W_base[layer]
            del W_aligned[layer]

    dW_aligned = {name : W_aligned[name] - W_base[name] for name in W_aligned}

    output = []

    for name in tqdm(dW_aligned):
        if len(dW_aligned[name].size()) > 1:
            U, S, Vt = torch.linalg.svd(dW_aligned[name].float(), full_matrices = False)
            output.append(S)
    out_name = f"SVF_coeffs_{args.model.split('/')[-1]}.pt"
    torch.save(output, out_name)

if __name__ == "__main__":
    main()