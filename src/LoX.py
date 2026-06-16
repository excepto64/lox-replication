"""
Modified from LoX paper.
"""

from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch
from tqdm import tqdm 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--k", type=int, default=0) # Top-ranks to extrapolate. k=0 extrapolates full rank.
parser.add_argument("--coef", type=float, default=1.) # Extrapolation coefficient
parser.add_argument("--lora", type=int, default=0)

args = parser.parse_args()
print(args)

def main():
    k = args.k

    # Load the models
    #tokenizer = AutoTokenizer.from_pretrained(aligned_path)
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    if args.lora > 0:
        aligned_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
        aligned_model = PeftModel.from_pretrained(aligned_model, args.model)
        aligned_model = aligned_model.merge_and_unload()
    else:
        aligned_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    remove = ["model.embed_tokens.weight", "input_layernorm.weight", "post_attention_layernorm.weight", "model.norm.weight", "lm_head.weight"]

    # Take their weights and compute difference.
    W_aligned = aligned_model.state_dict()
    del aligned_model
    W_base = pretrained_model.state_dict()
    del pretrained_model

    for layer in list(W_base.keys()):
        for name in remove:
            if name in layer:
                del W_base[layer]
                del W_aligned[layer]

    dW_aligned = {}

    output = []

    for name in tqdm(W_aligned):
        dW_aligned[name] = W_aligned[name] - W_base[name]
        if len(dW_aligned[name].size()) > 1:
            U, S, Vt = torch.linalg.svd(dW_aligned[name].float(), full_matrices = False)
            output.append(S)
        del W_base[name], dW_aligned[name]
    out_name = f"SVD_coeffs_{args.model.split('/')[-1]}.pt"
    torch.save(output, out_name)

if __name__ == "__main__":
    main()