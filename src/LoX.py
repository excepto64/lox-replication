"""
Modified from LoX paper.
"""

import argparse

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--k", type=int, default=0) # Top-ranks to extrapolate. k=0 extrapolates full rank.
parser.add_argument("--coef", type=float, default=1.) # Extrapolation coefficient
parser.add_argument("--lora", type=int, default=0)
parser.add_argument("--revision", type=str, default=None, help="HF revision (e.g. checkpoint step tag) of --model to load.")

args = parser.parse_args()
print(args)

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    k = args.k  # noqa: F841

    # Load the models
    #tokenizer = AutoTokenizer.from_pretrained(aligned_path)
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32, device_map=device)
    if args.lora > 0:
        aligned_model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32, device_map=device)
        aligned_model = PeftModel.from_pretrained(aligned_model, args.model, revision=args.revision)
        aligned_model = aligned_model.merge_and_unload()
    else:
        aligned_model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.float32, device_map=device)

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
            U, S, Vt = torch.linalg.svd(dW_aligned[name].float(), full_matrices = False)  # noqa: RUF059
            output.append(S.cpu())
        del W_base[name], dW_aligned[name]
    model_local = args.model.split('/')[-1]
    if args.revision:
        model_local += f"_{args.revision.replace('/', '-')}"
    out_name = f"SVD_coeffs_{model_local}.pt"
    torch.save(output, out_name)

if __name__ == "__main__":
    main()