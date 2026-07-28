import argparse

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--save-path", type=str, default="./output")
parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--k", type=int, default=0) # Top-ranks to extrapolate. k=0 extrapolates full rank.
parser.add_argument("--coef", type=float, default=1.) # Extrapolation coefficient
parser.add_argument("--base", action="store_true")

args = parser.parse_args()
print(args)

def main():
    k = args.k
    coef = args.coef
    aligned_path = args.model

    tokenizer = AutoTokenizer.from_pretrained(aligned_path)
    aligned_model = AutoModelForCausalLM.from_pretrained(aligned_path, dtype=torch.float32)
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32)

    remove = ["model.embed_tokens.weight", "input_layernorm.weight", "post_attention_layernorm.weight", "model.norm.weight", "lm_head.weight"]

    W_aligned = aligned_model.state_dict()
    W_base = pretrained_model.state_dict()

    dW_aligned = {name : W_aligned[name] - W_base[name] for name in W_aligned}

    new_state_dict = {}

    for name in tqdm(dW_aligned):
        if any(r in name for r in remove):
            new_state_dict[name] = W_base[name] if args.base else W_aligned[name]
        elif len(dW_aligned[name].size()) > 1:
            if k>0: 
                U, S, Vt = torch.linalg.svd(dW_aligned[name].float(), full_matrices = False)
                S[k:] = 0
                m = U @ torch.diag(S) @ Vt
            else: # k=0 extrapolates full rank
                m = dW_aligned[name]

            if args.base:
                new_state_dict[name] = W_base[name] + coef * m
            else:
                new_state_dict[name] = W_aligned[name] + coef * m
            
        else:
            if args.base:
                new_state_dict[name] = W_base[name]
            else:
                new_state_dict[name] = W_aligned[name] 

    aligned_model.load_state_dict(new_state_dict)

    save_path = args.save_path
    repo_id = aligned_path + f"_extracted_k{k}"
    aligned_model.save_pretrained(save_path, push_to_hub=True, repo_id=repo_id)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    main()