"""
Extract input activations X_in from the aligned model on AdvBench calibration
data, then compute the SVD spectrum of dW @ X_in^T (dW = W_aligned - W_base),
following the activation-extraction approach of Wei et al. 2024
(https://github.com/boyiwei/alignment-attribution-code).

Output format matches extract_ranks.py's SVD_coeffs_{model}.pt (a list of 1D
singular-value tensors, one per 2D weight matrix) so it can be consumed by
graph.py unchanged.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--data-path", type=str, default="harmful_behaviors.csv")
parser.add_argument("--n", type=int, default=128, help="Number of AdvBench prompts used for calibration.")
parser.add_argument("--max-new-tokens", type=int, default=60)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--max-length", type=int, default=512)
parser.add_argument("--out", type=str, default=None)

args = parser.parse_args()
print(args)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_local = args.model.split("/")[-1]
if args.out is None:
    args.out = f"SVD_coeffs_dWX_{model_local}.pt"

REMOVE = ["model.embed_tokens.weight", "input_layernorm.weight", "post_attention_layernorm.weight", "model.norm.weight", "lm_head.weight"]


def build_prompt(tokenizer, goal):
    messages = [{"role": "user", "content": goal}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_responses(model, tokenizer, goals):
    """Greedy-decode the aligned model's own response to each AdvBench goal."""
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    examples = []
    for goal in tqdm(goals, desc="Generating calibration responses", ncols=90):
        prompt = build_prompt(tokenizer, goal)
        model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response_ids = out[0][model_inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        examples.append({"prompt": prompt, "response": response})
    return examples


def find_linear_layers(model):
    """All nn.Linear submodules inside the transformer's decoder layers, keyed
    by their full dotted name (matches state_dict keys used for dW)."""
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(r in f"{name}.weight" for r in REMOVE):
            layers[name] = module
    return layers


def collect_activations(model, tokenizer, examples, linear_layers):
    """Forward the aligned model over the calibration examples, recording each
    linear layer's input activations restricted to response-token positions."""
    buffers = {name: [] for name in linear_layers}
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            buffers[name].append(inp[0].detach())
        return hook

    for name, module in linear_layers.items():
        handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    try:
        for i in tqdm(range(0, len(examples), args.batch_size), desc="Collecting activations", ncols=90):
            batch = examples[i:i + args.batch_size]
            full_texts = [ex["prompt"] + ex["response"] for ex in batch]
            inputs = tokenizer(
                full_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length
            ).to(device)
            seq_lens = inputs["attention_mask"].sum(dim=1)

            mask = torch.zeros(inputs["input_ids"].shape, dtype=torch.bool)
            for j, ex in enumerate(batch):
                end = seq_lens[j].item()
                prompt_len = len(tokenizer(ex["prompt"], truncation=True, max_length=args.max_length)["input_ids"])
                start = min(prompt_len, end - 1)
                mask[j, start:end] = True
            mask = mask.to(device)

            for name in buffers:
                buffers[name].clear()

            with torch.no_grad():
                model(**inputs)

            for name in linear_layers:
                acts = buffers[name][-1]  # (batch, seq, d_in) captured by the hook for this batch
                m = mask
                if acts.dim() == 3:
                    flat_acts = acts.reshape(-1, acts.shape[-1])
                    flat_mask = m.reshape(-1)
                else:
                    flat_acts = acts
                    flat_mask = m.reshape(-1)
                buffers[name][-1] = flat_acts[flat_mask].cpu()
    finally:
        for h in handles:
            h.remove()

    return {name: torch.cat(chunks, dim=0) for name, chunks in buffers.items() if chunks}


def main():
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    aligned_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32)

    goals = list(pd.read_csv(args.data_path)["goal"])[:args.n]
    examples = generate_responses(aligned_model, tokenizer, goals)

    W_aligned = aligned_model.state_dict()
    W_base = base_model.state_dict()
    del base_model

    dW = {}
    for name in list(W_aligned.keys()):
        if any(r in name for r in REMOVE):
            continue
        if len(W_aligned[name].size()) > 1:
            dW[name] = (W_aligned[name] - W_base[name]).cpu()
    del W_base

    linear_layers = find_linear_layers(aligned_model)
    activations = collect_activations(aligned_model, tokenizer, examples, linear_layers)

    output = []
    for name, module in tqdm(linear_layers.items(), desc="Computing SVD(dW @ X_in)"):
        weight_name = f"{name}.weight"
        if weight_name not in dW or name not in activations:
            continue
        X = activations[name].float().to(device)     # (n_tokens, d_in)
        dW_mat = dW[weight_name].float().to(device)   # (d_out, d_in)
        score = X @ dW_mat.T                          # (n_tokens, d_out)
        _, S, _ = torch.linalg.svd(score, full_matrices=False)
        output.append(S.cpu())
        del X, dW_mat, score

    torch.save(output, args.out)
    print(f"Saved {len(output)} singular-value spectra to {args.out}")


if __name__ == "__main__":
    main()
