"""
Extract input activations X_in and compute SVD(dW @ X_in^T), where
dW = W_aligned - W_base, feeding the result into graph.py's Gini/Lorenz-curve
analysis.

This is a direct adaptation of lib/model_wrapper_low.py and lib/data.py from
Wei et al. 2024 (https://github.com/boyiwei/alignment-attribution-code). The
ActLinear / make_Act / set_mask / no_act_recording / revert_Act_to_Linear
machinery below is their module-replacement approach for recording
per-layer input activations, kept close to verbatim. The get_align_loader
function reproduces their lib/data.py::get_align(disentangle=True) exactly
(tokenize prompt and response separately, mask the prompt to -100, one
example per batch).

What's different from their compute_dwx (i.e. make_low_rank): instead of
computing a low-rank *projection* of activation_norms @ W^T and using it to
edit weights in place (their pruning objective), we compute
activation_norms @ dW^T -- dW being the aligned-minus-base weight
difference, following this project's LoX.py/extract_ranks.py -- and take
its full SVD, keeping only the singular values as a measurement. No weights
are ever modified.

The calibration (prompt, response) pairs are the paper's own vendored
data/SFT_aligned_llama2-7b-chat-hf_train.csv (AdvBench_attr harmful
instructions + Llama2-7b-chat's own refusals), loaded exactly like their
"align" dataset (disentangle=True), not regenerated from whichever model is
under analysis -- this keeps the calibration distribution fixed across
every checkpoint being compared.

Output format matches extract_ranks.py's SVD_coeffs_{model}.pt (a list of 1D
singular-value tensors, one per 2D weight matrix) so it can be consumed by
graph.py unchanged.
"""

import argparse
import random
from functools import reduce

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--data-path", type=str, default="data/SFT_aligned_llama2-7b-chat-hf_train.csv")
parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples.")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--revision", type=str, default=None, help="HF revision (e.g. checkpoint step tag) of --model to load.")

args = parser.parse_args()
print(args)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_local = args.model.split("/")[-1]
if args.revision:
    model_local += f"_{args.revision.replace('/', '-')}"
if args.out is None:
    args.out = f"SVD_coeffs_{model_local}_dWX.pt"

REMOVE = ["model.embed_tokens.weight", "input_layernorm.weight", "post_attention_layernorm.weight", "model.norm.weight", "lm_head.weight"]


# ---------------------------------------------------------------------------
# lib/model_wrapper_low.py -- module-replacement activation recording.
# ---------------------------------------------------------------------------

class ActLinear(nn.Module):
    """drop in replacement of nn.Linear"""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.activation_norms = []  # offload to CPU
        self.record_activation = True
        self.mask = None

    def clear_act_buffer(self):
        self.activation_norms = []

    def forward(self, x):
        if self.record_activation:
            if self.mask is not None:
                x_ = x[self.mask]  # num * dim
            else:
                x_ = x  # bs * seq_len * dim
            self.activation_norms.append(x_.view(-1, x_.shape[-1]).cpu())  # offload to CPU.

        out = self.base(x)
        return out


class no_act_recording:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = True


class set_mask:
    def __init__(self, model, mask):
        self.model = model
        self.mask = mask

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.mask = self.mask

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.mask = None


def make_Act(model, verbose=False):
    replace_map = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(r in f"{name}.weight" for r in REMOVE):
            replace_map[name] = ActLinear(module)

    for name, module in model.named_modules():
        if verbose:
            print("current:", name)
        for k, v in replace_map.items():
            k_ = k.split(".")
            name_prefix, name_suffix = ".".join(k_[:-1]), k_[-1]
            if name_prefix == "":  # outer layer
                if name == name_suffix and verbose:
                    print(" not modifying ", name_suffix)
            elif name == name_prefix:
                if verbose:
                    print("    modifying ", name_suffix, "inside", name)
                setattr(module, name_suffix, v)
    return model


def revert_Act_to_Linear(model):
    """Reverts ActLinear modules back to their original nn.Linear layers."""
    for name, module in model.named_modules():
        if isinstance(module, ActLinear):
            linear_module = module.base
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent_module = model if parent_name == "" else reduce(getattr, parent_name.split("."), model)
            setattr(parent_module, name.split(".")[-1], linear_module)
    return model


def clear_act_buffer(act_model):
    for name, module in act_model.named_modules():
        if isinstance(module, ActLinear):
            module.clear_act_buffer()


# ---------------------------------------------------------------------------
# lib/data.py::get_align(disentangle=True) -- reproduced exactly, but reading
# from our already-vendored calibration CSV instead of calling load_dataset.
# ---------------------------------------------------------------------------

def get_align_loader(data_path, nsamples, seed, tokenizer):
    traindata = pd.read_csv(data_path)
    random.seed(seed)
    traindata_sampled = traindata.sample(n=nsamples, random_state=seed).reset_index(drop=True)

    trainloader = []
    for i in range(nsamples):
        trainenc_prompt = tokenizer(traindata_sampled["prompt"][i], return_tensors="pt")
        trainenc_response = tokenizer(traindata_sampled["response"][i], return_tensors="pt")
        inp = torch.cat((trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1)
        tar = inp.clone()
        trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
        tar[:, :trainenc_prompt_len] = -100
        trainloader.append((inp, tar))
    return trainloader


# ---------------------------------------------------------------------------
# Adapted from lib/model_wrapper_low.py::make_low_rank -- same layer-by-layer
# activation collection, but computing SVD(dW @ X_in^T) as a measurement
# instead of building a low-rank projection to edit weights with.
# ---------------------------------------------------------------------------

def compute_dWX_svd(args, model, dW, tokenizer, device):
    model = make_Act(model, verbose=False)
    model.requires_grad_(False)
    clear_act_buffer(model)

    # globally disable recording.
    for name, module in model.named_modules():
        if isinstance(module, ActLinear):
            module.record_activation = False

    print(f"loading calibration data from {args.data_path}")
    dataloader = get_align_loader(args.data_path, args.nsamples, args.seed, tokenizer)
    print("dataset loading complete")

    num_hidden_layers = model.config.num_hidden_layers
    output = []

    for layer in range(num_hidden_layers):
        layer_filter_fn = lambda x: f"layers.{layer}." in x  # noqa: B023 E731 RUF100 (hack for llama series, matches upstream)

        # enable recording for the current layer.
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                module.record_activation = True

        # forward pass and get activation records.
        with torch.no_grad():
            for inp, tar in tqdm(dataloader, desc=f"Layer {layer}: collecting activations", ncols=90):
                inp, tar = inp.to(device), tar.to(device)
                mask = tar.ne(-100)
                with set_mask(model, mask):
                    model(inp)

        # SVD(dW @ X_in^T) for each linear layer in this decoder layer.
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                weight_name = f"{name}.weight"
                if weight_name not in dW:
                    module.record_activation = False
                    module.clear_act_buffer()
                    continue

                module.activation_norms = torch.cat(module.activation_norms, dim=0).to(device)  # size * d_in
                dW_mat = dW[weight_name].to(device)  # d_out * d_in
                score = module.activation_norms.float() @ dW_mat.float().T  # (size, d_out)
                _, S, _ = torch.linalg.svd(score, full_matrices=False)
                output.append(S.cpu())
                del score

        # disable recording for the current layer.
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                module.record_activation = False
                module.clear_act_buffer()

        if torch.cuda.is_available():
            print(torch.cuda.memory_allocated() / 1024 / 1024 / 1024)

    model = revert_Act_to_Linear(model)
    model.zero_grad()  # freeze gradient to save cuda memory
    return output


def main():
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    aligned_model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.float32).to(device)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32)

    W_aligned = aligned_model.state_dict()
    W_base = base_model.state_dict()
    del base_model

    dW = {}
    for name in list(W_aligned.keys()):
        if any(r in name for r in REMOVE):
            continue
        if len(W_aligned[name].size()) > 1:
            dW[name] = (W_aligned[name].cpu() - W_base[name])
    del W_base

    output = compute_dWX_svd(args, aligned_model, dW, tokenizer, device)

    torch.save(output, args.out)
    print(f"Saved {len(output)} singular-value spectra to {args.out}")


if __name__ == "__main__":
    main()
