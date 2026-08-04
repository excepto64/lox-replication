"""Scan a run's SVD_coeffs_*.pt files across steps and print the global max
average singular value, for use as graph.py's --svd-ylim-max so the
average-singular-value plot keeps a fixed scale across all steps.

Reimplements graph.py's average_by_size on the raw per-step tensors (not the
plotted output), since this needs to run before graph.py does.
"""

import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--fine-tune-name", type=str, required=True, help="Local run name (e.g. measure_update.sh's local_name), without the trailing _step-N revision.")
parser.add_argument("--steps", type=int, nargs="+", required=True, help="Checkpoint step numbers to scan, e.g. 30 60 90.")
parser.add_argument("--suffix", type=str, default="", help="e.g. dWX to scan SVD_coeffs_{name}_step-N_dWX.pt instead of SVD_coeffs_{name}_step-N.pt.")
parser.add_argument("--n-main", type=int, default=2048)
parser.add_argument("--n-sec", type=int, default=0)
args = parser.parse_args()

tag = f"_{args.suffix}" if args.suffix else ""

def average_by_size(svd_coeffs, n):
    total = torch.zeros(n)
    count = 0
    for coeff in svd_coeffs:
        if coeff.size()[0] == n:
            count += 1
            total += coeff
    return total / count if count else total

global_max = 0.0
for step in args.steps:
    model_local = f"{args.fine_tune_name}_step-{step}"
    svd_coeffs = torch.load(f"SVD_coeffs_{model_local}{tag}.pt", weights_only=True)
    sum_main = average_by_size(svd_coeffs, args.n_main)
    global_max = max(global_max, sum_main.max().item())
    if args.n_sec > 0:
        sum_sec = average_by_size(svd_coeffs, args.n_sec)
        global_max = max(global_max, sum_sec.max().item())

print(global_max)
