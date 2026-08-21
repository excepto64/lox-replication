"""Scan a run's SVD_coeffs_*.pt files across steps and print the global max
average singular value, for use as graph.py's --svd-ylim-max so the
average-singular-value plot keeps a fixed scale across all steps.

Reimplements graph.py's average_by_shape on the raw per-step tensors (not the
plotted output), since this needs to run before graph.py does.
"""

import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--fine-tune-name", type=str, required=True, help="Local run name (e.g. measure_update.sh's local_name), without the trailing _step-N revision.")
parser.add_argument("--steps", type=int, nargs="+", required=True, help="Checkpoint step numbers to scan, e.g. 30 60 90.")
parser.add_argument("--suffix", type=str, default="", help="e.g. dWX to scan SVD_coeffs_{name}_step-N_dWX.pt instead of SVD_coeffs_{name}_step-N.pt.")
parser.add_argument(
    "--shapes", type=str, nargs="+", default=["2048,2048"],
    help='Weight-matrix shapes to scan, e.g. --shapes "2048,2048" "512,2048" "2048,8192".',
)
args = parser.parse_args()

tag = f"_{args.suffix}" if args.suffix else ""

def parse_shape(s):
    """Parse a "d1,d2" CLI shape string into a sorted (min_dim, max_dim) tuple."""
    return tuple(sorted(int(x) for x in s.split(",")))

shapes = [parse_shape(s) for s in args.shapes]

def entry_shape(entry):
    """Weight-space entries are {"shape": ..., "S": ...} dicts. Legacy activation-space
    (dWX) entries are bare S tensors, saved before the dict format existed; keyed by
    their own length instead, matching graph.py's entry_shape."""
    return tuple(sorted(entry["shape"])) if isinstance(entry, dict) else (entry.shape[0],)

def entry_coeff(entry):
    """Return an entry's singular-value tensor, regardless of dict or legacy bare-tensor format."""
    return entry["S"] if isinstance(entry, dict) else entry

def average_by_shape(svd_coeffs, shape):
    """Average the singular-value spectra of all entries matching `shape`, elementwise."""
    n = min(shape)
    total = torch.zeros(n)
    count = 0
    for entry in svd_coeffs:
        if entry_shape(entry) == shape:
            count += 1
            total += entry_coeff(entry)
    return total / count if count else total

global_max = 0.0
for step in args.steps:
    model_local = f"{args.fine_tune_name}_step-{step}"
    svd_coeffs = torch.load(f"SVD_coeffs_{model_local}{tag}.pt", weights_only=True)
    for shape in shapes:
        avg = average_by_shape(svd_coeffs, shape)
        global_max = max(global_max, avg.max().item())

print(global_max)
