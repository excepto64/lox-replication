"""Load one SVD_coeffs_*.pt file (weight-space, from LoX.py, or activation-space
dWX, from extract_activations.py) and, per weight-matrix shape, plot the
cumulative proportion of singular-value energy, the Lorenz curve (+ Gini
coefficient), and the average singular value spectrum -- the low-rankedness
measurements this project uses to compare training configurations. Summary
numbers (Gini, crossing index, cum-at-10) are appended to a CSV via
write_result. Called by measure_update.sh once per checkpoint per run.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument(
    "--shapes", type=str, nargs="+", default=["2048,2048"],
    help='Weight-matrix shapes to plot separately, e.g. --shapes "2048,2048" "512,2048" "2048,8192". '
         "Each is matched against the sorted (rows, cols) shape tagged onto every saved SVD spectrum.",
)
parser.add_argument("--suffix", type=str, default="") # e.g. "dWX" to read SVD_coeffs_dWX_{model}.pt instead of SVD_coeffs_{model}.pt, and tag outputs accordingly.
parser.add_argument("--revision", type=str, default=None, help="HF revision (e.g. checkpoint step tag) --model was loaded at; must match the tag used when writing SVD_coeffs_*.pt.")
parser.add_argument("--svd-ylim-max", type=float, default=None, help="Fixed y-axis max for the average-singular-value plot (lower bound is always 0); keeps scale consistent across steps/checkpoints.")
parser.add_argument("--out", type=str, default="graph_out.csv", help="CSV file to append the computed summary numbers (Gini, crossing index, cum-at-10) to.")

args = parser.parse_args()
model_local = args.model.split('/')[-1]
if args.revision:
    model_local += f"_{args.revision.replace('/', '-')}"
tag = f"_{args.suffix}" if args.suffix else ""

def parse_shape(s):
    return tuple(sorted(int(x) for x in s.split(",")))

shapes = [parse_shape(s) for s in args.shapes]

def shape_tag(shape):
    """Format a shape tuple as a filename-safe label, e.g. (512, 2048) -> "512x2048"."""
    return "x".join(str(d) for d in shape)

def write_result(row):
    """Append one row (prefixed with model/revision/suffix) to args.out, writing a header first if the file is new."""
    file_exists = os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "revision", "suffix", "series", "metric", "value"])
        writer.writerow([args.model, args.revision or "", args.suffix, *row])

def find_cum(coeff):
    """Cumulative proportion of squared-singular-value energy, in descending singular-value order."""
    total = torch.sum(coeff**2)
    if total == 0:
        return torch.ones_like(coeff)
    cum = torch.cumsum(coeff**2, dim = 0) / total
    return cum

def lorenz_curve(coeff):
    """Lorenz curve of squared-singular-value energy: cumulative proportion in ascending order."""
    ascending = torch.flip(coeff, dims = [0])
    total = torch.sum(ascending**2)
    cum = torch.cumsum(ascending**2, dim = 0) / total
    return cum

def normalized_x(n):
    """n evenly spaced points on [0, 1], for plotting/integrating a Lorenz curve of length n."""
    return torch.linspace(0, 1, n)

def gini_coefficient(lorenz_y, x = None):
    """Gini coefficient (1 - 2 * area under the Lorenz curve) of a low-rankedness metric; 0 = uniform, 1 = maximally concentrated."""
    if x is None:
        x = normalized_x(len(lorenz_y))
    area_under_curve = torch.trapz(lorenz_y, x)
    return 1 - 2 * area_under_curve.item()

def entry_shape(entry):
    """Weight-space entries are {"shape": ..., "S": ...} dicts. Activation-space
    (dWX) entries are bare S tensors, saved before the dict format existed;
    since SVD(dWX) rank is always d_out (calibration tokens always outnumber
    d_out -- see extract_activations.py), grouping by len(S) is equivalent
    and loses nothing, so they're keyed by their own length instead."""
    return tuple(sorted(entry["shape"])) if isinstance(entry, dict) else (entry.shape[0],)

def entry_coeff(entry):
    return entry["S"] if isinstance(entry, dict) else entry

def average_by_shape(svd_coeffs, shape, transform = None):
    """Average (optionally transform()-ed first, e.g. into a cum/Lorenz curve) the singular-value spectra of all entries matching `shape`, elementwise."""
    n = min(shape)
    total = torch.zeros(n)
    count = 0
    for entry in svd_coeffs:
        if entry_shape(entry) == shape:
            coeff = entry_coeff(entry)
            count += 1
            total += transform(coeff) if transform else coeff
    return total / count if count else total

def plot_cum(svd_coeffs):
    """Plot cumulative singular-value energy per shape and log its 0.8-crossing index and value at rank 10."""
    plt.figure(0, figsize=(10, 6))

    for shape in shapes:
        label = shape_tag(shape)
        cum = average_by_shape(svd_coeffs, shape, transform = find_cum)
        torch.save(cum, f"cum_{label}_{model_local}{tag}.pt")
        idx = torch.where(cum > 0.8)[0]
        crossing_idx = idx[0].item() if len(idx) else "never crosses 0.8"
        print(label, crossing_idx, cum[10])
        write_result([label, "crossing_idx_0.8", crossing_idx])
        write_result([label, "cum_at_10", cum[10].item()])
        plt.plot(cum.numpy(), label = label)

    plt.xlabel("Singular Values")
    plt.ylabel("Cumulative Proportion")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"cumulative_proportion_{model_local}{tag}.pdf")

def plot_lorenz(svd_coeffs):
    """Plot the Lorenz curve per shape and log its Gini coefficient -- the project's core low-rankedness measurement."""
    plt.figure(2, figsize=(10, 6))

    for shape in shapes:
        label = shape_tag(shape)
        cum = average_by_shape(svd_coeffs, shape, transform = lorenz_curve)
        gini = gini_coefficient(cum)
        torch.save(cum, f"lorenz_{label}_{model_local}{tag}.pt")
        print(f"Gini ({label}): {gini}")
        write_result([label, "gini", gini])
        plt.plot(normalized_x(len(cum)).numpy(), cum.numpy(), label = f"{label} (Gini = {gini:.3f})")

    plt.plot([0, 1], [0, 1], linestyle = "--", color = "gray", label = "Equality")
    plt.xlabel("Cumulative Share of Singular Values")
    plt.ylabel("Cumulative Proportion")
    plt.legend()
    plt.savefig(f"lorenz_curve_{model_local}{tag}.pdf")

def plot_svd(svd_coeffs):
    """Plot the average singular value spectrum per shape, optionally with a fixed y-axis (--svd-ylim-max)."""
    plt.figure(1, figsize=(10, 6))

    for shape in shapes:
        label = shape_tag(shape)
        avg = average_by_shape(svd_coeffs, shape)
        torch.save(avg, f"sum_{label}_{model_local}{tag}.pt")
        plt.plot(avg.numpy(), label = label, linewidth = 2, marker = "o", markersize = 4)

    plt.xscale("log")
    plt.xlabel("Singular Values")
    plt.ylabel("Average Singular Value")
    if args.svd_ylim_max is not None:
        plt.ylim(0, args.svd_ylim_max)
    plt.legend()
    plt.savefig(f"average_singular_value_{model_local}{tag}.pdf")

def main():
    """Load this run's SVD_coeffs file and produce all three plots (cumulative, Lorenz/Gini, average singular value)."""
    svd_coeffs = torch.load(f"SVD_coeffs_{model_local}{tag}.pt", weights_only = True)

    plot_cum(svd_coeffs)

    plot_lorenz(svd_coeffs)

    plot_svd(svd_coeffs)

if __name__ == "__main__":
    main()
