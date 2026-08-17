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
    return "x".join(str(d) for d in shape)

def write_result(row):
    file_exists = os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "revision", "suffix", "series", "metric", "value"])
        writer.writerow([args.model, args.revision or "", args.suffix, *row])

def find_cum(coeff):
    total = torch.sum(coeff**2)
    if total == 0:
        return torch.ones_like(coeff)
    cum = torch.cumsum(coeff**2, dim = 0) / total
    return cum

def lorenz_curve(coeff):
    ascending = torch.flip(coeff, dims = [0])
    total = torch.sum(ascending**2)
    cum = torch.cumsum(ascending**2, dim = 0) / total
    return cum

def normalized_x(n):
    return torch.linspace(0, 1, n)

def gini_coefficient(lorenz_y, x = None):
    if x is None:
        x = normalized_x(len(lorenz_y))
    area_under_curve = torch.trapz(lorenz_y, x)
    return 1 - 2 * area_under_curve.item()

def average_by_shape(svd_coeffs, shape, transform = None):
    n = min(shape)
    total = torch.zeros(n)
    count = 0
    for entry in svd_coeffs:
        if tuple(entry["shape"]) == shape:
            coeff = entry["S"]
            count += 1
            total += transform(coeff) if transform else coeff
    return total / count if count else total

def plot_cum(svd_coeffs):
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
    svd_coeffs = torch.load(f"SVD_coeffs_{model_local}{tag}.pt", weights_only = True)

    plot_cum(svd_coeffs)

    plot_lorenz(svd_coeffs)

    plot_svd(svd_coeffs)

if __name__ == "__main__":
    main()
