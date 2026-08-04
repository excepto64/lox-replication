import argparse
import csv
import os

import matplotlib.pyplot as plt
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--n-main", type=int, default=2048) # Number of main singular values to consider for extrapolation.
parser.add_argument("--n-sec", type=int, default=0) # Number of extra singular values to consider for extrapolation.
parser.add_argument("--suffix", type=str, default="") # e.g. "dWX" to read SVD_coeffs_dWX_{model}.pt instead of SVD_coeffs_{model}.pt, and tag outputs accordingly.
parser.add_argument("--revision", type=str, default=None, help="HF revision (e.g. checkpoint step tag) --model was loaded at; must match the tag used when writing SVD_coeffs_*.pt.")
parser.add_argument("--svd-ylim-max", type=float, default=None, help="Fixed y-axis max for the average-singular-value plot (lower bound is always 0); keeps scale consistent across steps/checkpoints.")
parser.add_argument("--out", type=str, default="graph_out.csv", help="CSV file to append the computed summary numbers (Gini, crossing index, cum-at-10) to.")

args = parser.parse_args()
model_local = args.model.split('/')[-1]
if args.revision:
    model_local += f"_{args.revision.replace('/', '-')}"
tag = f"_{args.suffix}" if args.suffix else ""

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

def average_by_size(svd_coeffs, n, transform = None):
    total = torch.zeros(n)
    count = 0
    for coeff in svd_coeffs:
        if coeff.size()[0] == n:
            count += 1
            total += transform(coeff) if transform else coeff
    return total / count

def plot_cum(svd_coeffs):
    plt.figure(0, figsize=(10, 6))

    cum_main = average_by_size(svd_coeffs, args.n_main, transform = find_cum)
    torch.save(cum_main, f"cum_main_{model_local}{tag}.pt")
    idx = torch.where(cum_main > 0.8)[0]
    crossing_idx = idx[0].item() if len(idx) else "never crosses 0.8"
    print(crossing_idx, cum_main[10])
    write_result(["main", "crossing_idx_0.8", crossing_idx])
    write_result(["main", "cum_at_10", cum_main[10].item()])

    if args.n_sec > 0:
        cum_sec = average_by_size(svd_coeffs, args.n_sec, transform = find_cum)
        torch.save(cum_sec, f"cum_sec_{model_local}{tag}.pt")
        idx = torch.where(cum_sec > 0.8)[0]
        crossing_idx = idx[0].item() if len(idx) else "never crosses 0.8"
        print(crossing_idx, cum_sec[10])
        write_result(["secondary", "crossing_idx_0.8", crossing_idx])
        write_result(["secondary", "cum_at_10", cum_sec[10].item()])
        plt.plot(cum_sec.numpy(), label = "Secondary")

    plt.plot(cum_main.numpy(), label = "Main")
    plt.xlabel("Singular Values")
    plt.ylabel("Cumulative Proportion")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"cumulative_proportion_{model_local}{tag}.pdf")

def plot_lorenz(svd_coeffs):
    plt.figure(2, figsize=(10, 6))

    cum_main = average_by_size(svd_coeffs, args.n_main, transform = lorenz_curve)
    gini_main = gini_coefficient(cum_main)
    torch.save(cum_main, f"lorenz_main_{model_local}{tag}.pt")
    print(f"Gini (main): {gini_main}")
    write_result(["main", "gini", gini_main])

    if args.n_sec > 0:
        cum_sec = average_by_size(svd_coeffs, args.n_sec, transform = lorenz_curve)
        gini_sec = gini_coefficient(cum_sec)
        torch.save(cum_sec, f"lorenz_sec_{model_local}{tag}.pt")
        print(f"Gini (secondary): {gini_sec}")
        write_result(["secondary", "gini", gini_sec])
        plt.plot(normalized_x(len(cum_sec)).numpy(), cum_sec.numpy(), label = f"Secondary (Gini = {gini_sec:.3f})")

    plt.plot(normalized_x(len(cum_main)).numpy(), cum_main.numpy(), label = f"Main (Gini = {gini_main:.3f})")
    plt.plot([0, 1], [0, 1], linestyle = "--", color = "gray", label = "Equality")
    plt.xlabel("Cumulative Share of Singular Values")
    plt.ylabel("Cumulative Proportion")
    plt.legend()
    plt.savefig(f"lorenz_curve_{model_local}{tag}.pdf")

def plot_svd(svd_coeffs):
    plt.figure(1, figsize=(10, 6))

    sum_main = average_by_size(svd_coeffs, args.n_main)
    torch.save(sum_main, f"sum_main_{model_local}{tag}.pt")

    if args.n_sec > 0:
        sum_sec = average_by_size(svd_coeffs, args.n_sec)
        torch.save(sum_sec, f"sum_sec_{model_local}{tag}.pt")
        plt.plot(sum_sec.numpy(), label = "Secondary", color = "blue", linewidth = 2, marker = "o", markersize = 4)

    plt.plot(sum_main.numpy(), label = "Main", color = "orange", linewidth = 2, marker = "o", markersize = 4)
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