"""
Average the per-seed outputs of graph.py across seeds (option "c" from the
Gini-averaging discussion: average the already per-matrix-averaged,
normalized curves).

This is mathematically equivalent to averaging the final Gini coefficients
(option "d"), since normalization already happened per-seed inside graph.py
before this script runs, and everything downstream (mean over matrices, mean
over seeds, trapz-based Gini) is linear and therefore commutes. cum_at_10 is
also a linear read of the curve so it commutes too. crossing_idx_0.8 does
NOT commute (it's a threshold on a nonlinear read) -- so it is always
recomputed from the averaged curve here, never averaged directly from the
per-seed CSV values.

Reads cum_*, lorenz_*, sum_* .pt files that graph.py already wrote for each
seed, averages them elementwise per shape, re-derives Gini/crossing_idx/
cum_at_10 from the averaged curves, cross-checks Gini and cum_at_10 against
the mean of the per-seed values in --graph-csv (should match to numerical
precision), and re-plots the averaged curves.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Base fine-tune name as it appears in the config (e.g. excepto64/lox_Llama-3.2-1B_hhrlhf_r0_1e), without the _s{seed} suffix measure_update.sh appends.")
parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds to average over, e.g. --seeds 0 1 2.")
parser.add_argument(
    "--shapes", type=str, nargs="+", default=["2048,2048"],
    help='Weight-matrix shapes to average, e.g. --shapes "2048,2048" "512,2048" "2048,8192".',
)
parser.add_argument("--suffix", type=str, default="", help="e.g. dWX to average the dWX-tagged curves instead of the weight-SVD curves.")
parser.add_argument("--revision", type=str, default=None, help="HF revision (e.g. checkpoint step tag) used when the per-seed curves were written.")
parser.add_argument("--graph-csv", type=str, default="graph_out.csv", help="CSV written by graph.py, used to cross-check the recomputed Gini/cum_at_10 against the mean of the per-seed values.")
parser.add_argument("--out", type=str, default="graph_out_seed_avg.csv", help="CSV file to append the seed-averaged summary numbers to.")

args = parser.parse_args()

def parse_shape(s):
    return tuple(sorted(int(x) for x in s.split(",")))

shapes = [parse_shape(s) for s in args.shapes]
tag = f"_{args.suffix}" if args.suffix else ""

def model_local_for_seed(seed):
    base = args.model.split("/")[-1]
    local = f"{base}_s{seed}"
    if args.revision:
        local += f"_{args.revision.replace('/', '-')}"
    return local

avg_model_local = f"{args.model.split('/')[-1]}_savg-{'-'.join(str(s) for s in args.seeds)}"
if args.revision:
    avg_model_local += f"_{args.revision.replace('/', '-')}"

def shape_tag(shape):
    return "x".join(str(d) for d in shape)

def normalized_x(n):
    return torch.linspace(0, 1, n)

def gini_coefficient(lorenz_y, x = None):
    if x is None:
        x = normalized_x(len(lorenz_y))
    area_under_curve = torch.trapz(lorenz_y, x)
    return 1 - 2 * area_under_curve.item()

def write_result(row):
    file_exists = os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "revision", "suffix", "seeds", "series", "metric", "value"])
        writer.writerow([args.model, args.revision or "", args.suffix, "-".join(str(s) for s in args.seeds), *row])

def load_csv_values(model_local, label, metric):
    if not os.path.exists(args.graph_csv):
        return None
    values = []
    with open(args.graph_csv, newline="") as f:
        for row in csv.DictReader(f):
            if (
                row["model"].split("/")[-1] == model_local
                and row["revision"] == (args.revision or "")
                and row["suffix"] == args.suffix
                and row["series"] == label
                and row["metric"] == metric
            ):
                try:
                    values.append(float(row["value"]))
                except ValueError:
                    pass
    return values

def average_curves(prefix):
    """Load {prefix}_{shape}_{model_local}{tag}.pt for every seed and shape, average elementwise."""
    averaged = {}
    for shape in shapes:
        label = shape_tag(shape)
        curves = []
        for seed in args.seeds:
            path = f"{prefix}_{label}_{model_local_for_seed(seed)}{tag}.pt"
            curves.append(torch.load(path, weights_only=True))
        averaged[shape] = torch.stack(curves).mean(dim=0)
    return averaged

def main():
    avg_cum = average_curves("cum")
    avg_lorenz = average_curves("lorenz")
    avg_sum = average_curves("sum")

    plt.figure(0, figsize=(10, 6))
    for shape in shapes:
        label = shape_tag(shape)
        cum = avg_cum[shape]
        torch.save(cum, f"cum_{label}_{avg_model_local}{tag}.pt")
        idx = torch.where(cum > 0.8)[0]
        crossing_idx = idx[0].item() if len(idx) else "never crosses 0.8"
        cum_at_10 = cum[10].item()
        write_result([label, "crossing_idx_0.8", crossing_idx])
        write_result([label, "cum_at_10", cum_at_10])
        # cross-check against the mean of the per-seed values actually recorded in graph_out.csv
        per_seed_vals = []
        for seed in args.seeds:
            vals = load_csv_values(f"{args.model.split('/')[-1]}_s{seed}", label, "cum_at_10")
            per_seed_vals += vals or []
        if per_seed_vals:
            mean_per_seed = sum(per_seed_vals) / len(per_seed_vals)
            print(f"[{label}] cum_at_10: from averaged curve = {cum_at_10:.6f}, mean of per-seed = {mean_per_seed:.6f} (should match)")
        plt.plot(cum.numpy(), label = label)
    plt.xlabel("Singular Values")
    plt.ylabel("Cumulative Proportion")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"cumulative_proportion_{avg_model_local}{tag}.pdf")

    plt.figure(2, figsize=(10, 6))
    for shape in shapes:
        label = shape_tag(shape)
        cum = avg_lorenz[shape]
        gini = gini_coefficient(cum)
        torch.save(cum, f"lorenz_{label}_{avg_model_local}{tag}.pt")
        write_result([label, "gini", gini])
        per_seed_vals = []
        for seed in args.seeds:
            vals = load_csv_values(f"{args.model.split('/')[-1]}_s{seed}", label, "gini")
            per_seed_vals += vals or []
        if per_seed_vals:
            mean_per_seed = sum(per_seed_vals) / len(per_seed_vals)
            print(f"[{label}] gini: from averaged curve = {gini:.6f}, mean of per-seed = {mean_per_seed:.6f} (should match)")
        plt.plot(normalized_x(len(cum)).numpy(), cum.numpy(), label = f"{label} (Gini = {gini:.3f})")
    plt.plot([0, 1], [0, 1], linestyle = "--", color = "gray", label = "Equality")
    plt.xlabel("Cumulative Share of Singular Values")
    plt.ylabel("Cumulative Proportion")
    plt.legend()
    plt.savefig(f"lorenz_curve_{avg_model_local}{tag}.pdf")

    plt.figure(1, figsize=(10, 6))
    for shape in shapes:
        label = shape_tag(shape)
        avg = avg_sum[shape]
        torch.save(avg, f"sum_{label}_{avg_model_local}{tag}.pt")
        plt.plot(avg.numpy(), label = label, linewidth = 2, marker = "o", markersize = 4)
    plt.xscale("log")
    plt.xlabel("Singular Values")
    plt.ylabel("Average Singular Value")
    plt.legend()
    plt.savefig(f"average_singular_value_{avg_model_local}{tag}.pdf")

if __name__ == "__main__":
    main()
