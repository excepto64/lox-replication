"""
Average the per-seed outputs of graph.py across seeds: average the already 
per-matrix-averaged, normalized curves.

This is mathematically equivalent to averaging the final Gini coefficients, 
since normalization already happened per-seed inside graph.py
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

Pass --steps to loop over a whole run's checkpoints automatically (revision
is built as step-{N} for each N), matching measure_update.sh's step
numbering; pass --revision for a single checkpoint (or omit both for
un-revisioned models).
"""

import argparse
import csv
import math
import os
import statistics

import matplotlib.pyplot as plt
import torch

# Two-tailed 95% critical values of the Student's t distribution, keyed by
# degrees of freedom (n_seeds - 1). Falls back to the normal-approximation
# value (1.96) for df not in the table (i.e. large seed counts).
T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042,
}

def std_and_ci95(values):
    """Sample standard deviation and 95% CI half-width (mean +/- ci95) over seeds."""
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    std = statistics.stdev(values)
    df = n - 1
    t_crit = T_CRIT_95.get(df, 1.96)
    ci95 = t_crit * std / math.sqrt(n)
    return std, ci95

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Base fine-tune name as it appears in the config (e.g. excepto64/lox_Llama-3.2-1B_hhrlhf_r0_1e), without the _s{seed} suffix measure_update.sh appends.")
parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds to average over, e.g. --seeds 0 1 2.")
parser.add_argument(
    "--shapes", type=str, nargs="+", default=["2048,2048"],
    help='Weight-matrix shapes to average, e.g. --shapes "2048,2048" "512,2048" "2048,8192".',
)
parser.add_argument("--suffix", type=str, default="", help="e.g. dWX to average the dWX-tagged curves instead of the weight-SVD curves.")
revision_group = parser.add_mutually_exclusive_group()
revision_group.add_argument("--revision", type=str, default=None, help="HF revision (e.g. checkpoint step tag) used when the per-seed curves were written, for a single checkpoint.")
revision_group.add_argument("--steps", type=int, nargs="+", default=[30, 60, 90, 120, 150, 180, 210, 240], help="Checkpoint step numbers to average automatically. Revision for step N is read as step-N, matching measure_update.sh. Default: 30 60 ... 240.")
parser.add_argument("--graph-csv", type=str, default="graph_out.csv", help="CSV written by graph.py, used to cross-check the recomputed Gini/cum_at_10 against the mean of the per-seed values.")
parser.add_argument("--out", type=str, default="graph_out_seed_ave.csv", help="CSV file to append the seed-averaged summary numbers to.")

args = parser.parse_args()

def parse_shape(s):
    """Parse a "d1,d2" CLI shape string into a sorted (min_dim, max_dim) tuple."""
    return tuple(sorted(int(x) for x in s.split(",")))

shapes = [parse_shape(s) for s in args.shapes]
tag = f"_{args.suffix}" if args.suffix else ""

if args.revision:
    revisions = [args.revision]
else:
    revisions = [f"step-{s}" for s in args.steps]

def model_local_for_seed(seed, revision):
    """Reconstruct the model_local name graph.py used for one seed/revision's saved curves."""
    base = args.model.split("/")[-1]
    local = f"{base}_s{seed}"
    if revision:
        local += f"_{revision.replace('/', '-')}"
    return local

def avg_model_local_for(revision):
    """Build the output name for a seed-averaged run, e.g. "<model>_savg-0-2-26[_step-N]"."""
    local = f"{args.model.split('/')[-1]}_savg-{'-'.join(str(s) for s in args.seeds)}"
    if revision:
        local += f"_{revision.replace('/', '-')}"
    return local

def shape_tag(shape):
    """Format a shape tuple as a filename-safe label, e.g. (512, 2048) -> "512x2048"."""
    return "x".join(str(d) for d in shape)

def normalized_x(n):
    """n evenly spaced points on [0, 1], for plotting/integrating a Lorenz curve of length n."""
    return torch.linspace(0, 1, n)

def gini_coefficient(lorenz_y, x = None):
    """Gini coefficient (1 - 2 * area under the Lorenz curve) of the averaged low-rankedness curve."""
    if x is None:
        x = normalized_x(len(lorenz_y))
    area_under_curve = torch.trapz(lorenz_y, x)
    return 1 - 2 * area_under_curve.item()

def write_result(revision, label, metric, value, std = "", ci95 = ""):
    """Append one seed-averaged metric row to args.out, writing a header first if the file is new."""
    file_exists = os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "revision", "suffix", "series", "metric", "value", "std", "ci95"])
        writer.writerow([args.model, revision or "", args.suffix, label, metric, value, std, ci95])

def load_csv_values(model_local, revision, label, metric):
    """Returns the single value for this (model, revision, suffix, series, metric) key.
    graph_out.csv can contain duplicate rows (graph.py re-run without clearing the
    file), so only the first match is used rather than accumulating repeats."""
    if not os.path.exists(args.graph_csv):
        return None
    with open(args.graph_csv, newline="") as f:
        for row in csv.DictReader(f):
            if (
                row["model"].split("/")[-1] == model_local
                and row["revision"] == (revision or "")
                and row["suffix"] == args.suffix
                and row["series"] == label
                and row["metric"] == metric
            ):
                try:
                    return [float(row["value"])]
                except ValueError:
                    pass
    return []

def average_curves(prefix, revision):
    """Load {prefix}_{shape}_{model_local}{tag}.pt for every seed and shape, average elementwise."""
    averaged = {}
    for shape in shapes:
        label = shape_tag(shape)
        curves = []
        for seed in args.seeds:
            path = f"{prefix}_{label}_{model_local_for_seed(seed, revision)}{tag}.pt"
            curves.append(torch.load(path, weights_only=True))
        averaged[shape] = torch.stack(curves).mean(dim=0)
    return averaged

def run_for_revision(revision):
    """Average this revision's per-seed cum/lorenz/sum curves, re-derive Gini/crossing_idx/cum_at_10, cross-check against graph_out.csv, write results, and re-plot."""
    avg_model_local = avg_model_local_for(revision)

    avg_cum = average_curves("cum", revision)
    avg_lorenz = average_curves("lorenz", revision)
    avg_sum = average_curves("sum", revision)

    plt.figure(0, figsize=(10, 6))
    for shape in shapes:
        label = shape_tag(shape)
        cum = avg_cum[shape]
        torch.save(cum, f"cum_{label}_{avg_model_local}{tag}.pt")
        idx = torch.where(cum > 0.8)[0]
        crossing_idx = idx[0].item() if len(idx) else "never crosses 0.8"
        cum_at_10 = cum[10].item()

        # cross-check against the mean of the per-seed values actually recorded in graph_out.csv,
        # and report the seed-to-seed spread (std, 95% CI) alongside the mean.
        crossing_seed_vals = []
        for seed in args.seeds:
            vals = load_csv_values(f"{args.model.split('/')[-1]}_s{seed}", revision, label, "crossing_idx_0.8")
            crossing_seed_vals += [v for v in (vals or []) if not isinstance(v, str)]
        crossing_std, crossing_ci95 = std_and_ci95(crossing_seed_vals) if len(crossing_seed_vals) >= 2 else ("", "")
        write_result(revision, label, "crossing_idx_0.8", crossing_idx, crossing_std, crossing_ci95)

        per_seed_vals = []
        for seed in args.seeds:
            vals = load_csv_values(f"{args.model.split('/')[-1]}_s{seed}", revision, label, "cum_at_10")
            per_seed_vals += vals or []
        if per_seed_vals:
            mean_per_seed = sum(per_seed_vals) / len(per_seed_vals)
            print(f"[{revision or 'no-revision'}][{label}] cum_at_10: from averaged curve = {cum_at_10:.6f}, mean of per-seed = {mean_per_seed:.6f} (should match)")
        std, ci95 = std_and_ci95(per_seed_vals) if len(per_seed_vals) >= 2 else ("", "")
        write_result(revision, label, "cum_at_10", cum_at_10, std, ci95)
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
        per_seed_vals = []
        for seed in args.seeds:
            vals = load_csv_values(f"{args.model.split('/')[-1]}_s{seed}", revision, label, "gini")
            per_seed_vals += vals or []
        if per_seed_vals:
            mean_per_seed = sum(per_seed_vals) / len(per_seed_vals)
            print(f"[{revision or 'no-revision'}][{label}] gini: from averaged curve = {gini:.6f}, mean of per-seed = {mean_per_seed:.6f} (should match)")
        std, ci95 = std_and_ci95(per_seed_vals) if len(per_seed_vals) >= 2 else ("", "")
        write_result(revision, label, "gini", gini, std, ci95)
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

    plt.close("all")

def main():
    """Run run_for_revision for every requested revision (a single --revision, or one per --steps)."""
    for revision in revisions:
        run_for_revision(revision)

if __name__ == "__main__":
    main()
