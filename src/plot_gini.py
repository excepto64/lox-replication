"""
Plot Gini coefficient vs. training step, one line per run, faceted by weight
matrix shape. Reads the seed-averaged summary CSV produced by average_seeds.py
(model, revision, suffix, series, metric, value, std, ci95), pulling out the
"gini" rows and their ci95 column (shaded as mean +/- 95% CI, t-distribution
based so it accounts for the small number of seeds).
"""

import argparse
import csv
import re
from collections import defaultdict

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="graph_out_seed_ave.csv", help="Seed-averaged CSV written by average_seeds.py.")
parser.add_argument("--suffix", type=str, default="", help="Only plot rows matching this suffix tag (default: untagged weight-SVD curves).")
parser.add_argument("--out", type=str, default="gini_by_step.pdf", help="Output figure path.")
args = parser.parse_args()

step_re = re.compile(r"step-(\d+)")

def run_label(model):
    # e.g. "excepto64/lox_Llama-3_2-1B_r0_1e_dpo_adam" -> "dpo_adam"
    base = model.split("/")[-1]
    m = re.search(r"_(dpo|sft)_(adam|sgd)$", base)
    return f"{m.group(1)}_{m.group(2)}" if m else base

# data[shape][run][step] = (gini, ci95)
data = defaultdict(lambda: defaultdict(dict))

with open(args.csv, newline="") as f:
    reader = csv.DictReader(f)
    rows = [row for row in reader if row["suffix"] == args.suffix]

for row in rows:
    m = step_re.search(row["revision"])
    if not m or row["metric"] != "gini":
        continue
    step = int(m.group(1))
    shape = row["series"]
    run = run_label(row["model"])
    data[shape][run][step] = {
        "gini": float(row["value"]),
        "gini_ci95": float(row["ci95"]) if row["ci95"] else 0.0,
    }

shapes = sorted(data.keys())
fig, axes = plt.subplots(len(shapes), 1, figsize=(7, 4.5 * len(shapes)), sharex=True, squeeze=False)
axes = axes[:, 0]

runs = sorted({run for shape_data in data.values() for run in shape_data})
colors = plt.get_cmap("tab10").colors

for ax, shape in zip(axes, shapes):
    for i, run in enumerate(runs):
        steps_dict = data[shape].get(run, {})
        if not steps_dict:
            continue
        steps = sorted(steps_dict)
        ginis = [steps_dict[s]["gini"] for s in steps]
        ci95s = [steps_dict[s].get("gini_ci95", 0.0) for s in steps]
        color = colors[i % len(colors)]
        ax.plot(steps, ginis, label=run, color=color, marker="o", markersize=4, linewidth=2)
        ax.fill_between(
            steps,
            [g - c for g, c in zip(ginis, ci95s)],
            [g + c for g, c in zip(ginis, ci95s)],
            color=color, alpha=0.2,
        )
    ax.set_title(shape, fontsize=16)
    ax.set_ylabel("Gini coefficient", fontsize=16)
    ax.legend(fontsize=14, ncol=2)
    ax.tick_params(axis='both', labelsize=14)

axes[-1].set_xlabel("Step", fontsize=16)
fig.tight_layout()
fig.savefig(args.out)
print(f"Wrote {args.out}")
