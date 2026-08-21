"""Bar chart comparing ASR before vs. after the fine-tuning attack, for each of
the four 1B options. Bars are grouped in pairs (before/after) within four
clusters (one per option), with 95% CI error bars (t-distribution based).

Reads a CSV with columns: option, before_mean, before_ci95, after_mean, after_ci95.
"""
import argparse
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

OPTIONS = ["dpo_adam", "dpo_sgd", "sft_adam", "sft_sgd"]

# Same categorical hues used in plot_asr.py, one per option; "before" is drawn
# at reduced alpha, "after" at full opacity, so entity (option) still maps to
# a fixed color and before/after is a secondary (opacity) encoding.
COLORS = {
    "dpo_adam": "#2a78d6",
    "dpo_sgd": "#eb6834",
    "sft_adam": "#1baf7a",
    "sft_sgd": "#eda100",
}


def read_rows(csv_path):
    """Read the before/after ASR CSV into {option: (before_mean, before_ci95, after_mean, after_ci95)}."""
    rows = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows[row["option"]] = (
                float(row["before_mean"]), float(row["before_ci95"]),
                float(row["after_mean"]), float(row["after_ci95"]),
            )
    return rows


def main():
    """Plot a grouped before/after ASR bar chart, one cluster per option, with 95% CI error bars."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="asr_attack.csv", help="Source file.")
    parser.add_argument("--out", default="asr_attack.pdf", help="Filename for plot.")
    args = parser.parse_args()

    try:
        rows = read_rows(args.csv)
    except FileNotFoundError:
        print(f"{args.csv} not found. Please create it.")
        sys.exit(1)

    options = [o for o in OPTIONS if o in rows]

    x = np.arange(len(options))
    width = 0.35

    before_means = [rows[o][0] for o in options]
    before_cis = [rows[o][1] for o in options]
    after_means = [rows[o][2] for o in options]
    after_cis = [rows[o][3] for o in options]

    def clipped_yerr(means, cis):
        lower = [m - max(0.0, m - c) for m, c in zip(means, cis)]
        upper = [min(1.0, m + c) - m for m, c in zip(means, cis)]
        return [lower, upper]

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.bar(
        x - width / 2, before_means, width,
        yerr=clipped_yerr(before_means, before_cis), capsize=4,
        color=[COLORS[o] for o in options], edgecolor="#3d3d3a", linewidth=0.8,
    )
    ax.bar(
        x + width / 2, after_means, width,
        yerr=clipped_yerr(after_means, after_cis), capsize=4,
        color=[COLORS[o] for o in options], edgecolor="#3d3d3a", linewidth=0.8,
        hatch="///",
    )
    ax.bar(0, 0, color="#8a8a86", edgecolor="#3d3d3a", linewidth=0.8, label="Before attack")
    ax.bar(0, 0, color="#8a8a86", edgecolor="#3d3d3a", linewidth=0.8, hatch="///", label="After attack")

    ax.set_xticks(x)
    ax.set_xticklabels(options, fontsize=14)
    ax.set_ylabel("ASR", fontsize=16)
    ax.set_title("ASR before vs. after fine-tuning attack", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=13)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=14)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
    