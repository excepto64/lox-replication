#!/usr/bin/env python3
"""Line plot of ASR vs. alignment-training step, one line per 1B option, with
95% CI error bars (t-distribution based, to account for the small number of
seeds).

Reads the tidy long-format CSV produced by `average_asr.py --csv`
(columns: option, revision, step, is_attack, mean_asr, std_asr, ci95_asr, n).

The pre-training baseline (step 0) is fixed at 0.666 for every option -- it
is not read from the CSV -- and rows with is_attack=True are excluded
entirely, since that's a different measurement (ASR after an adversarial
fine-tune), not a point on the alignment-training curve.

Usage:
    python src/plot_asr.py [--csv results_asr.csv] [--out FILE]
"""
import argparse
import csv
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

BASELINE_ASR = 0.666

OPTIONS = ["dpo_adam", "dpo_sgd", "sft_adam", "sft_sgd"]

# dataviz reference palette, categorical slots 1-4 (light mode)
COLORS = {
    "dpo_adam": "#2a78d6",  # blue
    "dpo_sgd": "#eb6834",   # orange
    "sft_adam": "#1baf7a",  # aqua
    "sft_sgd": "#eda100",   # yellow
}


def read_rows(csv_path):
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["is_attack"] == "True":
                continue  # not a point on the alignment-training curve
            yield row["option"], int(row["step"]), float(row["mean_asr"]), float(row["ci95_asr"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results_asr.csv", help="Tidy ASR CSV produced by average_asr.py --csv.")
    parser.add_argument("--out", default="asr_by_step.pdf", help="Output image path.")
    args = parser.parse_args()

    try:
        rows = list(read_rows(args.csv))
    except FileNotFoundError:
        print(f"{args.csv} not found. Generate it with: python src/average_asr.py --csv > {args.csv}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("No non-attack ASR rows found in CSV.", file=sys.stderr)
        sys.exit(1)

    by_option = defaultdict(dict)
    for option, step, mean, ci in rows:
        by_option[option][step] = (mean, ci)

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for option in OPTIONS:
        points = by_option.get(option)
        if not points:
            continue
        steps = sorted(points)
        x = [0] + steps
        means = [BASELINE_ASR] + [points[s][0] for s in steps]
        cis = [0.0] + [points[s][1] for s in steps]
        # ASR is a proportion confined to [0, 1]; the CI itself is left
        # unclipped in the underlying data, but the band drawn here is
        # clipped so it doesn't show an impossible ASR.
        lower_err = [m - max(0.0, m - c) for m, c in zip(means, cis)]
        upper_err = [min(1.0, m + c) - m for m, c in zip(means, cis)]

        ax.errorbar(
            x, means, yerr=[lower_err, upper_err],
            label=option, color=COLORS[option],
            linewidth=2, marker="o", markersize=5,
            capsize=3, capthick=1.2, elinewidth=1.2,
        )

    ax.set_xlabel("Alignment-training step", color="#0b0b0b")
    ax.set_ylabel("ASR", color="#0b0b0b")
    ax.set_title("Attack success rate over alignment training", color="#0b0b0b")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
