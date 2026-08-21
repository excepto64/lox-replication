#!/usr/bin/env python3
"""Average ASR (attack success rate) over seeds for each 1B training option.

Scans inspect-ai .eval logs (advbench task) under inspect-logs/ and
results/logs/, parses the model name to recover model size, tuning method
(sft/dpo), optimizer (adam/sgd), and seed, then reports the mean ASR (with a
95% confidence interval across seeds) for each of the four 1B options:
dpo-adam, dpo-sgd, sft-adam, sft-sgd, pivoted with one column per revision.

Usage:
    python src/average_asr.py [LOG_DIR ...]
"""
import argparse
import csv
import glob
import math
import re
import statistics
import sys
from collections import defaultdict

from inspect_ai.log import read_eval_log

MODEL_RE = re.compile(
    r"lox_Llama-3_2-(?P<size>\d+B)_.*?_(?P<method>sft|dpo)_(?P<optim>adam|sgd)_s(?P<seed>\d+)"
)

# Two-tailed 95% critical values of the Student's t distribution, keyed by
# degrees of freedom (n_seeds - 1). With few seeds, the sample std is itself an
# uncertain estimate of the population std, so the t distribution's fatter
# tails widen the CI accordingly; it converges to the z=1.96 normal
# approximation as n grows. Falls back to 1.96 for df not in the table.
T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042,
}


def iter_asr_records(log_dirs):
    """Yield one dict per advbench eval log found under log_dirs, with model size/method/optim/seed parsed from the model name and its ASR score."""
    for log_dir in log_dirs:
        for path in glob.glob(f"{log_dir}/*.eval"):
            log = read_eval_log(path)
            if log.eval.task != "advbench":
                continue

            m = MODEL_RE.search(log.eval.model)
            if not m:
                continue

            revision = (log.eval.model_args or {}).get("revision")
            if revision is None:
                for tag in log.eval.tags or []:
                    if tag.startswith("revision:"):
                        revision = tag.split(":", 1)[1]
                        break
            if revision is None:
                revision = (log.eval.metadata or {}).get("revision")

            for score in log.results.scores if log.results else []:
                if "asr" not in score.metrics:
                    continue
                yield {
                    "size": m["size"],
                    "method": m["method"],
                    "optim": m["optim"],
                    "seed": m["seed"],
                    "revision": revision,
                    "asr": score.metrics["asr"].value,
                    "path": path,
                }


def summarize(vals):
    """Return (mean, 95%-CI half-width, n) for a list of ASR values across seeds."""
    n = len(vals)
    mean = sum(vals) / n
    if n > 1:
        std = statistics.stdev(vals)
        t_crit = T_CRIT_95.get(n - 1, 1.96)
        ci = t_crit * std / math.sqrt(n)
    else:
        ci = float("nan")
    return mean, ci, n


def format_cell(vals):
    """Format a list of ASR values as "mean±ci95 (n=...)" for the text table."""
    mean, ci, n = summarize(vals)
    if n > 1:
        return f"{mean:.4f}±{ci:.4f} (n={n})"
    return f"{mean:.4f} (n={n})"


def revision_step(rev):
    """Numeric step for a revision (0 for the post-attack/None case), for plotting on a shared x-axis."""
    m = re.search(r"\d+", rev or "")
    return int(m.group()) if m else 0


def revision_key(rev):
    """Sort key for revisions: numeric step ascending, with the post-attack (no-step) revision sorted last."""
    m = re.search(r"\d+", rev or "")
    return int(m.group()) if m else float("inf")


def print_table(rows, headers):
    """Print rows/headers as a plain aligned text table."""
    widths = [
        max(len(str(h)), *(len(str(row[i])) for row in rows)) if rows else len(str(h))
        for i, h in enumerate(headers)
    ]
    def fmt_row(cells):
        return " | ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def main():
    """Scan log_dirs for advbench eval logs, average ASR across seeds per (option, revision), and print a table or emit tidy CSV (--csv)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_dirs", nargs="*", default=["inspect-logs", "results/logs"]
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help=(
            "Emit tidy long-format CSV (one row per option x revision, with "
            "separate numeric mean/std/ci95/n columns) instead of an aligned "
            "text table. Intended for downstream loading with pandas/plotting, "
            "not for eyeballing."
        ),
    )
    args = parser.parse_args()

    records = list(iter_asr_records(args.log_dirs))
    if not records:
        print("No ASR records found.", file=sys.stderr)
        sys.exit(1)

    options = ["dpo_adam", "dpo_sgd", "sft_adam", "sft_sgd"]

    # per (option, revision) -> list of asr values (across seeds)
    by_option_revision = defaultdict(list)
    revisions = set()
    for r in records:
        if r["size"] != "1B":
            continue
        option = f"{r['method']}_{r['optim']}"
        by_option_revision[(option, r["revision"])].append(r["asr"])
        revisions.add(r["revision"])

    # Step-N revisions are checkpoints from the original alignment run;
    # revision=None is a differently-named "<...>_attack_alpaca" model
    # evaluated *after* a fine-tuning attack (see measure_safety.sh), so it
    # is flagged separately (is_attack) rather than treated as just another step.
    training_revisions = sorted((r for r in revisions if r is not None), key=revision_key)

    if args.csv:
        writer = csv.writer(sys.stdout)
        writer.writerow(["option", "revision", "step", "is_attack", "mean_asr", "std_asr", "ci95_asr", "n"])
        ordered_revisions = training_revisions + ([None] if None in revisions else [])
        for option in options:
            for rev in ordered_revisions:
                vals = by_option_revision.get((option, rev))
                if not vals:
                    continue
                mean, ci, n = summarize(vals)
                std = statistics.stdev(vals) if n > 1 else float("nan")
                writer.writerow([
                    option, rev or "", revision_step(rev), rev is None,
                    f"{mean:.6f}", f"{std:.6f}", f"{ci:.6f}", n,
                ])
        return

    if training_revisions:
        headers = ["revision"] + options
        rows = []
        for rev in training_revisions:
            row = [rev]
            for option in options:
                vals = by_option_revision.get((option, rev))
                row.append(format_cell(vals) if vals else "no data")
            rows.append(row)
        print("Alignment-training checkpoints:")
        print_table(rows, headers)

    if None in revisions:
        print()
        print("Post-attack (revision=None, i.e. <fine_tune_name>_attack_alpaca):")
        headers = ["option", "ASR"]
        rows = [
            [option, format_cell(by_option_revision[(option, None)])]
            for option in options
            if (option, None) in by_option_revision
        ]
        print_table(rows, headers)


if __name__ == "__main__":
    main()
