#!/usr/bin/env python3
"""Average ASR (attack success rate) over seeds for each 1B training option.

Scans inspect-ai .eval logs (advbench task) under inspect-logs/ and
results/logs/, parses the model name to recover model size, tuning method
(sft/dpo), optimizer (adam/sgd), and seed, then reports the mean ASR for
each of the four 1B options: dpo-adam, dpo-sgd, sft-adam, sft-sgd.

Usage:
    python src/average_asr.py [LOG_DIR ...]
"""
import argparse
import glob
import re
import sys
from collections import defaultdict

from inspect_ai.log import read_eval_log

MODEL_RE = re.compile(
    r"lox_Llama-3_2-(?P<size>\d+B)_.*?_(?P<method>sft|dpo)_(?P<optim>adam|sgd)_s(?P<seed>\d+)"
)


def iter_asr_records(log_dirs):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_dirs", nargs="*", default=["inspect-logs", "results/logs"]
    )
    args = parser.parse_args()

    records = list(iter_asr_records(args.log_dirs))
    if not records:
        print("No ASR records found.", file=sys.stderr)
        sys.exit(1)

    options = ["dpo_adam", "dpo_sgd", "sft_adam", "sft_sgd"]

    def revision_key(rev):
        m = re.search(r"\d+", rev or "")
        return int(m.group()) if m else float("inf")

    # per (option, revision) -> list of asr values (across seeds)
    by_option_revision = defaultdict(list)
    for r in records:
        if r["size"] != "1B":
            continue
        option = f"{r['method']}_{r['optim']}"
        by_option_revision[(option, r["revision"])].append(r["asr"])

    print(f"{'option':10s} {'revision':10s} {'n_seeds':8s} {'mean_asr':10s}")
    for option in options:
        rev_keys = sorted(
            (k for k in by_option_revision if k[0] == option),
            key=lambda k: revision_key(k[1]),
        )
        if not rev_keys:
            print(f"{option:10s} {'-':10s} {'-':>8s} {'no data':>10s}")
            continue

        for key in rev_keys:
            vals = by_option_revision[key]
            mean = sum(vals) / len(vals)
            print(f"{option:10s} {str(key[1]):10s} {len(vals):8d} {mean:10.4f}")


if __name__ == "__main__":
    main()
