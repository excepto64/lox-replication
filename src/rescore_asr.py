"""Recompute ASR/mean-score metrics from an existing advbench .eval log, without
re-running the solver or the judge. Useful after changing a metric function
(e.g. accuracy() -> asr()) so old logs don't need to be re-evaluated.

Usage:
    python src/rescore_asr.py logs/2026-07-27T15-39-16-00-00_advbench_....eval
"""

import argparse

from inspect_ai.log import EvalMetric, read_eval_log, write_eval_log
from inspect_ai.scorer._metric import SampleScore

from ASR import asr, mean_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=str)
    parser.add_argument("--scorer-name", type=str, default="advbench_judge")
    parser.add_argument("--out", type=str, default=None, help="write updated log here (defaults to overwriting log_path)")
    args = parser.parse_args()

    log = read_eval_log(args.log_path)
    if log.samples is None:
        raise SystemExit("log has no samples (was it run with --no-log-samples?)")

    sample_scores = [
        SampleScore(
            score=sample.scores[args.scorer_name],
            sample_id=sample.id,
            sample_metadata=sample.metadata,
        )
        for sample in log.samples
        if sample.scores and args.scorer_name in sample.scores
    ]

    asr_value = asr()(sample_scores)
    mean_value = mean_score()(sample_scores)

    print(f"ASR: {asr_value}")
    print(f"Score: {mean_value}")

    if log.results is not None:
        for score in log.results.scores:
            if score.scorer == args.scorer_name:
                score.metrics.pop("accuracy", None)
                score.metrics["asr"] = EvalMetric(name="asr", value=asr_value)
                score.metrics["mean_score"] = EvalMetric(name="mean_score", value=mean_value)

        write_eval_log(log, args.out or args.log_path)
        print(f"Updated log written to {args.out or args.log_path}")


if __name__ == "__main__":
    main()
