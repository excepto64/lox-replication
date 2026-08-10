#!/usr/bin/env python3
"""Surface an inspect eval log's model_args.revision as a tag, so the log
viewer shows/filters on it while the model name stays clean for sorting.

Usage:
    python scripts/tag_revision_into_model.py LOG.eval [LOG2.eval ...] [--out-dir DIR]

Writes a copy of each log with the tag added; originals are left untouched.
"""
import argparse
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log


def tag_revision(path: Path):
    log = read_eval_log(str(path))

    model_args = log.eval.model_args or {}
    revision = model_args.get("revision")

    if revision is None:
        print(f"[skip] {path.name}: no model_args.revision found")
        return None

    tag = f"revision:{revision}"
    tags = list(log.eval.tags or [])
    if tag in tags:
        print(f"[skip] {path.name}: already tagged")
        return None

    tags.append(tag)
    log.eval.tags = tags

    metadata = dict(log.eval.metadata or {})
    metadata["revision"] = revision
    log.eval.metadata = metadata

    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None, help="directory to write tagged copies into")
    args = parser.parse_args()

    for log_path in args.logs:
        log = tag_revision(log_path)
        if log is None:
            continue

        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.out_dir / log_path.name
        else:
            out_path = log_path.with_stem(log_path.stem + "_tagged")

        write_eval_log(log, str(out_path))
        print(f"[ok] {log_path.name} -> {out_path.name}  (tags: {log.eval.tags})")


if __name__ == "__main__":
    main()
