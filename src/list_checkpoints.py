"""List the HF Hub checkpoint revisions for a model repo, in training order.

align.sh uploads each checkpoint to its own revision named "step-<N>".
Queried from the Hub rather than hardcoded so the caller always gets
exactly the checkpoints that exist for that repo.

Usage: python src/list_checkpoints.py <repo_id>
"""

import argparse
import re

from huggingface_hub import list_repo_refs

parser = argparse.ArgumentParser()
parser.add_argument("repo_id", type=str)
args = parser.parse_args()


def main():
    refs = list_repo_refs(args.repo_id)
    steps = sorted(
        (int(m.group(1)), b.name)
        for b in refs.branches
        if (m := re.fullmatch(r"step-(\d+)", b.name))
    )
    for _, name in steps:
        print(name)


if __name__ == "__main__":
    main()
