"""
Gather the flat mess of per-seed, per-checkpoint files that run_stage_A.sh/
run_stage_B.sh leave in the repo root (SVD_coeffs_*.pt from LoX.py/
extract_activations.py, and the cum_*/lorenz_*/sum_*.pt curves + graph_out.csv
that measure_update.sh's graph.py calls already wrote there) into two flat,
per-mode directories -- results-param/<mode>/ (weight-space) and
results-act/<mode>/ (activation-space, --suffix dWX) -- the layout
average_seeds.py expects (it reads/writes relative to its own working
directory, so it needs one run's files, across all 3 seeds, sitting together
flat rather than scattered). Nothing is recomputed: graph.py has already been
run (by measure_update.sh) for every checkpoint, so this script only
symlinks existing files into place. Run this once, then run average_seeds.py
directly (optionally --suffix dWX) from inside each results-param/<mode>/ or
results-act/<mode>/ directory.
"""

import glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODES = ["dpo_adam", "dpo_sgd", "sft_adam", "sft_sgd"]
SEEDS = [2, 0, 26]
STEPS = [30, 60, 90, 120, 150, 180, 210, 240]
PREFIXES = ["SVD_coeffs", "cum", "lorenz", "sum"]


def link_matching_files(local_name_base, steps, out_dir, suffix):
    """Symlink every {prefix}_*{local_name}_step-N{tag}.pt file for this run (all seeds/steps/prefixes) from the repo root into out_dir."""
    tag = f"_{suffix}" if suffix else ""
    for seed in SEEDS:
        local_name = f"{local_name_base}_s{seed}"
        for step in steps:
            for prefix in PREFIXES:
                pattern = f"{prefix}_*{local_name}_step-{step}{tag}.pt"
                for src in REPO_ROOT.glob(pattern):
                    dest = out_dir / src.name
                    dest.unlink(missing_ok=True)
                    dest.symlink_to(src)


def link_graph_csv(out_dir):
    """Symlink the repo root's global graph_out.csv into out_dir, for average_seeds.py's optional cross-check."""
    src = REPO_ROOT / "graph_out.csv"
    if not src.exists():
        return
    dest = out_dir / "graph_out.csv"
    dest.unlink(missing_ok=True)
    dest.symlink_to(src)


def gather(root_out_name, suffix):
    """Gather every mode's matching SVD/curve files into flat root_out_name/<mode>/ directories."""
    root_out = REPO_ROOT / root_out_name
    for mode in MODES:
        local_name_base = f"lox_Llama-3_2-1B_r0_1e_{mode}"
        out_dir = root_out / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        link_matching_files(local_name_base, STEPS, out_dir, suffix)
        link_graph_csv(out_dir)


def main():
    """Gather weight-space files into results-param/<mode>/ and activation-space (dWX) files into results-act/<mode>/."""
    gather("results-param", suffix="")
    gather("results-act", suffix="dWX")


if __name__ == "__main__":
    main()
