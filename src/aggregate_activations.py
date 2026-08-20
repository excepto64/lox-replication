"""
Aggregate activation-space (dWX) SVD spectra across seeds.

Mirrors the results-param/ flat-directory convention used for weight-space
aggregation, but for the already-computed SVD_coeffs_*_dWX.pt files (per
config/seed/step, living under results/<mode>/<name>_s<seed>/): symlinks
them into a flat results-act/<mode>/ directory, then reuses graph.py (grouped
by len(S), i.e. d_out -- 512/2048/8192 for Llama-3.2-1B) per seed/step and
average_seeds.py across seeds unmodified, via subprocess. No SVD_coeffs are
recomputed -- those are only read, never written.
"""

import configparser
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODES = ["dpo_adam", "dpo_sgd", "sft_adam", "sft_sgd"]
SEEDS = [2, 0, 26]
DIMS = ["512", "2048", "8192"]


def read_cfg(mode):
    """configs/*.cfg are `key=value` (with quoted strings) bash snippets; a
    bare ConfigParser with a dummy section header handles them fine."""
    text = "[cfg]\n" + (REPO_ROOT / "configs" / f"lox_Llama-3_2-1B_r0_1e_{mode}.cfg").read_text()
    parser = configparser.ConfigParser()
    parser.read_string(text)
    section = parser["cfg"]
    return {k: shlex.split(v)[0] if v.strip() else v for k, v in section.items()}


def steps_for(cfg):
    num_checkpoints = int(cfg["num_samples"]) // (int(cfg["batch_size"]) * int(cfg["save_steps"]))
    return [i * int(cfg["save_steps"]) for i in range(1, num_checkpoints + 1)]


def link_activation_files(mode, local_name_base, steps, out_dir):
    for seed in SEEDS:
        local_name = f"{local_name_base}_s{seed}"
        src_dir = REPO_ROOT / "results" / mode / local_name
        for step in steps:
            fname = f"SVD_coeffs_{local_name}_step-{step}_dWX.pt"
            dest = out_dir / fname
            dest.unlink(missing_ok=True)
            dest.symlink_to(src_dir / fname)


def main():
    root_out = REPO_ROOT / "results-act"
    root_out.mkdir(parents=True, exist_ok=True)
    graph_csv = root_out / "graph_out.csv"
    seed_ave_csv = root_out / "graph_out_seed_ave.csv"

    for mode in MODES:
        cfg = read_cfg(mode)
        steps = steps_for(cfg)
        local_name_base = f"lox_Llama-3_2-1B_r0_1e_{mode}"
        out_dir = root_out / mode
        out_dir.mkdir(parents=True, exist_ok=True)

        link_activation_files(mode, local_name_base, steps, out_dir)

        for seed in SEEDS:
            local_name = f"{local_name_base}_s{seed}"
            for step in steps:
                subprocess.run(
                    [
                        sys.executable, str(REPO_ROOT / "src" / "graph.py"),
                        "--model", f"excepto64/{local_name}",
                        "--shapes", *DIMS,
                        "--suffix", "dWX",
                        "--revision", f"step-{step}",
                        "--out", str(graph_csv),
                    ],
                    cwd=out_dir, check=True,
                )

        subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "src" / "average_seeds.py"),
                "--model", f"excepto64/{local_name_base}",
                "--seeds", *(str(s) for s in SEEDS),
                "--shapes", *DIMS,
                "--suffix", "dWX",
                "--steps", *(str(s) for s in steps),
                "--graph-csv", str(graph_csv),
                "--out", str(seed_ave_csv),
            ],
            cwd=out_dir, check=True,
        )


if __name__ == "__main__":
    main()
