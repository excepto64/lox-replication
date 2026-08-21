# Are low-rank safety updates in LLMs less robust?

Code and data for Adam Harrison's MSc dissertation (School of Informatics, University of Edinburgh, 2026), supervised by Ivan and Neil.

## Description

Open-weight LLMs can be fine-tuned by anyone who downloads them, which makes it easy to accidentally (or deliberately) strip away their safety training — even benign fine-tuning on an unrelated, non-harmful dataset can degrade a model's refusal behaviour. Prior work has observed that the safety update introduced by alignment fine-tuning is *low-rank*, both in weight space and in activation space, and has hypothesised that this low-rankedness is itself a cause of the brittleness: a safety mechanism concentrated in a few directions is easier to disrupt than one spread across many.

This project tests that hypothesis directly. We safety-tune a pretrained model using four training configurations — Supervised Fine-Tuning (SFT) vs. Direct Preference Optimisation (DPO), each with either the Adam or SGD optimiser — using a filtered version of the PKU-SafeRLHF dataset that is constructed to make the four settings directly comparable. For each resulting aligned model we measure:

1. **The low-rankedness of the safety update**, via the Gini coefficient of the singular value spectrum of the weight-space update (`ΔW = W_aligned − W_base`) and of activation-space projections, following the SVD-based approach used in prior low-rank safety update work.
2. **Safety before and after a benign fine-tuning attack**, measured as Attack Success Rate (ASR) on AdvBench-style harmful prompts, before and after further fine-tuning the aligned model on a benign utility dataset (Stanford Alpaca).

We then relate the low-rankedness of each configuration's safety update to how much its safety degrades under the benign fine-tuning attack, to test whether low-rank updates really are more brittle, and separately investigate what training-time factors (method, optimiser) cause a safety update to be more or less low-rank.

**Findings:** Adam produces more low-rank safety updates than SGD, as hypothesised. For DPO runs specifically, lower-rank safety updates are associated with more brittle safety mechanisms that degrade further after benign fine-tuning.

Model checkpoints and large intermediate outputs (`results/`, `results-param/`, `results-act/`, `results-prelim/`, `inspect-logs/`, `logs/`) are not included in this archive, since they're fully regenerable from the code and configs here (`figs/` has the final summary CSVs/plots — see below). Wherever these directories are mentioned below, that's documenting where a script writes to or reads from when you run it, not something already present in the archive.

## How to run

The entire experiment (aligning, measuring the safety update, attacking, and re-measuring safety) is driven by two entry-point scripts in `src/`, which both call `src/install.sh`, `src/run_stage_A.sh`, and `src/run_stage_B.sh` under the hood:

- **`src/runs.sh`** — runs everything locally/on a single VM (sequentially, `cluster=0`). Run from the repo root.
- **`src/submit.sh`** — submits the same runs as SLURM jobs on the Informatics `Teaching` partition (`cluster=1`). Run from the cluster head; edit the `SCRATCH` path to your own scratch directory first.

In both scripts, edit the top of the file to select:
- `runs` — which config file(s) from `configs/` to run (one per model/rank/method/optimiser combination, e.g. `configs/lox_Llama-3_2-1B_r0_1e_dpo_adam.cfg`).
- `seeds` — which random seeds to use (paper used `2 0 26`).
- `stage` — `"A"` to align the model and measure the safety update/pre-attack safety, or `"B"` to run the benign fine-tuning attack and measure post-attack safety. Stage B requires stage A to have been run first for the same config/seed.

Each config file under `configs/` sets the model, rank, method (SFT/DPO), and optimiser (Adam/SGD) for one training configuration; add a new `.cfg` file there to test another combination.

#### Config fields (`configs/*.cfg`)

Each file is a bash snippet (`source`d by `align.sh`/`measure_update.sh`/`attack.sh`/`measure_safety.sh`) setting:

- `model_name` — base pretrained model on the Hub (e.g. `meta-llama/Llama-3.2-1B`).
- `fine_tune_name` — Hub repo id the aligned model is pushed to (`_s<seed>` is appended per seed); `_attack_alpaca` is further appended for the Stage B attacked model.
- `shapes` — space-separated weight-matrix shapes to track (e.g. `"2048,2048 512,2048 2048,8192"`), passed to `graph.py`/`find_svd_ylim.py` as `--shapes`.
- `job_name` — SLURM job name prefix used by `submit.sh` (cluster runs only; cosmetic).
- `lora` — LoRA rank; `0` means full-parameter fine-tuning (this project's setting throughout — LoRA support exists in the scripts but wasn't used for the reported results).
- `num_epochs`, `batch_size`, `num_samples` — training length (`num_samples` is capped at 24,000 across all runs; see the dissertation for why).
- `save_steps` — how many samples between checkpoints (each "step" here is `save_steps` × 100 samples, per `align.sh`'s fixed batch size of 100 — see `measure_update.sh`'s step-number convention).
- `method` — `"dpo"` or `"sft"`.
- `optimiser` — `"adam"` or `"sgd"`.
- `revision` — which alignment checkpoint (e.g. `"step-210"`) Stage B's benign fine-tuning attack (`attack.sh`) is applied to, and which checkpoint `measure_safety.sh`/`ASR.py` tag their pre-attack measurement with for the post-attack comparison. This is *not* necessarily the final checkpoint (`step-240`) — each config's `revision` was chosen individually, so check this field rather than assuming the last step was used.

Note `measure_update.sh` (Stage A's per-checkpoint sweep) does not use this `revision` field — it loops over every saved checkpoint itself, computing `revision="step-${step}"` for each one internally.

### Recreating the datasets (optional)

The filtered SFT/DPO datasets are already published on HuggingFace (`excepto64/PKU-SafeRLHF-filtered-sft`, `excepto64/PKU-SafeRLHF-filtered-dpo`) and are pulled automatically by the training scripts, so this step isn't needed to reproduce the experiment. To regenerate them from scratch (e.g. after changing the filtering logic), run `src/prep_data.py`, which filters `PKU-Alignment/PKU-SafeRLHF` down to rows where at least one response is safe, builds matching chosen/rejected (DPO) and input/output (SFT) datasets, and pushes both back to the Hub.

## Averaging across seeds and producing the final plots

Each run of `run_stage_A.sh`/`run_stage_B.sh` produces per-seed, per-checkpoint results (Gini/Lorenz plots via `graph.py`, ASR via `ASR.py`'s Inspect eval logs). Since `graph.py`/`LoX.py`/`extract_activations.py` all read/write relative to the current working directory and a plain local run (`cluster=0`) never `cd`s elsewhere, a fresh run leaves everything — every seed's `SVD_coeffs_*.pt`, `cum_*/lorenz_*/sum_*.pt`, `graph_out.csv`, and `.pdf` plots — flat in the repo root (filenames already encode model/seed/step, so nothing collides). A second layer of scripts gathers and aggregates these across the 3 seeds and produces the plots used in the dissertation:

- **`src/aggregate_svd_results.py`** — run this first. Sorts that flat mess by (mode, weight-space vs. activation-space) into `results-param/<mode>/` and `results-act/<mode>/`: for each of the four training options, symlinks every seed/step's matching `SVD_coeffs_*.pt` and already-computed `cum_*/lorenz_*/sum_*.pt` curve files (`graph.py` already produced these during `measure_update.sh`, so nothing is recomputed here) plus the repo root's global `graph_out.csv`, into the two flat per-mode directories `average_seeds.py` expects.
- **`src/average_seeds.py`** — averages one run's per-seed weight-space (or, with `--suffix dWX`, activation-space) Gini/Lorenz/average-singular-value curves across seeds, re-derives the Gini coefficient etc. from the averaged curve, cross-checks against the per-seed values, and re-plots. Run once per training option (`dpo_adam`, `dpo_sgd`, `sft_adam`, `sft_sgd`) from inside `results-param/<mode>/` (weight-space) or `results-act/<mode>/` (`--suffix dWX`, activation-space) — it reads/writes relative to the current working directory, so it expects each seed's curve files already sitting together, flat, in that directory (which is what `aggregate_svd_results.py` sets up).
- **`src/average_asr.py`** — scans Inspect `.eval` logs (default: `inspect-logs/`, `results/logs/`) for `advbench` task results, parses model size/method/optimiser/seed from the model name, and averages ASR across seeds per (option, checkpoint step). Use `--csv` to emit the tidy long-format CSV consumed by `plot_asr.py`.
- **`src/calc_gini_asr.py`** — joins a Gini summary (`gini_attack.csv`) with an ASR summary (`asr_attack.csv`), computes `del_ASR = after_attack − before_attack` per option, and plots Gini vs. `del_ASR` — the core plot testing the dissertation's central hypothesis. Expects both CSVs in the working directory.

#### Manual step: building `gini_attack.csv` and `asr_attack.csv`

These two summary CSVs were built by hand from the outputs above (no script produces them automatically):

- **`gini_attack.csv`** (columns `option, series, value, std, ci95`) is the `metric == "gini"` rows of `average_seeds.py`'s `graph_out_seed_ave.csv`, filtered to the selected alignment checkpoint (set in config as revision), with the `model` column (e.g. `excepto64/lox_Llama-3_2-1B_r0_1e_dpo_adam`) shortened to just the option (e.g. `dpo_adam`).
- **`asr_attack.csv`** (columns `option, before_mean, before_std, before_ci95, after_mean, after_std, after_ci95`) pairs, per option, the pre-attack row with the post-attack row (`is_attack=True`) from `results_asr.csv`, renaming `mean_asr/std_asr/ci95_asr` to `before_*`/`after_*` accordingly.

Final plotting scripts, all reading from the CSVs the scripts above produce:

- **`src/plot_gini.py`** — Gini coefficient vs. alignment-training step, one line per option, faceted by weight-matrix shape. Reads `average_seeds.py`'s seed-averaged CSV (default `graph_out_seed_ave.csv`).
- **`src/plot_asr.py`** — ASR vs. alignment-training step, one line per option, with 95% CI bands. Reads `average_asr.py --csv`'s tidy CSV (default `results_asr.csv`).
- **`src/plot_change_asr.py`** — grouped bar chart of ASR before vs. after the benign fine-tuning attack, one cluster per option. Reads a CSV with `option, before_mean, before_ci95, after_mean, after_ci95` columns (default `asr_attack.csv`).

`figs/` already contains the final CSVs and plots these scripts produced for the dissertation (`gini_attack.csv`, `asr_attack.csv`, `results_asr.csv`, and the corresponding `.pdf` plots), so they don't need to be regenerated to inspect the results.
