#!/bin/bash
# measure_update.sh
#
# For every saved checkpoint of one aligned run, computes and plots the
# low-rankedness of the safety update in weight space (LoX.py) and
# activation space (extract_activations.py), i.e. the SVD_coeffs_*.pt files
# consumed by calc_gini_asr.py. Phase 1 computes the SVDs per checkpoint;
# phase 2 finds/fixes a shared y-axis max across checkpoints (unless
# svd_ylim_max / svd_ylim_max_dwx are set in `config`) so plots are
# comparable; phase 3 plots each checkpoint with that fixed scale and
# assembles the per-step plots into GIFs (make_gifs.py).
# Called by run_stage_A.sh, not run directly.
# Args: SCRATCH seed cluster config
# `config` additionally supplies num_samples, batch_size, save_steps, shapes.

set -e

SCRATCH=${1}
seed=${2}
cluster=${3}
config=${4}

source ${config}

fine_tune_name=${fine_tune_name}_s${seed}
local_name=${fine_tune_name##*/}
local_dir=${SCRATCH}/${local_name}

# Activate env and set final data upload.
source ${SCRATCH}/.venv/bin/activate
if [ ${cluster} -eq 1 ]; then
    set -a; source ~/lox-replication/.env; set +a
    source /home/htang2/toolchain-20251006/toolchain.rc
    cd ${local_dir}
    trap 'rsync -a ${local_dir}/ ~/lox-replication/${local_name} || [ $? -eq 24 ]' EXIT
elif [ ${cluster} -eq 0 ]; then
    set -a; source ${SCRATCH}/.env; set +a
fi

# Iterate over checkpoints.
num_checkpoints=$((num_samples / (batch_size * save_steps)))
steps=()
for i in $(seq 1 ${num_checkpoints}); do
    steps+=($((i * save_steps)))
done

# Phase 1: compute SVD_coeffs_*.pt for every step (weights + activations).
for step in "${steps[@]}"; do
    revision="step-${step}"
    python src/LoX.py --base-model ${model_name} --model ${fine_tune_name} --lora ${lora} --revision ${revision}
    python src/extract_activations.py --base-model ${model_name} --model ${fine_tune_name} --seed ${seed} --revision ${revision}
done

# Phase 2: fix the average-singular-value plot's y-axis across all steps.
# Config vars svd_ylim_max / svd_ylim_max_dwx override the auto-computed max if set.
if [ -z "${svd_ylim_max}" ]; then
    svd_ylim_max=$(python src/find_svd_ylim.py --fine-tune-name "${local_name}" --steps "${steps[@]}" --shapes ${shapes})
fi
if [ -z "${svd_ylim_max_dwx}" ]; then
    svd_ylim_max_dwx=$(python src/find_svd_ylim.py --fine-tune-name "${local_name}" --steps "${steps[@]}" --suffix dWX --shapes ${shapes})
fi

svd_ylim_args=(--svd-ylim-max "${svd_ylim_max}")
svd_ylim_args_dwx=(--svd-ylim-max "${svd_ylim_max_dwx}")

# Phase 3: plot every step with the fixed scale.
for step in "${steps[@]}"; do
    revision="step-${step}"
    python src/graph.py --base-model ${model_name} --model ${fine_tune_name} --shapes ${shapes} --revision ${revision} "${svd_ylim_args[@]}"
    python src/graph.py --base-model ${model_name} --model ${fine_tune_name} --shapes ${shapes} --suffix dWX --revision ${revision} "${svd_ylim_args_dwx[@]}"
done

python src/make_gifs.py --results-dir "$(pwd)" --run "${local_name}"
