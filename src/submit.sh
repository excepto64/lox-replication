#!/bin/bash
# submit.sh

# Main script for running experiment from the icd cluster.
# Run this from the cluster head, where you have the repository cloned.

# Edit username. Do not change otherwise
SCRATCH=/disk/scratch/USERNAME

# Name the configuration files you want to run the experiment on.
runs=( \
    "configs/lox_Llama-3_2-1B_r0_1e_dpo_adam.cfg" \
    "configs/lox_Llama-3_2-1B_r0_1e_dpo_sgd.cfg" \
    "configs/lox_Llama-3_2-1B_r0_1e_sft_adam.cfg" \
    "configs/lox_Llama-3_2-1B_r0_1e_sft_sgd.cfg" \
)

seeds=(2 0 26) # Random seeds
cluster=1 # Execution mode. Do not touch! Use runs.sh instead.
# Set stage 'A' for aligning, update and safety measurement.
# Set stage 'B' for attack and safety measurement.
stage="A" 

echo Script started.

# Install packages.
install_id=$(sbatch \
    --partition Teaching \
    --nodelist=landonia11 \
    --gres=gpu:1 \
    --time=1:00:00 \
    --cpus-per-task=1 \
    --job-name=Install \
    ./src/install.sh ${SCRATCH} ${cluster} | awk '{print $NF}')
echo "Install job submitted: ${install_id}."

#Once installed, run approriate stage for all selected runs.
if [ "${stage}" = "A" ]; then
    for seed in "${seeds[@]}"; do
        for run in "${runs[@]}"; do
            source ${run}
            sbatch \
                --partition Teaching \
                --nodelist=landonia11 \
                --gres=gpu:1 \
                --cpus-per-task=1 \
                --job-name=${seed}_${job_name} \
                --dependency=afterok:${install_id} \
                ./src/run_stage_A.sh ${SCRATCH} ${seed} ${cluster} ${run}
        done
    done
elif [ "${stage}" = "B" ]; then
    for seed in "${seeds[@]}"; do
        for run in "${runs[@]}"; do
            source ${run}
            sbatch \
                --partition Teaching \
                --nodelist=landonia11 \
                --gres=gpu:1 \
                --cpus-per-task=1 \
                --job-name=${seed}_${job_name} \
                --dependency=afterok:${install_id} \
                ./src/run_stage_B.sh ${SCRATCH} ${seed} ${cluster} ${run}
        done
    done
fi