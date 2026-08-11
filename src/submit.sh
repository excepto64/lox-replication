#!/bin/bash
# submit.sh

# Run this from the cluster head, where you have the repository cloned.

SCRATCH=/disk/scratch/s2028118

runs=( \
    "configs/lox_Llama-3_2-1B_r0_1e_dpo_adam.cfg" \
    "configs/lox_Llama-3_2-1B_r0_1e_dpo_sgd.cfg" \
    "configs/lox_Llama-3_2-1B_r0_1e_sft_adam.cfg" \
    "configs/lox_Llama-3_2-1B_r0_1e_sft_sgd.cfg" \
)

seeds=(2 0 26)
cluster=1
stage="B"

echo Script started.

install_id=$(sbatch \
    --partition Teaching \
    --nodelist=landonia11 \
    --gres=gpu:1 \
    --time=1:00:00 \
    --cpus-per-task=1 \
    --job-name=Install \
    ./src/install.sh ${SCRATCH} ${cluster} | awk '{print $NF}')
echo "Install job submitted: ${install_id}."

if [ "${stage}" = "A" ]; then
    for run in "${runs[@]}"; do
        source ${run}
        for seed in "${seeds[@]}"; do
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
    for run in "${runs[@]}"; do
        source ${run}
        for seed in "${seeds[@]}"; do
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