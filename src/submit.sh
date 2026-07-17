#!/bin/bash
# submit.sh

# Run this from the cluster head, where you have the repository cloned.

SCRATCH=/disk/scratch/s2028118

runs=("lox_SmolLM2-360M_hhrlhf_r0_1e_test_dpo_adam.cfg" "lox_SmolLM2-360M_hhrlhf_r0_1e_test_dpo_sgd.cfg" "lox_SmolLM2-360M_hhrlhf_r0_1e_test_sft_adam.cfg" 
"lox_SmolLM2-360M_hhrlhf_r0_1e_test_sft_sgd.cfg")
seeds=(2 0 26)
cluster=1

echo Script started.

install_id=$(sbatch \
        --partition Teaching \
        --nodelist=landonia03 \
        --gres=gpu:1
        --time=1:00:00 \
        --cpus-per-task=1 \
        --job-name=Install \
        ./src/install.sh ${SCRATCH} ${cluster} | awk '{print $NF}')
echo "Install job submitted: ${install_id}."

for run in "${runs[@]}"; do
    source ${run}
    for seed in "${seeds[@]}"; do
        sbatch \
                --partition Teaching \
                --nodelist=landonia03 \
                --gres=gpu:1 \
                --cpus-per-task=1 \
                --job-name=${seed}_${job_name} \
                --dependency=afterok:${install_id} \
                ./src/run.sh ${SCRATCH} ${seed} ${cluster} ${run}
        done
done