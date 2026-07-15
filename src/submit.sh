#!/bin/bash
# submit.sh

SCRATCH=/disk/scratch/s2028118

runs=("SmolLM2-360M_r0_1e_ckpt.cfg")
seeds=(2 0 26)

install_id=$(sbatch \
        --partition Teaching \
        --gres=gpu:nvidia_rtx_a6000:1 \
        --time=1:00:00 \
        --job-name=Install \
        ./src/install.sh ${SCRATCH} | awk '{print $NF}')
echo "Install job submitted: ${install_id}."

for run in "${runs[@]}"; do
    source ${run}
    for seed in "${seeds[@]}"; do
        sbatch \
                --partition Teaching \
                --gres=gpu:nvidia_rtx_a6000:1 \
                --cpus-per-task=1 \
                --job-name=${seed}_${job_name} \
                --dependency=afterok:${install_id} \
                ./src/run.sh ${SCRATCH} ${seed} ${run}
done