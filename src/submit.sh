#!/bin/bash
# submit.sh

SCRATCH=/disk/scratch/s2028118

runs=("SmolLM2-360M_r6_1e.cfg" "SmolLM2-360M_r0_1e.cfg" "Llama-3_2-1B_r6_1e.cfg")

install_id=$(sbatch \
        --partition Teaching \
        --gres=gpu:nvidia_rtx_a6000:1 \
        --time=1:00:00 \
        --job-name=Install \
        ./src/install.sh ${SCRATCH} | awk '{print $NF}')
echo "Install job submitted: ${install_id}."

for run in "${runs[@]}"; do
    source ${run}
    sbatch \
        --partition Teaching \
        --gres=gpu:nvidia_rtx_a6000:1 \
        --cpus-per-task=1 \
        --time=12:00:00 \
        --job-name=${job_name} \
        --dependency=afterok:${install_id} \
        ./src/run.sh ${SCRATCH} ${run} 
done