#!/bin/bash
# submit.sh

# Run this from within the vm from within the lox-replication repo

# SCRATCH=~/diss/lox-replication
SCRATCH=~/lox-replication

runs=( \
    "configs/lox_Llama-3_2-1B_r0_1e_dpo_adam.cfg" \
    "configs/lox_Llama-3_2-1B_r0_1e_dpo_sgd.cfg" \
    "configs/lox_Llama-3_2-1B_r0_1e_sft_adam.cfg" \
    "configs/lox_Llama-3_2-1B_r0_1e_sft_sgd.cfg" \
)
seeds=(2 0 26)
# seeds=(2)
cluster=0
stage="A"

# ./src/install.sh ${SCRATCH} ${cluster}

if [ "${stage}" = "A" ]; then
    for run in "${runs[@]}"; do
        source ${run}
        for seed in "${seeds[@]}"; do
        ./src/run_stage_A.sh ${SCRATCH} ${seed} ${cluster} ${run}
        done
        source .venv/bin/activate
        python src/average_seeds.py --model ${fine_tune_name} --seeds ${seeds} --shapes ${shapes}
    done
elif [ "${stage}" = "B" ]; then
    for run in "${runs[@]}"; do
        source ${run}
        for seed in "${seeds[@]}"; do
        ./src/run_stage_B.sh ${SCRATCH} ${seed} ${cluster} ${run}
        done
    done
fi