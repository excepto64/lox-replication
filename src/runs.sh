#!/bin/bash
# submit.sh

# Run this from within the vm from within the lox-replication repo

# SCRATCH=~/diss/lox-replication
SCRATCH=~/lox-replication

runs=("configs/lox_Llama-3_2-1B_r0_1e_dpo_adam.cfg")
seeds=(2 0 26)
# seeds=(2)
cluster=0
stage="A"

./src/install.sh ${SCRATCH} ${cluster}

if [ "${stage}" = "A" ]; then
    for run in "${runs[@]}"; do
        source ${run}
        for seed in "${seeds[@]}"; do
        ./src/run_stage_A.sh ${SCRATCH} ${seed} ${cluster} ${run}
        done
    done
elif [ "${stage}" = "B" ]; then
    for run in "${runs[@]}"; do
        source ${run}
        for seed in "${seeds[@]}"; do
        ./src/run_stage_B.sh ${SCRATCH} ${seed} ${cluster} ${run}
        done
    done
fi