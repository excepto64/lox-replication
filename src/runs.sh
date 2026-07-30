#!/bin/bash
# submit.sh

# Run this from within the vm from within the lox-replication repo

# SCRATCH=~/diss/lox-replication
SCRATCH=~/lox-replication

runs=("lox_SmolLM2-360M_hhrlhf_r0_1e_test_dpo_adam.cfg")
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