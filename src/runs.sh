#!/bin/bash
# submit.sh

# Run this from within the vm from within the lox-replication repo

# SCRATCH=~/diss/lox-replication
SCRATCH=~/lox-replication

runs=("lox_SmolLM2-360M_hhrlhf_r0_1e_test_sft_sgd.cfg")
# seeds=(2 0 26)
seeds=(2)
cluster=0

./src/install.sh ${SCRATCH} ${cluster}

for run in "${runs[@]}"; do
    source ${run}
    for seed in "${seeds[@]}"; do
    ./src/run.sh ${SCRATCH} ${seed} ${cluster} ${run}
    done
done