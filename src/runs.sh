#!/bin/bash
# submit.sh

# Main script for running experiment from a virtual machine.
# Run this from within the vm from within the lox-replication dir.

# Do not change.
SCRATCH=~/lox-replication

# Name the configuration files you want to run the experiment on.
runs=( \
    "configs/lox_Llama-3_2-1B_r0_1e_dpo_adam.cfg" \
    "configs/lox_Llama-3_2-1B_r0_1e_dpo_sgd.cfg" \
    "configs/lox_Llama-3_2-1B_r0_1e_sft_adam.cfg" \
    "configs/lox_Llama-3_2-1B_r0_1e_sft_sgd.cfg" \
)

seeds=(2 0 26) # Random seeds
cluster=0 # Execution mode. Do not touch! Use submit.sh instead.
# Set stage 'A' for aligning, update and safety measurement.
# Set stage 'B' for attack and safety measurement.
stage="A" 

# Install packages.
./src/install.sh ${SCRATCH} ${cluster}

# Run approriate stage for all selected runs.
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