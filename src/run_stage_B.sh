#!/bin/bash
# run_stage_B.sh

SCRATCH=${1}
seed=${2}
cluster=${3}
config=${4}

echo $(date)

# Attack model
# echo "Initiate attack."
# src/attack.sh ${SCRATCH} ${seed} ${cluster} ${config}
# echo "Attack completed."

# Measure safety of the model post-attack
echo "Initiate safety measurement."
src/measure_safety.sh ${SCRATCH} ${seed} ${cluster} ${config} 1
echo "Safety measurement completed."

