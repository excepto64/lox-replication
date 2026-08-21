#!/bin/bash
# run_stage_A.sh
#
# Stage A: align a model on one config/seed, then measure its safety update
# (rank/Gini) and pre-attack safety. Called by runs.sh/submit.sh, not run
# directly. Args: SCRATCH seed cluster config

SCRATCH=${1}
seed=${2}
cluster=${3}
config=${4}

echo $(date)

# Align model
echo "Initiate model fine-tuning."
./src/align.sh ${SCRATCH} ${seed} ${cluster} ${config}
echo "Model fine-tune complete."

# Measure the safety update and calculate gini of the update.
echo "Initiate update measurement."
./src/measure_update.sh ${SCRATCH} ${seed} ${cluster} ${config}
echo "Update measurement complete."

# Measure the safety of the model pre-attack
echo "Initiate safety measurement."
./src/measure_safety.sh ${SCRATCH} ${seed} ${cluster} ${config} 0
echo "Safety measurement complete."