#!/bin/bash
# run.sh

SCRATCH=${1}
seed=${2}
cluster=${3}
config=${4}

echo $(date)

# Fine-tune model
echo "Initiate model fine-tuning."
./src/align.sh ${SCRATCH} ${seed} ${cluster} ${config}
echo "Model fine-tune complete."

# Run analysis
echo "Initiate model analysis."
./src/run_analysis.sh ${SCRATCH} ${seed} ${cluster} ${config}
echo "Model analysis complete."