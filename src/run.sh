#!/bin/bash
# run.sh

SCRATCH=${1}
seed=${2}
config=${3}

echo $(date)

source ${config}

# Fine-tune model
echo "Initiate model fine-tuning."
./src/align.sh ${SCRATCH} ${seed} ${config}
echo "Model fine-tune complete."

# Run analysis
echo "Initiate model analysis."
./src/run_analysis.sh ${SCRATCH} ${seed} ${config}
echo "Model analysis complete."