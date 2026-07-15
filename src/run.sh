#!/bin/bash
# run.sh

SCRATCH=${1}
config=${2}

echo $(date)

source ${config}

# Fine-tune model
echo "Initiate model fine-tuning."
./src/align.sh ${SCRATCH} ${config}
echo "Model fine-tune complete."

# Run analysis
echo "Initiate model analysis."
./src/run_analysis.sh ${SCRATCH} ${config}
echo "Model analysis complete."