#!/bin/bash
# run.sh

SCRATCH=${1}
config=${2}

echo $(date)

source ${config}

# Fine-tune model
echo "Initiate model fine-tuning."
./src/align_dpo.sh ${SCRATCH} ${model_name} ${fine_tune_name} ${lora} ${num_epochs}
echo "Model fine-tune complete."

# Run analysis
echo "Initiate model analysis."
./src/run_analysis.sh ${SCRATCH} ${model_name} ${fine_tune_name} ${main_dim} ${sec_dim}
echo "Model analysis complete."