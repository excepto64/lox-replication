#!/bin/bash

config=${1}

source ${config}

install=1

# Clear scratch space.
rm -rf /disk/scratch/s2028118

# Install dependencies
if [ ${install} -eq 1 ]; then
    ./src/install.sh
    echo "Dependencies installed!"
fi

# Fine-tune model
echo "Initiate model fine-tuning."
./src/align_dpo.sh ${model_name} ${fine_tune_name} ${lora}
echo "Model fine-tune complete."



# Run analysis
echo "Initiate model analysis."
./src/run_analysis.sh ${model_name} ${fine_tune_name} ${main_dim} ${sec_dim}
echo "Model analysis complete."
