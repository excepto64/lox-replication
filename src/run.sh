#!/bin/bash

#SBATCH --gres=gpu:nvidia_rtx_a6000:1
# #SBATCH --gres=gpu:3g.71gb
#SBATCH --cpus-per-task=1
#SBATCH --partition=Teaching
#SBATCH --time=12:00:00
#SBATCH --job-name=lox-align

config=$1

source $config

install=1

# Clear scratch space.
rm -rf /disk/scratch/s2028118

# Install dependencies
if [ $install -eq 1 ]; then
    ./src/install.sh
    echo "Dependencies installed!"
fi

# Fine-tune model
echo "Initiate model fine-tuning."
./src/align_dpo.sh $model_name $fine_tune_name
echo "Model fine-tune complete."



# Run analysis
echo "Initiate model analysis."
./src/run_analysis.sh $model_name $fine_tune_name $main_dim $sec_dim
echo "Model analysis complete."
