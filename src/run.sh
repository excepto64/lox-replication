#!/bin/bash

#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=Teaching
#SBATCH --time=12:00:00
#SBATCH --job-name=lox-align

install=1

model_name="HuggingFaceTB/SmolLM2-360M"
fine_tune_name="excepto64/lox_SmolLM2-360M_hhrlhf"
# model_name="meta-llama/Llama-3.2-1B"
# fine_tune_name="excepto64/lox_Llama-3.2-1B_hhrlhf"

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
./src/run_analysis.sh $model_name $fine_tune_name
echo "Model analysis complete."
