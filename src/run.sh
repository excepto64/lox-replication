#!/bin/bash

online=0
install=1

model_name="HuggingFaceTB/SmolLM2-360M"
fine_tune_name="excepto64/lox_SmolLM2-360M_hhrlhf"

# Install dependencies
if [ $install -eq 1 ]; then
    ./src/install.sh
fi

# Download and fine-tune model
if [ $online -eq 0 ]; then
    ./src/download.sh $model_name
    sbatch -p Teaching --gres=gpu:1 ./src/align_dpo.sh $model_name 0
else 
    ./src/align_dpo.sh $model_name 1;
fi

# Run analysis

if [ $online -eq 0 ]; then
    sbatch -p Teaching --gres=gpu:1 ./src/run_analysis.sh $model_name $fine_tune_name
else 
    ./src/run_analysis.sh $model_name $fine_tune_name
fi
