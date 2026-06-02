#!/bin/bash

online=1
install=0

model_name="HuggingFaceTB/SmolLM2-360M"

# Install dependencies
if [$install -eq 1 && $online -eq 1]; then
    ./install.sh
fi

source .venv/bin/activate

# 
if [$online -eq 0]; then
    ./download.sh $model_name
    sbatch ./fine_tune.sh $model_name 0
else 
    ./fine_tune.sh $model_name 1;
fi