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
    sbatch ./align_dpo.sh $model_name 0
else 
    ./align_dpo.sh $model_name 1;
fi

python LoX.py --base-model "HuggingFaceTB/SmolLM2-360M" --model excepto64/lox_SmolLM2-360M_hhrlhf