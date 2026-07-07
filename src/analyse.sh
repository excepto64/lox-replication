#!/bin/bash

# analyse.sh

datasets=("hh-rlhf" "alpaca" "wikipedia" "overrefusal" "overrefusal-toxic")

source .venv/bin/activate

for dataset in "${datasets[@]}"; do
    python src/kl_compare.py \
    --base-model HuggingFaceTB/SmolLM2-360M \
    --model excepto64/lox_SmolLM2-360M_hhrlhf_r0_1e_extracted_k6 \
    --dataset ${dataset} \
    --limit 20 \
    --batch-size 10
done