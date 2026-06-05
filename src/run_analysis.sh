#!/bin/bash

model_name=$1
fine_tune_name=$2

source .venv/bin/activate

python LoX.py --base-model $model_name --model $fine_tune_name
python graph.py