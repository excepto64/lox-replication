#!/bin/bash

model_name=$1

source .env
hf auth login --token $HF_TOKEN

hf download $model_name --local-dir ./models
hf download Anthropic/hh-rlhf --repo-type dataset --local-dir ./dataset