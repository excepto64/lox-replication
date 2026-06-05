#!/bin/bash

model_name=$1
online=$2
source .env

scratch="disk/scratch/s2028118/lox-replication"

if [ $online -eq 0 ]; then
    mkdir $scratch
    rsync ~/lox-replication $scratch
    cd $scratch
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
fi

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Modified from LoX paper.
deepspeed --module openrlhf.cli.train_dpo  \
    --model.model_name_or_path $model_name \
    --model.beta 0.1 \
    --model.gradient_checkpointing_enable \
    --data.dataset Anthropic/hh-rlhf \
    --data.chosen_key chosen \
    --data.rejected_key rejected \
    --data.max_len 1024 \
    --data.max_samples 8000 \
    --train.batch_size 128 \
    --train.micro_batch_size 1 \
    --train.max_epochs 1 \
    --train.seed 48 \
    --adam.lr 5e-6 \
    --ds.packing_samples \
    --ds.zero_stage 3 \
    --ds.param_dtype bf16 \
    --ds.attn_implementation flash_attention_2 \
    --ds.adam_offload \
    --ckpt.output_dir ./model \
    --ckpt.save_steps -1 \
    --eval.steps -1 \
    --logger.logging_steps 1 \
    --logger.wandb.key $WANDB_TOKEN \
    --ref.offload

if [ $online -eq 0 ]; then
    rsync /disk/scratch/lox-replication/model ~/lox-replication
fi

# HuggingFaceTB/SmolLM2-360M
# unsloth/Llama-3.2-1B
# meta-llama/Llama-3.2-1B
# meta-llama/Llama-2-7b-hf