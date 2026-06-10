#!/bin/bash

model_name=$1
fine_tune_name=$2

dataset_name=Anthropic/hh-rlhf
scratch=/disk/scratch/s2028118/lox-replication/


rsync -a --exclude .env --exclude .venv ~/lox-replication/ $scratch
cd $scratch
echo "In $(pwd)"

source ~/lox-replication/.env
source .venv/bin/activate
. /home/htang2/toolchain-20251006/toolchain.rc

hf auth login $HF_TOKEN

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Modified from LoX paper.
deepspeed --module openrlhf.cli.train_dpo  \
    --model.model_name_or_path $model_name \
    --model.beta 0.1 \
    --model.gradient_checkpointing_enable \
    --data.dataset $dataset_name \
    --data.chosen_key chosen \
    --data.rejected_key rejected \
    --data.max_len 1024 \
    --data.max_samples 22500 \
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

rsync -a ${scratch}model/ ~/lox-replication/model/


hf upload $fine_tune_name ./model 

# HuggingFaceTB/SmolLM2-360M
# unsloth/Llama-3.2-1B
# meta-llama/Llama-3.2-1B
# meta-llama/Llama-2-7b-hf