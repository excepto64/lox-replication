#!/bin/bash
# align_dpo.sh

set -e

SCRATCH=${1}
model_name=${2}
fine_tune_name=${3}
lora=${4}
num_epochs=${5}

local_dir=${SCRATCH}/lox_${fine_tune_name}
dataset_name=Anthropic/hh-rlhf

source ~/lox-replication/.env
source ${SCRATCH}/.venv/bin/activate
source /home/htang2/toolchain-20251006/toolchain.rc

mkdir -p ${local_dir}
rsync -a --exclude .env --exclude .venv ~/lox-replication/ ${local_dir}/
cd ${local_dir}
echo "In $(pwd)"

hf auth login --token ${HF_TOKEN} --no-add-to-git-credential

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Modified from LoX paper.
deepspeed --module openrlhf.cli.train_dpo  \
    --model.model_name_or_path ${model_name} \
    --model.beta 0.1 \
    --model.gradient_checkpointing_enable \
    --data.dataset ${dataset_name} \
    --data.chosen_key chosen \
    --data.rejected_key rejected \
    --data.max_len 1024 \
    --data.max_samples 22500 \
    --train.batch_size 128 \
    --train.micro_batch_size 1 \
    --train.max_epochs ${num_epochs} \
    --train.seed 48 \
    --adam.lr 5e-6 \
    --ds.packing_samples \
    --ds.zero_stage 3 \
    --ds.param_dtype bf16 \
    --ds.attn_implementation flash_attention_2 \
    --ds.lora.rank ${lora} \
    --ds.lora.alpha $((${lora}*2)) \
    --ckpt.output_dir ./model \
    --ckpt.save_steps -1 \
    --eval.steps -1 \
    --logger.logging_steps 1 \
    --logger.wandb.key ${WANDB_TOKEN} \

if [ $lora -eq 0 ]; then
    rsync -a ${local_dir}/model/ ~/lox-replication/model/${fine_tune_name}/
    hf upload ${fine_tune_name} ./model 
else
    python -m openrlhf.cli.lora_combiner \
        --model_path ${model_name} \
        --lora_path ./model \
        --output_path ./model-combined \
        --param_dtype bf16
    rsync -a ${local_dir}/model-combined/ ~/lox-replication/model/${fine_tune_name}
    hf upload ${fine_tune_name} ./model-combined 
fi