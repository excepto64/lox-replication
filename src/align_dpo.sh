#!/bin/bash

model_name=${1}
fine_tune_name=${2}
lora=${3}
num_epochs=${4}

dataset_name=Anthropic/hh-rlhf
scratch=/disk/scratch/s2028118/lox-replication


rsync -a --exclude .env --exclude .venv ~/lox-replication/ ${scratch}/
cd ${scratch}
echo "In $(pwd)"

source ~/lox-replication/.env
source .venv/bin/activate
. /home/htang2/toolchain-20251006/toolchain.rc

hf auth login --token ${HF_TOKEN} --no-add-to-git-credential

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

while true; do
    sleep 1800  # every 30 mins
    rsync -a ${scratch}/model/${fine_tune_name}/ ~/lox-replication/model/${fine_tune_name}/
done &
RSYNC_PID=$!

# Modified from LoX paper.
deepspeed --num_gpus=1 --module openrlhf.cli.train_dpo  \
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
    --ds.adam_offload \
    --ds.lora.rank ${lora} \
    --ds.lora.alpha $((${lora}*2)) \
    --ckpt.output_dir ./model/${fine_tune_name} \
    --ckpt.save_steps -1 \
    --eval.steps -1 \
    --logger.logging_steps 1 \
    --logger.wandb.key ${WANDB_TOKEN} \
    --ref.offload

if [ $lora -eq 0 ]; then
    rsync -a ${scratch}/model/${fine_tune_name}/ ~/lox-replication/model/${fine_tune_name}
else
    python -m openrlhf.cli.lora_combiner \
        --model_path ${model_name} \
        --lora_path ./model/${fine_tune_name} \
        --output_path ./model/${fine_tune_name}-combined \
        --param_dtype bf16
    rsync -a ${scratch}/model/${fine_tune_name}-combined/ ~/lox-replication/model/${fine_tune_name}
fi

hf upload ${fine_tune_name} ./model 