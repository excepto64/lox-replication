#!/bin/bash
# align_dpo.sh

set -e

SCRATCH=${1}
config=${2}

local_name=${fine_tune_name##*/}

local_dir=${SCRATCH}/${local_name}
dataset_name=Anthropic/hh-rlhf

MASTER_PORT=$((29500 + ${SLURM_JOB_ID} % 1000))

source ~/lox-replication/.env
source ${SCRATCH}/.venv/bin/activate
source /home/htang2/toolchain-20251006/toolchain.rc

mkdir -p ${local_dir}
rsync -a --exclude .env --exclude .venv ~/lox-replication/ ${local_dir}/
cd ${local_dir}
echo "In $(pwd)"

hf auth login --token ${HF_TOKEN} --no-add-to-git-credential

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

if [ ${lora} -eq 0 ]; then
    GRAD_CKPT="--model.gradient_checkpointing_enable"
    ZERO_STAGE=3
else
    GRAD_CKPT=""
    ZERO_STAGE=2
fi

if [ "${optimiser}" = "adam" ]; then
    optim="adam"
    lr="--adam.lr 5e-6"
elif [ "${optimiser}" = "sgd" ]; then
    # optim="sgd"
    # lr="--sgd.lr te-6"
    # TO DO - implement SGD
    echo -e "SGD is not currently impemented. \n Execution stopped."
    exit 1
else
    echo -e "Optimiser not recognised. \n Execution stopped."
    exit 1
fi


if [ "${method}" = "dpo" ]; then 
    # Modified from LoX paper.
    deepspeed --master_port ${MASTER_PORT} --module openrlhf.cli.train_dpo  \
        --model.model_name_or_path ${model_name} \
        --model.beta 0.1 \
        --data.dataset ${dataset_name} \
        --data.chosen_key chosen \
        --data.rejected_key rejected \
        --data.max_len 1024 \
        --data.max_samples ${num_samples} \
        --train.batch_size ${batch_size} \
        --train.micro_batch_size 1 \
        --train.max_epochs ${num_epochs} \
        --train.seed ${seed} \
        --optim ${optim} \
        ${lr} \
        --ds.packing_samples \
        --ds.zero_stage ${ZERO_STAGE} \
        --ds.param_dtype bf16 \
        --ds.attn_implementation flash_attention_2 \
        --ds.lora.rank ${lora} \
        --ds.lora.alpha $((${lora}*2)) \
        --ckpt.output_dir ./model \
        --ckpt.save_steps ${save_steps} \
        --ckpt.path ./checkpoint \
        --ckpt.load_enable \
        --ckpt.save_hf \
        --eval.steps -1 \
        --logger.logging_steps 1 \
        --logger.wandb.key ${WANDB_TOKEN} \
        ${GRAD_CKPT}
elif [ "${method}" = "sft" ]; then # TO DO Investigate input/output key. 
    deepspeed --master_port ${MASTER_PORT} --module openrlhf.cli.train_sft  \
        --model.model_name_or_path ${model_name} \
        --data.dataset ${dataset_name} \
        --data.input_key assistant \
        --data.output_key response \
        --data.max_len 1024 \
        --data.max_samples ${num_samples} \
        --train.batch_size ${batch_size} \
        --train.micro_batch_size 1 \
        --train.max_epochs ${num_epochs} \
        --train.seed ${seed} \
        --optim ${optim} \
        ${lr} \
        --ds.packing_samples \
        --ds.zero_stage ${ZERO_STAGE} \
        --ds.param_dtype bf16 \
        --ds.attn_implementation flash_attention_2 \
        --ds.lora.rank ${lora} \
        --ds.lora.alpha $((${lora}*2)) \
        --ckpt.output_dir ./model \
        --ckpt.save_steps ${save_steps} \
        --ckpt.path ./checkpoint \
        --ckpt.load_enable \
        --ckpt.save_hf \
        --eval.steps -1 \
        --logger.logging_steps 1 \
        --logger.wandb.key ${WANDB_TOKEN} \
        ${GRAD_CKPT}

mkdir -p ~/lox-replication/model/${local_name}/
hf upload ${fine_tune_name} ./model
for ckpt_dir in ./checkpoint/global_step*_hf; do
    step=$(basename "$ckpt_dir" | sed -E 's/global_step([0-9]+)_hf/\1/')
    hf upload ${fine_tune_name} "$ckpt_dir" --revision "step-${step}"
done
rsync -a ${local_dir}/model ~/lox-replication/model/${local_name}/