#!/bin/bash
# align.sh
#
# Runs one safety-alignment fine-tune (DPO or SFT, Adam or SGD, as set by
# `config`) via OpenRLHF/deepspeed, on the PKU-SafeRLHF filtered dataset.
# Checkpoints are periodically uploaded to the HF repo `fine_tune_name` as
# they appear under ./checkpoint (a background watcher polls every 10 min,
# plus a final sweep at the end), and the final model is uploaded too.
# Called by run_stage_A.sh, not run directly.
# Args: SCRATCH seed cluster config
# `config` is one of the configs/*.cfg files, sourced to set model_name,
# fine_tune_name, method, optimiser, lora, num_samples, num_epochs,
# batch_size, save_steps.

set -e

SCRATCH=${1}
seed=${2}
cluster=${3}
config=${4}

source ${config}

fine_tune_name=${fine_tune_name}_s${seed}
local_name=${fine_tune_name##*/}

local_dir=${SCRATCH}/${local_name}

source ${SCRATCH}/.venv/bin/activate
if [ ${cluster} -eq 1 ]; then
    MASTER_PORT=$((29500 + ${SLURM_JOB_ID} % 1000))
    # Activate environment
    set -a; source ~/lox-replication/.env; set +a
    source /home/htang2/toolchain-20251006/toolchain.rc
    # Download necessary data.
    mkdir -p ${local_dir}
    # Other concurrent jobs write into ~/lox-replication/model/*, so tolerate
    # rsync's "some files vanished" warning (exit code 24) instead of aborting.
    rsync -a ~/lox-replication/src ~/lox-replication/data ${local_dir}/ || [ $? -eq 24 ]
    cd ${local_dir}

elif [ ${cluster} -eq 0 ]; then
    MASTER_PORT=29500
    set -a; source ${SCRATCH}/.env; set +a
fi

echo "In $(pwd)"
rm -rf ./checkpoint    
hf auth login --token ${HF_TOKEN} --no-add-to-git-credential

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# Sets Lora parameters.
if [ ${lora} -eq 0 ]; then
    GRAD_CKPT="--model.gradient_checkpointing_enable"
    ZERO_STAGE=3
else
    GRAD_CKPT=""
    ZERO_STAGE=2
fi

# Uploads checkpoints to huggingface every 10 minutes.
UPLOADED_MARKER=$(mktemp)
STOP_FILE=$(mktemp -u)
watch_and_upload_checkpoints() {
    while [ ! -f "${STOP_FILE}" ]; do
        for ckpt_dir in ./checkpoint/global_step*_hf; do
            [ -d "${ckpt_dir}" ] || continue
            step=$(basename "${ckpt_dir}" | sed -E 's/global_step([0-9]+)_hf/\1/')
            if ! grep -qx "${step}" "${UPLOADED_MARKER}" 2>/dev/null; then
                echo "${step}" >> "${UPLOADED_MARKER}"   # mark before, not after
                hf upload ${fine_tune_name} "${ckpt_dir}" --revision "step-${step}" \
                    || sed -i "/^${step}\$/d" "${UPLOADED_MARKER}"  # unmark on failure so it retries next tick
            fi
        done
        sleep 600
    done
}
watch_and_upload_checkpoints &
WATCHER_PID=$!
trap 'touch "${STOP_FILE}"; wait ${WATCHER_PID} 2>/dev/null' EXIT

echo ${method}

# Method and optimiser settings
if [ "${method}" = "dpo" ]; then
    echo "Running DPO"
    if [ "${optimiser}" = "adam" ]; then
        optim="adam"
        lr="--adam.lr 5e-6"
    elif [ "${optimiser}" = "sgd" ]; then
        optim="sgd"
        lr="--sgd.lr 1e-2"
    else
        echo -e "Optimiser not recognised. \n Execution stopped."
        exit 1
    fi

    dataset_name=excepto64/PKU-SafeRLHF-filtered-dpo
    # Run DPO aligning.
    # Modified from LoX paper. 
    # Gabriel Jacob Perin, Runjin Chen, Xuxi Chen, et al. Lox: Low-rank extrapolation 
    # robustifies LLM safety against fine-tuning. In Second Conference on Language 
    # Modeling, 2025. https://openreview.net/forum?id=ASS5YD4hL4.
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
elif [ "${method}" = "sft" ]; then
    echo "Running SFT"
    if [ "${optimiser}" = "adam" ]; then
        optim="adam"
        lr="--adam.lr 5e-5"
    elif [ "${optimiser}" = "sgd" ]; then
        optim="sgd"
        lr="--sgd.lr 1e-1"
    else
        echo -e "Optimiser not recognised. \n Execution stopped."
        exit 1
    fi
    # Run SFT aligning.
    dataset_name=excepto64/PKU-SafeRLHF-filtered-sft
    deepspeed --master_port ${MASTER_PORT} --module openrlhf.cli.train_sft  \
        --model.model_name_or_path ${model_name} \
        --data.dataset ${dataset_name} \
        --data.input_key input \
        --data.output_key output \
        --data.max_len 1024 \
        --data.max_samples ${num_samples} \
        --train.batch_size ${batch_size} \
        --train.micro_batch_size 1 \
        --train.max_epochs ${num_epochs} \
        --train.seed ${seed} \
        --optim ${optim} \
        ${lr} \
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
else 
    echo -e "Method not recognised. \n Execution stopped."
    exit 1
fi

touch "${STOP_FILE}"
wait ${WATCHER_PID} 2>/dev/null
trap - EXIT
rm -f "${STOP_FILE}"

# Final sweep in case any checkpoint appeared after the watcher's last poll.
for ckpt_dir in ./checkpoint/global_step*_hf; do
    [ -d "${ckpt_dir}" ] || continue
    step=$(basename "${ckpt_dir}" | sed -E 's/global_step([0-9]+)_hf/\1/')
    if ! grep -qx "${step}" "${UPLOADED_MARKER}" 2>/dev/null; then
        hf upload ${fine_tune_name} "${ckpt_dir}" --revision "step-${step}"
    fi
done
rm -f "${UPLOADED_MARKER}"

# Upload mode
mkdir -p ~/lox-replication/model/${local_name}/
hf upload ${fine_tune_name} ./model
# Pull back results from worker node.
if [ ${cluster} -eq 1 ]; then
    rsync -a ${local_dir}/model ~/lox-replication/model/${local_name}/ || [ $? -eq 24 ]
fi
