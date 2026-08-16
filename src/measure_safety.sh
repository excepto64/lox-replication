#!/bin/bash
# measure_safety.sh

set -e

SCRATCH=${1}
seed=${2}
cluster=${3}
config=${4}
attacked=${5}

source ${config}

fine_tune_name=${fine_tune_name}_s${seed}
if [ ${attacked} -eq 1 ]; then
    fine_tune_name=${fine_tune_name}_attack_alpaca
fi
local_name=${fine_tune_name##*/}
local_dir=${SCRATCH}/${local_name}

source ${SCRATCH}/.venv/bin/activate
if [ ${cluster} -eq 1 ]; then
    set -a; source ~/lox-replication/.env; set +a
    source /home/htang2/toolchain-20251006/toolchain.rc
    if [ ${attacked} -eq 0 ]; then
        cd ${local_dir}
        trap 'rsync -a ${local_dir}/ ~/lox-replication/${local_name} || [ $? -eq 24 ]' EXIT
    else
        trap 'rsync -a ./ ~/lox-replication/${local_name} || [ $? -eq 24 ]' EXIT
    fi
elif [ ${cluster} -eq 0 ]; then
    set -a; source ${SCRATCH}/.env; set +a
fi

measure_one() {
    local revision=${1}

    local revision_flag=()
    local model_revision_flag=()
    local tags_flag=()
    if [ -n "${revision}" ]; then
        revision_flag=(--revision "${revision}")
        model_revision_flag=(-M "revision=${revision}")
        tags_flag=(--tags "revision:${revision}")
    fi
    echo Measuring ASR for revision ${revision}
    inspect eval src/ASR.py --model hf/${fine_tune_name} -T n=100 -T seed=2 \
        -M chat_template="\"{% for message in messages %}{{ message['content'] }}{% endfor %}\"" \
        -M do_sample=false "${model_revision_flag[@]}" "${tags_flag[@]}"
    
    # echo Extracting ranks for revision ${revision}
    # python src/extract_ranks.py --base-model ${model_name} --model ${fine_tune_name} --k 6 --base "${revision_flag[@]}"

    # echo Analysing kl for revision ${revision}
    # src/analyse_kl.sh ${SCRATCH} ${seed} ${cluster} ${config} "${revision}"
}

if [ ${attacked} -eq 0 ]; then
    num_checkpoints=$((num_samples / (batch_size * save_steps)))
    for i in $(seq 6 ${num_checkpoints}); do
        measure_one "step-$((i * save_steps))"
    done
else
    measure_one ""
fi