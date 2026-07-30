#!/bin/bash
# measure_update.sh

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
    source ~/lox-replication/.env
    source /home/htang2/toolchain-20251006/toolchain.rc
    cd ${local_dir}
elif [ ${cluster} -eq 0 ]; then
    source ${SCRATCH}/.env
fi

num_checkpoints=$((num_samples / (batch_size * save_steps)))

for i in $(seq 1 ${num_checkpoints}); do
    revision="step-$((i * save_steps))"

    python src/LoX.py --base-model ${model_name} --model ${fine_tune_name} --lora ${lora} --revision ${revision}
    python src/graph.py --base-model ${model_name} --model ${fine_tune_name} --n-main ${main_dim} --n-sec ${sec_dim} --revision ${revision}

    python src/extract_activations.py --base-model ${model_name} --model ${fine_tune_name} --seed ${seed} --revision ${revision}
    python src/graph.py --base-model ${model_name} --model ${fine_tune_name} --n-main ${main_dim} --n-sec ${sec_dim} --suffix dWX --revision ${revision}
done

if [ ${cluster} -eq 1 ]; then
    rsync -a ${local_dir}/ ~/lox-replication/${local_name} || [ $? -eq 24 ]
fi
