#!/bin/bash
# run_analysis.sh

set -e

SCRATCH=${1}
seed=${2}
cluster=${3}
config=${4}

source ${config}

fine_tune_name=${fine_tune_name}_s${seed}
local_name=${fine_tune_name##*/}
local_dir=${SCRATCH}/${local_name}

# source .venv/bin/activate

source ${SCRATCH}/.venv/bin/activate
if [ ${cluster} -eq 1 ]; then
    source ~/lox-replication/.env
    source /home/htang2/toolchain-20251006/toolchain.rc
    cd ${local_dir}
elif [ ${cluster} -eq 0 ]; then
    source ${SCRATCH}/.env
fi



python src/LoX.py --base-model ${model_name} --model ${fine_tune_name} --lora ${lora}
python src/graph.py --base-model ${model_name} --model ${fine_tune_name} --n-main ${main_dim} --n-sec ${sec_dim}

python src/extract_activations.py --base-model ${model_name} --model ${fine_tune_name} --seed ${seed}
python src/graph.py --base-model ${model_name} --model ${fine_tune_name} --n-main ${main_dim} --n-sec ${sec_dim} --suffix dWX

inspect eval src/ASR.py --model hf/${fine_tune_name} -T n=100 -T seed=2 \
    -M chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}" \
    -M do_sample=false

if [ ${cluster} -eq 1 ]; then
    rsync -a ${local_dir}/ ~/lox-replication/${local_name} || [ $? -eq 24 ]
fi