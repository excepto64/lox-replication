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
    source /home/htang2/toolchain-20251006/toolchain.rc
    cd ${local_dir}
fi



python src/LoX.py --base-model ${model_name} --model ${fine_tune_name} --lora ${lora}
python src/graph.py --base-model ${model_name} --model ${fine_tune_name} --n-main ${main_dim} --n-sec ${sec_dim}

python src/extract_activations.py --base-model ${model_name} --model ${fine_tune_name} --seed ${seed}
python src/graph.py --base-model ${model_name} --model ${fine_tune_name} --n-main ${main_dim} --n-sec ${sec_dim} --suffix dWX

if [ ${cluster} -eq 1 ]; then
    rsync -a ${local_dir}/ ~/lox-replication/${local_name} || [ $? -eq 24 ]
fi