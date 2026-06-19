#!/bin/bash
# run_analysis.sh

set -e

SCRATCH=${1}
model_name=${2}
fine_tune_name=${3}
main_dim=${4}
sec_dim=${5}
lora=${6}

local_name=${fine_tune_name##*/}
local_dir=${SCRATCH}/${local_name}

# source .venv/bin/activate

source ${SCRATCH}/.venv/bin/activate
source /home/htang2/toolchain-20251006/toolchain.rc

cd ${local_dir}

python src/LoX.py --base-model ${model_name} --model ${fine_tune_name} --lora ${lora}
python src/graph.py --base-model ${model_name} --model ${fine_tune_name} --n-main ${main_dim} --n-sec ${sec_dim}

rsync -a ${local_dir}/ ~/lox-replication/${local_name}