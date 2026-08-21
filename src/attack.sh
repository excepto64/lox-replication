#!/bin/bash
# attack.sh
#
# Benign fine-tuning attack: SFTs an already-aligned model (config's
# fine_tune_name, at `revision` if set) on the Alpaca instruction-following
# dataset via attack.py, pushing the result to the HF Hub as
# <fine_tune_name>_attack_alpaca. This is the "benign fine-tuning" threat
# model this project studies -- fine-tuning for a legitimate task that
# incidentally degrades safety.
# Called by run_stage_B.sh, not run directly.
# Args: SCRATCH seed cluster config

set -e

SCRATCH=${1}
seed=${2}
cluster=${3}
config=${4}

source ${config}
dataset=alpaca

fine_tune_name=${fine_tune_name}_s${seed}
attacked_name=${fine_tune_name}_attack_${dataset}
local_name=${fine_tune_name##*/}

local_dir=${SCRATCH}/${local_name}

source ${SCRATCH}/.venv/bin/activate
if [ ${cluster} -eq 1 ]; then
    set -a; source ~/lox-replication/.env; set +a
    source /home/htang2/toolchain-20251006/toolchain.rc

    mkdir -p ${local_dir}
    # Other concurrent jobs write into ~/lox-replication/model/*, so tolerate
    # rsync's "some files vanished" warning (exit code 24) instead of aborting.
    rsync -a ~/lox-replication/src ~/lox-replication/data ${local_dir}/ || [ $? -eq 24 ]
    cd ${local_dir}

elif [ ${cluster} -eq 0 ]; then
    set -a; source ${SCRATCH}/.env; set +a
fi

revision_flag=()
if [ -n "${revision}" ]; then
    revision_flag=(--revision "${revision}")
fi

python src/attack.py \
    --base-model ${fine_tune_name} \
    --dataset ${dataset} \
    --fine-tune-name ${attacked_name} \
    "${revision_flag[@]}"