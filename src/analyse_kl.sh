#!/bin/bash
# analyse_kl.sh

set -e

SCRATCH=${1}
seed=${2}
cluster=${3}
config=${4}
revision=${5:-}

source ${config}

fine_tune_name=${fine_tune_name}_s${seed}
extracted_name=${fine_tune_name}_extracted_k6
if [ -n "${revision}" ]; then
    extracted_name=${extracted_name}_${revision}
fi
local_name=${fine_tune_name##*/}
local_dir=${SCRATCH}/${local_name}
rev_tag=${revision:-main}
kl_out=${local_dir}/kl_out_${rev_tag}.csv

rev_args=()
base_rev_args=()
if [ -n "${revision}" ]; then
    rev_args=(--revision "${revision}")
    base_rev_args=(--base-revision "${revision}")
fi

source ${SCRATCH}/.venv/bin/activate
if [ ${cluster} -eq 1 ]; then
    source ~/lox-replication/.env
    source /home/htang2/toolchain-20251006/toolchain.rc
    cd ${local_dir}
elif [ ${cluster} -eq 0 ]; then
    source ${SCRATCH}/.env
fi

datasets=("hh-rlhf" "alpaca" "wikipedia" "overrefusal" "overrefusal-toxic")
limit=100

for dataset in "${datasets[@]}"; do
    python src/kl_compare.py \
    --base-model ${model_name} \
    --model ${fine_tune_name} \
    "${rev_args[@]}" \
    --dataset ${dataset} \
    --limit ${limit} \
    --batch-size 10 \
    --out ${kl_out}
done

for dataset in "${datasets[@]}"; do
    python src/kl_compare.py \
    --base-model ${model_name} \
    --model ${extracted_name} \
    --dataset ${dataset} \
    --limit ${limit} \
    --batch-size 10 \
    --out ${kl_out}
done

for dataset in "${datasets[@]}"; do
    python src/kl_compare.py \
    --base-model ${fine_tune_name} \
    "${base_rev_args[@]}" \
    --model ${extracted_name} \
    --dataset ${dataset} \
    --limit ${limit} \
    --batch-size 10 \
    --out ${kl_out}
done

python src/read_difference.py ${kl_out}