#!/bin/bash

model_name=$1
fine_tune_name=$2
main_dim=$3
sec_dim=$4

scratch="/disk/scratch/s2028118/lox-replication"

cd $scratch

source .venv/bin/activate
. /home/htang2/toolchain-20251006/toolchain.rc

python src/LoX.py --base-model $model_name --model $fine_tune_name 
python src/graph.py --n-main $main_dim --n-sec $sec_dim

rsync -a ${scratch}/ ~/lox-replication/

rm -rf /disk/scratch/s2028118