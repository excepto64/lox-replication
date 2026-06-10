#!/bin/bash
# submit.sh

source ${1}

sbatch \
    --partition Teaching \
    --gres=${gpu} \
    --cpus-per-task=1 \
    --time=12:00:00 \
    --job-name=${job_name} \
     ./src/run.sh ${1}