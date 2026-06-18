#!/bin/bash
# submit.sh

SCRATCH=bla

runs=("SmolLM2-360M_r6_1e.cfg" "SmolLM2-360M_r0_1e.cfg" "Llama-3_2-1B_r6_1e.cfg" "Llama-3_2-1B_r0_1e.cfg")

for run in "${runs[@]}"; do
    source ${run}
    ./src/run.sh ${SCRATCH} ${run} 
done