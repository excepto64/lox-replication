models=( \
    "excepto64/lox_Llama-3_2-1B_r0_1e_sft_adam_s0" \
    "excepto64/lox_Llama-3_2-1B_r0_1e_sft_adam_s2" \
    "excepto64/lox_Llama-3_2-1B_r0_1e_sft_adam_s26" \
    "excepto64/lox_Llama-3_2-1B_r0_1e_sft_sgd_s0" \
    "excepto64/lox_Llama-3_2-1B_r0_1e_sft_sgd_s2" \
    "excepto64/lox_Llama-3_2-1B_r0_1e_sft_sgd_s26" \
    "excepto64/lox_Llama-3_2-3B_r0_1e_sft_sgd_s0" \
    "excepto64/lox_Llama-3_2-3B_r0_1e_sft_sgd_s2" \
    "excepto64/lox_Llama-3_2-3B_r0_1e_sft_sgd_s26" \
)

for model in "${models[@]}"; do
    hf repo branch create ${model} step-240
done