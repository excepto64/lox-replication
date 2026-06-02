export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

deepspeed --module openrlhf.cli.train_dpo  \
    --model.model_name_or_path HuggingFaceTB/SmolLM2-360M \
    --model.beta 0.1 \
    --model.gradient_checkpointing_enable \
    --data.dataset Anthropic/hh-rlhf \
    --data.chosen_key chosen \
    --data.rejected_key rejected \
    --data.max_len 1024 \
    --data.max_samples 8000 \
    --train.batch_size 128 \
    --train.micro_batch_size 1 \
    --train.max_epochs 1 \
    --train.seed 48 \
    --adam.lr 5e-6 \
    --ds.packing_samples \
    --ds.zero_stage 3 \
    --ds.param_dtype bf16 \
    --ds.attn_implementation flash_attention_2 \
    --ds.adam_offload \
    --ckpt.output_dir ./model \
    --ckpt.save_steps -1 \
    --eval.steps -1 \
    --logger.logging_steps 1 \
    --logger.wandb.key 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 \
    --ref.offload

# HuggingFaceTB/SmolLM2-360M
# unsloth/Llama-3.2-1B
# meta-llama/Llama-3.2-1B
# meta-llama/Llama-2-7b-hf