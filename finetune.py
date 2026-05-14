from unlsoth import FastLanguageModel

import torch

max_seq_length = 2048
load_in_4bit = True

model = FastLanguageModel.from_pretrained(
    model_name = "unsloth/SmolLM2-135M",
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit
)

# dataset
dataset = "Anthropic/hh-rlhf"