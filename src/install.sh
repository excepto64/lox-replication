#!/bin/bash

mkdir /disk/scratch/s2028118/lox-replication
cd /disk/scratch/s2028118/lox-replication

pwd
ls -a

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv
uv venv --relocatable --python 3.12
source .venv/bin/activate

# Install packages
uv pip install "torch==2.12.0" --index-url https://download.pytorch.org/whl/cu126
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3+cu126torch2.12-cp312-cp312-linux_x86_64.whl"
uv pip install openrlhf --no-build-isolation
uv pip install nvidia-ml-py
uv pip uninstall pynvml
uv pip install matplotlib