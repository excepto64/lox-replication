#!/bin/bash

mkdir -p /disk/scratch/s2028118/lox-replication
cd /disk/scratch/s2028118/lox-replication

pwd
ls -a

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"

. /home/htang2/toolchain-20251006/toolchain.rc
# Create venv
uv venv --python 3.12
source .venv/bin/activate

export UV_LINK_MODE=copy

# Install packages
uv pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu128torch2.11-cp312-cp312-linux_x86_64.whl"
uv pip install openrlhf --no-build-isolation
uv pip install nvidia-ml-py
uv pip uninstall pynvml
uv pip install matplotlib