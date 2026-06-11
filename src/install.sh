#!/bin/bash
# install.sh

SCRATCH=${1}

rm -rf ${SCRATCH}
mkdir -p ${SCRATCH}
cd ${SCRATCH}

pwd
ls -a

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"

source /home/htang2/toolchain-20251006/toolchain.rc
# Create venv
uv venv --python 3.12
source .venv/bin/activate

export UV_LINK_MODE=copy

# Install packages
uv pip install "torch==2.12.0" --index-url https://download.pytorch.org/whl/cu126
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.26/flash_attn-2.8.3+cu126torch2.12-cp312-cp312-win_amd64.whl"
uv pip install openrlhf --no-build-isolation
uv pip install nvidia-ml-py
uv pip uninstall pynvml # Version conflict.
uv pip install matplotlib

echo "Dependencies installed!"