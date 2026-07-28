#!/bin/bash
# install.sh

SCRATCH=${1}
cluster=${2}

echo $(date)

if [ ${cluster} -eq 1 ]; then
    rm -rf ${SCRATCH}
    mkdir -p ${SCRATCH}
    cd ${SCRATCH}
fi

pwd
ls -a

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"

if [ ${cluster} -eq 1 ]; then
    source /home/htang2/toolchain-20251006/toolchain.rc
fi
# Create venv
uv venv --python 3.12
source .venv/bin/activate

export UV_LINK_MODE=copy

# Install packages
uv pip install "torch==2.12.0" --index-url https://download.pytorch.org/whl/cu126
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3+cu126torch2.12-cp312-cp312-linux_x86_64.whl"
uv pip install openrlhf==0.10.3 --no-build-isolation
uv pip install nvidia-ml-py==13.610.43
uv pip uninstall pynvml # Version conflict.
uv pip install matplotlib==3.10.9
uv pip install inspect-ai==0.3.249
uv pip install openai==2.48.0
uv pip isntall trl==1.9.2

echo "Dependencies installed!"

# Add SGD optimizer support (--optim=sgd) to train_sft.py / train_dpo.py.
# Patch is line/context-sensitive to openrlhf==0.10.3; if the pinned version
# above changes, regenerate this patch against the new source.
if [ ${cluster} -eq 1 ]; then
    PATCH_FILE=~/lox-replication/src/patches/openrlhf-0.10.3-sgd.patch
else
    PATCH_FILE=src/patches/openrlhf-0.10.3-sgd.patch
fi
patch -p1 -d "$(python3 -c 'import openrlhf, os; print(os.path.dirname(openrlhf.__file__))')" \
    < "${PATCH_FILE}"
if [ $? -ne 0 ]; then
    echo "SGD optimizer patch FAILED to apply!" >&2
    exit 1
fi

echo "SGD optimizer patch applied!"