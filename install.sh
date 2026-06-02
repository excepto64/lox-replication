curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12
source .venv/bin/activate

uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3+cu126torch2.12-cp312-cp312-linux_x86_64.whl"
uv pip install openrlhf --no-build-isolation
uv pip install nvidia-ml-py
uv pip uninstall pynvml