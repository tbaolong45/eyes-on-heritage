#!/bin/bash
# Python 3.12 venv in <repo>/.venv with CUDA torch, then the pinned requirements.  usage: bash scripts/setup_env.sh
set -e
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
grep -v -E '^(torch|torchvision)==' requirements.txt | uv pip install -r /dev/stdin
python -c "import torch;print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
echo SETUP_DONE
