#!/usr/bin/env bash
# Set up a rented GPU machine or Kaggle session. This installs software only, so it
# costs nothing beyond the machine you're already paying for.
#
# Then log in to Hugging Face yourself with `hf auth login`, and see training/README.md.
set -euo pipefail

cd "$(dirname "$0")/.."
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12 .venv
uv pip install -e '.[dev,data]' --python .venv
uv venv --python 3.12 .venv-train
uv pip install -r training/requirements.txt --python .venv-train
uv pip install -e . --no-deps --python .venv-train

.venv-train/bin/python - <<'EOF'
import nemo, torch
print("NeMo", nemo.__version__, "| torch", torch.__version__, "| CUDA", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.cuda.get_device_properties(0)
    print("GPU", device.name, round(device.total_memory / 2**30), "GB")
EOF
.venv/bin/python -m pytest -q
