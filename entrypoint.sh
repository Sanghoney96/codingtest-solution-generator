#!/bin/bash
set -e

echo "===================================="
echo "▶ Checking Python kernel"
PYTHON_PATH="$(which python)"
echo "python path: $PYTHON_PATH"
python --version

if [[ "$PYTHON_PATH" != "/usr/bin/python"* ]]; then
  echo "❌ Unexpected python detected"
  echo "This pod expects system python with preinstalled PyTorch"
  exit 1
fi

echo "✅ System python detected"
echo "===================================="

echo "▶ Checking PyTorch"
python - << 'EOF'
import torch
print("torch version:", torch.__version__)
assert torch.__version__.startswith("2.4.0"), "PyTorch version mismatch"
print("cuda available:", torch.cuda.is_available())
EOF

echo "===================================="
echo "▶ Upgrade pip"
python -m pip install --upgrade pip

echo "▶ Install from requirements.txt"
pip install -r requirements.txt

echo "▶ Install flash-attn (no build isolation)"
pip show flash-attn >/dev/null 2>&1 || \
pip install flash-attn --no-build-isolation

echo "===================================="
echo "✅ Environment setup finished"
echo "===================================="

exec "$@"
