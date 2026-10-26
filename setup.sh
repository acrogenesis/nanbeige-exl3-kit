#!/usr/bin/env bash
set -euo pipefail
KIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$KIT_DIR"
if command -v uv >/dev/null 2>&1; then
  if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.12 .venv; fi
  uv pip install --python .venv/bin/python -r requirements.txt
else
  if [[ ! -x .venv/bin/python ]]; then python3.12 -m venv .venv; fi
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python -c 'import sys, torch, exllamav3; assert sys.version_info[:2] == (3, 12); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
