#!/usr/bin/env bash
set -euo pipefail
KIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXL3_GRAPHS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "${NANBEIGE_PYTHON:-$KIT_DIR/.venv/bin/python}" "$KIT_DIR/chat.py" \
  --kv-bits 0 --sparse-draft-head 16384 "$@"
