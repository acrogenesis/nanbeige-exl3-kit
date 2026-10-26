#!/usr/bin/env bash
set -euo pipefail
KIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXL3_GRAPHS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export EXL3_QC_STAGING_PAGES=64
exec "${NANBEIGE_PYTHON:-$KIT_DIR/.venv/bin/python}" "$KIT_DIR/chat.py" \
  --cache-tokens 180224 --kv-bits 4 --draft-kv-bits 4 --sparse-draft-head 16384 "$@"
