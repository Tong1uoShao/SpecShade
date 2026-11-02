#!/bin/bash
# Shared environment for all scripts (source this file).
#   PY        python interpreter (default: python on PATH)
#   CODE_DIR  repository root (default: the parent of this scripts/ directory)
#   HF_HOME   HuggingFace cache holding the prefetched models (optional; see scripts/prefetch_models.sh)
# Experiments run offline from the cache (HF_HUB_OFFLINE=1); export HF_HUB_OFFLINE=0 to allow downloads.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTHONUNBUFFERED=1
PY=${PY:-python}
CODE=${CODE_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$CODE" || exit 1
mkdir -p logs results
stamp(){ date '+%F %T'; }
mark(){ echo "[$(stamp)] STAGE $1" >> "$LOG"; }
run_stage(){  # run_stage <name> <cmd...>
  local name="$1"; shift
  mark "$name START"
  "$@" >> "$LOG" 2>&1 && mark "$name OK" || { mark "$name FAIL"; exit 1; }
}
