#!/bin/bash
# Offline placement-space probe -> results/probe/survival_map.npz
# usage: bash scripts/probe.sh [N=200] [SEEDS=97,101,202,303,404] [EXTRA...]
source "$(dirname "$0")/env.sh"
N=${1:-200}; SEEDS=${2:-97,101,202,303,404}; shift $(( $# < 2 ? $# : 2 ))
LOG=$CODE/logs/probe_$(date +%Y%m%d_%H%M).log
run_stage PROBE $PY -u src/probe.py --n "$N" --seeds "$SEEDS" "$@"
mark "PROBE ALL DONE"
