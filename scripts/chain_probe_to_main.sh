#!/bin/bash
# Runs ON THE GPU NODE (nohup): wait for the newest probe log to say ALL DONE, then launch
# main-experiment pass 1. Independent of any ssh session.
CODE=/root/workspace/stylering/code
cd $CODE || exit 1
CHAIN=$CODE/logs/chain.log
echo "[$(date '+%F %T')] chain armed" >> $CHAIN
while true; do
  L=$(ls -t $CODE/logs/probe_2026*.log | head -1)
  if grep -q "PROBE ALL DONE" "$L"; then
    echo "[$(date '+%F %T')] probe done ($L) -> launching main pass1" >> $CHAIN
    nohup bash scripts/main_exp.sh main 100 "none,stylering,gaussmarker" "style" "97,101" > /dev/null 2>&1 < /dev/null &
    echo "[$(date '+%F %T')] main pass1 launched pid=$!" >> $CHAIN
    exit 0
  fi
  if grep -q "STAGE PROBE FAIL" "$L"; then
    echo "[$(date '+%F %T')] probe FAILED ($L) -> not launching main" >> $CHAIN
    exit 1
  fi
  sleep 120
done
