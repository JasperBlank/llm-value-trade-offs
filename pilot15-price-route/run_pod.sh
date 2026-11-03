#!/bin/bash
# Pilot 15 (where the price is finished, and how it is compared) on a Runpod pod: install; quick test on Qwen3-0.6B;
# the real run (Qwen3-14B); print the report to the log. With the GitHub_token secret (GH_PUSH_TOKEN) the results go to
# the branch pod-results/pilot15 and the pod stops at once; without it the pod waits READ_WAIT seconds, then stops.
# No prediction gate: this pilot has bars fixed in advance but no probability predictions (DESIGN.md section 5).
set -o pipefail
export HF_HOME=/workspace/hf
REPO=/workspace/llm-value-trade-offs
P15=$REPO/pilot15-price-route
RUN_NAME=pilot15
LOG=/workspace/log_pilot15_pod.txt
exec > >(tee -a $LOG) 2>&1
source $REPO/tools/pod_ship.sh
finish() {  # STATUS [PATH...]
  local status=$1; shift
  mkdir -p $P15/results/real && cp $LOG $P15/results/real/log_pod.txt
  ship_and_stop "$status" "$@"
}
fail() { echo "PILOT15 FAILED: $1"; [ -n "$2" ] && tail -40 "$2"; finish FAILED pilot15-price-route/results/real/log_pod.txt; }
echo "PILOT15 START $(date -u +%FT%TZ)"
cd $P15 || fail "no repo"
REQ=$REPO/pilot5-value-circuit/requirements.txt
python3 -m pip install -q --break-system-packages -r $REQ || python3 -m pip install -q -r $REQ || fail pip
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $REPO && git fetch -q origin main && git reset -q --hard origin/main && cd $P15
git log -1 --format='PILOT15 commit %h: %s'
echo "PILOT15 QUICK TEST (Qwen3-0.6B) $(date -u +%FT%TZ)"
python3 pilot15.py --quick > /workspace/log_p15_quick.txt 2>&1 || fail "quick test" /workspace/log_p15_quick.txt
grep -E "read |Q1:|Q2:|Q4:|report written" /workspace/log_p15_quick.txt
echo "PILOT15 REAL RUN $(date -u +%FT%TZ)"
python3 pilot15.py 2>&1 | tee /workspace/log_p15_real.txt | grep --line-buffered -E "sentences;|read |readout |Q1:|route:|Q2:|Q4:|report written|Error|Traceback"
[ -f results/real/verdicts15.json ] || fail "real run" /workspace/log_p15_real.txt
cp /workspace/log_p15_real.txt results/real/log_real.txt
echo "PILOT15 REPORT BEGIN"
cat results/real/report.md
echo "PILOT15 REPORT END"
finish FINISHED pilot15-price-route/results/real
