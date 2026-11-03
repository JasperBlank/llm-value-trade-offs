#!/bin/bash
# Pilot 16 (is the finished price absolute, or relative to the alternative?) on a Runpod pod: install; quick test on
# Qwen3-0.6B; the real run (Qwen3-14B); print the report to the log. With the GitHub_token secret (GH_PUSH_TOKEN) the
# results go to the branch pod-results/pilot16 and the pod stops at once; without it the pod waits READ_WAIT seconds.
set -o pipefail
export HF_HOME=/workspace/hf
REPO=/workspace/llm-value-trade-offs
P16=$REPO/pilot16-relative-or-absolute
RUN_NAME=pilot16
LOG=/workspace/log_pilot16_pod.txt
exec > >(tee -a $LOG) 2>&1
source $REPO/tools/pod_ship.sh
finish() {  # STATUS [PATH...]
  local status=$1; shift
  mkdir -p $P16/results/real && cp $LOG $P16/results/real/log_pod.txt
  ship_and_stop "$status" "$@"
}
fail() { echo "PILOT16 FAILED: $1"; [ -n "$2" ] && tail -40 "$2"; finish FAILED pilot16-relative-or-absolute/results/real/log_pod.txt; }
echo "PILOT16 START $(date -u +%FT%TZ)"
cd $P16 || fail "no repo"
REQ=$REPO/pilot5-value-circuit/requirements.txt
python3 -m pip install -q --break-system-packages -r $REQ || python3 -m pip install -q -r $REQ || fail pip
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $REPO && git fetch -q origin main && git reset -q --hard origin/main && cd $P16
git log -1 --format='PILOT16 commit %h: %s'
[ -f predictions/hashes.sha256 ] && [ -f predictions/claude_predictions.yaml ] && sha256sum -c predictions/hashes.sha256
echo "PILOT16 QUICK TEST (Qwen3-0.6B) $(date -u +%FT%TZ)"
python3 pilot16.py --quick > /workspace/log_p16_quick.txt 2>&1 || fail "quick test" /workspace/log_p16_quick.txt
grep -E "choice:|readout B_nl|Q1 " /workspace/log_p16_quick.txt
echo "PILOT16 REAL RUN $(date -u +%FT%TZ)"
python3 pilot16.py 2>&1 | tee /workspace/log_p16_real.txt | grep --line-buffered -E "alternatives B|choice:|readout |Q1 |Error|Traceback"
[ -f results/real/verdicts16.json ] || fail "real run" /workspace/log_p16_real.txt
cp /workspace/log_p16_real.txt results/real/log_real.txt
echo "PILOT16 REPORT BEGIN"
cat results/real/report.md
echo "PILOT16 REPORT END"
finish FINISHED pilot16-relative-or-absolute/results/real
