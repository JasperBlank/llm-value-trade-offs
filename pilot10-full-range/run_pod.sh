#!/bin/bash
# Pilot 10 (the full range of goodness) on a Runpod pod: install; quick test on Qwen3-0.6B; the real run (Qwen3-14B);
# then, if both prediction sets are in the repo, analyse and print everything to the log (the report, then the raw
# results in compact lines for rebuild_from_log10.py). With the github_push_token secret the results go to the branch
# pod-results/pilot10 and the pod stops at once; without it the pod waits READ_WAIT seconds, then stops.
set -o pipefail
export HF_HOME=/workspace/hf
REPO=/workspace/llm-value-trade-offs
P10=$REPO/pilot10-full-range
RUN_NAME=pilot10
LOG=/workspace/log_pilot10_pod.txt
exec > >(tee -a $LOG) 2>&1
source $REPO/tools/pod_ship.sh
finish() {  # STATUS [PATH...]
  local status=$1; shift
  mkdir -p $P10/results/real && cp $LOG $P10/results/real/log_pod.txt
  ship_and_stop "$status" "$@"
}
fail() { echo "PILOT10 FAILED: $1"; [ -n "$2" ] && tail -30 "$2"; finish FAILED pilot10-full-range/results/real/log_pod.txt; }
echo "PILOT10 START $(date -u +%FT%TZ)"
cd $P10 || fail "no repo"
git log -1 --format='PILOT10 commit %h: %s'
REQ=$REPO/pilot5-value-circuit/requirements.txt
python3 -m pip install -q --break-system-packages -r $REQ || python3 -m pip install -q -r $REQ || fail pip
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -f results/real/ratings10.json ]; then
  echo "PILOT10 QUICK TEST (Qwen3-0.6B) $(date -u +%FT%TZ)"
  python3 pilot10.py --quick > /workspace/log_p10_quick.txt 2>&1 || fail "quick test" /workspace/log_p10_quick.txt
  grep -E "in .* min|menu done" /workspace/log_p10_quick.txt
  echo "PILOT10 REAL RUN $(date -u +%FT%TZ)"
  python3 pilot10.py 2>&1 | tee /workspace/log_p10_real.txt | grep --line-buffered -E "menu [0-9]+ of|menu done|in .* min|Error|Traceback"
  [ -f results/real/ratings10.json ] || fail "real run" /workspace/log_p10_real.txt
  echo "PILOT10 RUN FINISHED $(date -u +%FT%TZ)"
fi
cd $REPO && git fetch -q origin main && git reset -q --hard origin/main && cd $P10
if [ -f predictions/JASPER.md ] && [ -f predictions/claude_predictions.yaml ] && sha256sum -c predictions/hashes.sha256 \
   && (cd ../pilot9-vibe-check && sha256sum -c predictions/hashes_followup.sha256); then
  python3 analyze10.py results/real > /dev/null || fail analysis
  python3 score_predictions10.py results/real > /dev/null || echo "PILOT10 scoring failed"
  echo "PILOT10 REPORT BEGIN"
  cat results/real/report.md results/real/predictions_vs_results.md 2>/dev/null
  echo "PILOT10 REPORT END"
  python3 print_raw10.py results/real
  finish FINISHED pilot10-full-range/results/real
else
  echo "PILOT10 predictions not in the repo: results kept on /workspace, not printed or pushed"
  finish NO-PREDICTIONS pilot10-full-range/results/real/log_pod.txt
fi
