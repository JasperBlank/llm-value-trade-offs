#!/bin/bash
# Pilot 13 (the choice recipe in words) on a Runpod pod: install; quick test on Qwen3-0.6B; the real run (Qwen3-14B);
# then score both prediction sets and print the report to the log. With the GitHub_token secret (GH_PUSH_TOKEN) the
# results go to the branch pod-results/pilot13 and the pod stops at once; without it the pod waits READ_WAIT seconds,
# then stops. Runs only if both prediction sets are in the repo and Claude's matches its sealed hash.
set -o pipefail
export HF_HOME=/workspace/hf
REPO=/workspace/llm-value-trade-offs
P13=$REPO/pilot13-recipe-in-words
RUN_NAME=pilot13
LOG=/workspace/log_pilot13_pod.txt
exec > >(tee -a $LOG) 2>&1
source $REPO/tools/pod_ship.sh
finish() {  # STATUS [PATH...]
  local status=$1; shift
  mkdir -p $P13/results/real && cp $LOG $P13/results/real/log_pod.txt
  ship_and_stop "$status" "$@"
}
fail() { echo "PILOT13 FAILED: $1"; [ -n "$2" ] && tail -40 "$2"; finish FAILED pilot13-recipe-in-words/results/real/log_pod.txt; }
echo "PILOT13 START $(date -u +%FT%TZ)"
cd $P13 || fail "no repo"
REQ=$REPO/pilot5-value-circuit/requirements.txt
python3 -m pip install -q --break-system-packages -r $REQ || python3 -m pip install -q -r $REQ || fail pip
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $REPO && git fetch -q origin main && git reset -q --hard origin/main && cd $P13
git log -1 --format='PILOT13 commit %h: %s'
if [ -f predictions/JASPER.md ] && [ -f predictions/claude_predictions.yaml ] && sha256sum -c predictions/hashes.sha256; then
  echo "PILOT13 QUICK TEST (Qwen3-0.6B) $(date -u +%FT%TZ)"
  python3 pilot13.py --quick > /workspace/log_p13_quick.txt 2>&1 || fail "quick test" /workspace/log_p13_quick.txt
  grep -E "recipes:|report written" /workspace/log_p13_quick.txt
  echo "PILOT13 REAL RUN $(date -u +%FT%TZ)"
  python3 pilot13.py 2>&1 | tee /workspace/log_p13_real.txt | grep --line-buffered -E "sentences$|features done|new prices done|states done|recipes:|push (price|rating|disagreement|feature)|push random:(0|19):|report written|Error|Traceback"
  [ -f results/real/verdicts13.json ] || fail "real run" /workspace/log_p13_real.txt
  cp /workspace/log_p13_real.txt results/real/log_real.txt
  python3 score_predictions13.py results/real > /dev/null || echo "PILOT13 scoring failed"
  echo "PILOT13 REPORT BEGIN"
  cat results/real/report.md results/real/predictions_vs_results.md 2>/dev/null
  echo "PILOT13 REPORT END"
  finish FINISHED pilot13-recipe-in-words/results/real
else
  echo "PILOT13 predictions not in the repo: not run"
  finish NO-PREDICTIONS pilot13-recipe-in-words/results/real/log_pod.txt
fi
