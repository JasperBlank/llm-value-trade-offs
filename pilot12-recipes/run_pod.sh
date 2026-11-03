#!/bin/bash
# Pilot 12 (one vibe or different recipes?) on a Runpod pod: install; quick test on Qwen3-0.6B; the real run (Qwen3-14B);
# then, if both prediction sets are in the repo, score them and print the report to the log. With the
# github_push_token secret the results go to the branch pod-results/pilot12 and the pod stops at once; without it
# the pod waits READ_WAIT seconds, then stops.
set -o pipefail
export HF_HOME=/workspace/hf
REPO=/workspace/llm-value-trade-offs
P12=$REPO/pilot12-recipes
RUN_NAME=pilot12
LOG=/workspace/log_pilot12_pod.txt
exec > >(tee -a $LOG) 2>&1
source $REPO/tools/pod_ship.sh
finish() {  # STATUS [PATH...]
  local status=$1; shift
  mkdir -p $P12/results/real && cp $LOG $P12/results/real/log_pod.txt
  ship_and_stop "$status" "$@"
}
fail() { echo "PILOT12 FAILED: $1"; [ -n "$2" ] && tail -40 "$2"; finish FAILED pilot12-recipes/results/real/log_pod.txt; }
echo "PILOT12 START $(date -u +%FT%TZ)"
cd $P12 || fail "no repo"
git log -1 --format='PILOT12 commit %h: %s'
REQ=$REPO/pilot5-value-circuit/requirements.txt
python3 -m pip install -q --break-system-packages -r $REQ || python3 -m pip install -q -r $REQ || fail pip
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $REPO && git fetch -q origin main && git reset -q --hard origin/main && cd $P12
if [ -f predictions/JASPER.md ] && [ -f predictions/claude_predictions.yaml ] && sha256sum -c predictions/hashes.sha256; then
  echo "PILOT12 QUICK TEST (Qwen3-0.6B) $(date -u +%FT%TZ)"
  python3 pilot12.py --quick > /workspace/log_p12_quick.txt 2>&1 || fail "quick test" /workspace/log_p12_quick.txt
  grep -E "Q1:|report written" /workspace/log_p12_quick.txt
  echo "PILOT12 REAL RUN $(date -u +%FT%TZ)"
  python3 pilot12.py 2>&1 | tee /workspace/log_p12_real.txt | grep --line-buffered -E "sentences;|read done|Q1:|push |report written|Error|Traceback"
  [ -f results/real/verdicts12.json ] || fail "real run" /workspace/log_p12_real.txt
  python3 score_predictions12.py results/real > /dev/null || echo "PILOT12 scoring failed"
  echo "PILOT12 REPORT BEGIN"
  cat results/real/report.md results/real/predictions_vs_results.md 2>/dev/null
  echo "PILOT12 REPORT END"
  python3 -c "import json; print('PILOT12 DISAGREEMENT ' + json.dumps([round(d['d_heldout'], 4) for d in json.load(open('results/real/disagreement12.json'))]))"
  finish FINISHED pilot12-recipes/results/real
else
  echo "PILOT12 predictions not in the repo: not run"
  finish NO-PREDICTIONS pilot12-recipes/results/real/log_pod.txt
fi
