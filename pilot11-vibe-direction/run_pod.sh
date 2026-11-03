#!/bin/bash
# Pilot 11 (one direction for both?) on a Runpod pod: install; quick test on Qwen3-0.6B; the real run (Qwen3-14B);
# then, if both prediction sets are in the repo, score them and print the report to the log. With the
# github_push_token secret the results go to the branch pod-results/pilot11 and the pod stops at once; without it
# the pod waits READ_WAIT seconds, then stops.
set -o pipefail
export HF_HOME=/workspace/hf
REPO=/workspace/llm-value-trade-offs
P11=$REPO/pilot11-vibe-direction
RUN_NAME=pilot11
LOG=/workspace/log_pilot11_pod.txt
exec > >(tee -a $LOG) 2>&1
source $REPO/tools/pod_ship.sh
finish() {  # STATUS [PATH...]
  local status=$1; shift
  mkdir -p $P11/results/real && cp $LOG $P11/results/real/log_pod.txt
  ship_and_stop "$status" "$@"
}
fail() { echo "PILOT11 FAILED: $1"; [ -n "$2" ] && tail -40 "$2"; finish FAILED pilot11-vibe-direction/results/real/log_pod.txt; }
echo "PILOT11 START $(date -u +%FT%TZ)"
cd $P11 || fail "no repo"
git log -1 --format='PILOT11 commit %h: %s'
REQ=$REPO/pilot5-value-circuit/requirements.txt
python3 -m pip install -q --break-system-packages -r $REQ || python3 -m pip install -q -r $REQ || fail pip
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $REPO && git fetch -q origin main && git reset -q --hard origin/main && cd $P11
if [ -f predictions/JASPER.md ] && [ -f predictions/claude_predictions.yaml ] && sha256sum -c predictions/hashes.sha256; then
  echo "PILOT11 QUICK TEST (Qwen3-0.6B) $(date -u +%FT%TZ)"
  python3 pilot11.py --quick > /workspace/log_p11_quick.txt 2>&1 || fail "quick test" /workspace/log_p11_quick.txt
  grep -E "probe done|report written" /workspace/log_p11_quick.txt
  echo "PILOT11 REAL RUN $(date -u +%FT%TZ)"
  python3 pilot11.py 2>&1 | tee /workspace/log_p11_real.txt | grep --line-buffered -E "sentences;|read done|probe done|steer layer|report written|Error|Traceback"
  [ -f results/real/verdicts11.json ] || fail "real run" /workspace/log_p11_real.txt
  python3 score_predictions11.py results/real > /dev/null || echo "PILOT11 scoring failed"
  echo "PILOT11 REPORT BEGIN"
  cat results/real/report.md results/real/predictions_vs_results.md 2>/dev/null
  echo "PILOT11 REPORT END"
  python3 -c "import json; print('PILOT11 READINGS ' + json.dumps([[d['reading_menu_ctx'], d['projection_rating_ctx']] for d in json.load(open('results/real/readings11.json'))]))"
  finish FINISHED pilot11-vibe-direction/results/real
else
  echo "PILOT11 predictions not in the repo: not run"
  finish NO-PREDICTIONS pilot11-vibe-direction/results/real/log_pod.txt
fi
