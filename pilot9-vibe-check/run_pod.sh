#!/bin/bash
# Pilot 9 (the vibe check) on a Runpod pod: install; quick test on Qwen3-0.6B; the real run (Qwen3-14B);
# then, if both prediction sets are in the repo, analyse and print everything to the log.
# With the github_push_token secret the results go to the branch pod-results/pilot9 and the pod stops at
# once (tools/pod_ship.sh); without it the pod waits READ_WAIT seconds for the log to be read, then stops.
set -o pipefail
export HF_HOME=/workspace/hf
REPO=/workspace/llm-value-trade-offs
P9=$REPO/pilot9-vibe-check
RUN_NAME=pilot9
LOG=/workspace/log_pilot9_pod.txt
exec > >(tee -a $LOG) 2>&1
source $REPO/tools/pod_ship.sh
finish() {  # STATUS [PATH...]: copy the log next to the results and ship
  local status=$1; shift
  mkdir -p $P9/results/real && cp $LOG $P9/results/real/log_pod.txt
  ship_and_stop "$status" "$@"
}
fail() { echo "PILOT9 FAILED: $1"; [ -n "$2" ] && tail -30 "$2"; finish FAILED pilot9-vibe-check/results/real/log_pod.txt; }
echo "PILOT9 START $(date -u +%FT%TZ)"
cd $P9 || fail "no repo"
git log -1 --format='PILOT9 commit %h: %s'
REQ=$REPO/pilot5-value-circuit/requirements.txt
python3 -m pip install -q --break-system-packages -r $REQ || python3 -m pip install -q -r $REQ || fail pip
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -f results/real/ratings9.json ]; then
  echo "PILOT9 QUICK TEST (Qwen3-0.6B) $(date -u +%FT%TZ)"
  python3 pilot9.py --quick > /workspace/log_pilot9_quick.txt 2>&1 || fail "quick test" /workspace/log_pilot9_quick.txt
  grep -E "prompts in" /workspace/log_pilot9_quick.txt
  echo "PILOT9 REAL RUN $(date -u +%FT%TZ)"
  python3 pilot9.py > /workspace/log_pilot9_real.txt 2>&1 || fail "real run" /workspace/log_pilot9_real.txt
  grep -E "prompts in" /workspace/log_pilot9_real.txt
  echo "PILOT9 RUN FINISHED $(date -u +%FT%TZ)"
fi
cd $REPO && git fetch -q origin main && git reset -q --hard origin/main && cd $P9
if [ -f predictions/JASPER.md ] && [ -f predictions/CLAUDE.md ] && sha256sum -c predictions/hashes.sha256; then
  python3 analyze9.py results/real > /dev/null || fail analysis
  python3 score_predictions9.py results/real > /dev/null || echo "PILOT9 scoring failed"
  echo "PILOT9 REPORT BEGIN"
  cat results/real/report.md results/real/predictions_vs_results.md 2>/dev/null
  echo "PILOT9 REPORT END"
  echo "PILOT9 RATINGS $(python3 -c "import json; print(json.dumps(json.load(open('results/real/ratings9.json'))['rating']))")"
  finish FINISHED pilot9-vibe-check/results/real
else
  echo "PILOT9 predictions not in the repo: results kept on /workspace, not printed or pushed"
  finish NO-PREDICTIONS pilot9-vibe-check/results/real/log_pod.txt
fi
