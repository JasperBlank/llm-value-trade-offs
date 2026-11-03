#!/bin/bash
# Pilot 9 follow-up (exploratory): the same 576 rating prompts read on a log scale (pilot9_logscale.py).
# Install; quick test on Qwen3-0.6B; the run on Qwen3-14B; then, if the follow-up predictions are in the repo,
# analyse and print everything to the log. With the github_push_token secret the results go to the branch
# pod-results/pilot9-logscale and the pod stops at once; without it the pod waits READ_WAIT seconds, then stops.
set -o pipefail
export HF_HOME=/workspace/hf
REPO=/workspace/llm-value-trade-offs
P9=$REPO/pilot9-vibe-check
RUN_NAME=pilot9-logscale
LOG=/workspace/log_pilot9_logscale_pod.txt
exec > >(tee -a $LOG) 2>&1
source $REPO/tools/pod_ship.sh
finish() {  # STATUS [PATH...]
  local status=$1; shift
  mkdir -p $P9/results/logscale && cp $LOG $P9/results/logscale/log_pod.txt
  ship_and_stop "$status" "$@"
}
fail() { echo "PILOT9L FAILED: $1"; [ -n "$2" ] && tail -30 "$2"; finish FAILED pilot9-vibe-check/results/logscale/log_pod.txt; }
echo "PILOT9L START $(date -u +%FT%TZ)"
cd $P9 || fail "no repo"
git log -1 --format='PILOT9L commit %h: %s'
REQ=$REPO/pilot5-value-circuit/requirements.txt
python3 -m pip install -q --break-system-packages -r $REQ || python3 -m pip install -q -r $REQ || fail pip
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -f results/logscale/ratings9_logscale.json ]; then
  echo "PILOT9L QUICK TEST (Qwen3-0.6B) $(date -u +%FT%TZ)"
  python3 pilot9_logscale.py --quick > /workspace/log_l_quick.txt 2>&1 || fail "quick test" /workspace/log_l_quick.txt
  grep -E "prompts in" /workspace/log_l_quick.txt
  echo "PILOT9L RUN $(date -u +%FT%TZ)"
  python3 pilot9_logscale.py > /workspace/log_l_real.txt 2>&1 || fail "run" /workspace/log_l_real.txt
  grep -E "prompts in" /workspace/log_l_real.txt
  echo "PILOT9L RUN FINISHED $(date -u +%FT%TZ)"
fi
cd $REPO && git fetch -q origin main && git reset -q --hard origin/main && cd $P9
if [ -f predictions/claude_followup.yaml ] && [ -f predictions/FOLLOWUP.md ] && sha256sum -c predictions/hashes_followup.sha256; then
  python3 explore_logscale.py results/logscale > /dev/null || fail analysis
  echo "PILOT9L REPORT BEGIN"
  cat results/logscale/report.md
  echo "PILOT9L REPORT END"
  python3 - <<'EOF'
import json
F = json.load(open("results/logscale/ratings9_logscale.json"))
print("PILOT9L SCORES " + json.dumps(F["score"]))
print("PILOT9L RATINGS " + json.dumps(F["rating"]))
for i in range(0, len(F["logp"]), 48):
    print(f"PILOT9L LOGP {i} " + json.dumps(F["logp"][i:i + 48]))
EOF
  finish FINISHED pilot9-vibe-check/results/logscale
else
  echo "PILOT9L predictions not in the repo: results kept on /workspace, not printed or pushed"
  finish NO-PREDICTIONS pilot9-vibe-check/results/logscale/log_pod.txt
fi
