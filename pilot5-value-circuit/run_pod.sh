#!/bin/bash
# Pilot 5 without a notebook: the pod's start command copies this script to /tmp and runs it.
# Steps: install; quick test on Qwen3-0.6B (skipped if it already passed on this pod); the REAL RUN
# straight away (no idle GPU), with its output kept in /workspace/log_pilot5.txt and only progress
# lines shown in the pod's log; pack the results; then, once BOTH prediction sets are committed
# (pre-registration: nobody sees a result before that), print the report and the results
# (JSON, base64 tar) to the pod's log for Claude to read.
set -o pipefail
export HF_HOME=/workspace/hf
REPO=/workspace/llm-value-trade-offs
P5=$REPO/pilot5-value-circuit
exec > >(tee -a /workspace/log_pilot5_pod.txt) 2>&1
echo "PILOT5 START $(date -u +%FT%TZ)"
cd $P5 || { echo "PILOT5 FAILED: no repo"; exit 1; }
git log -1 --format='PILOT5 commit %h: %s'
python3 -m pip install -q --break-system-packages -r requirements.txt || python3 -m pip install -q -r requirements.txt \
  || { echo "PILOT5 FAILED: pip install"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace | tail -1

if [ -f /workspace/pilot5_results.tar.gz ]; then
  echo "PILOT5 results already on this pod: not running again"
else
  if grep -q "ALL OK" /workspace/log_pilot5_quick.txt 2>/dev/null; then
    echo "PILOT5 QUICK TEST already passed on this pod"
  else
    echo "PILOT5 QUICK TEST (Qwen3-0.6B)"
    python3 pipeline.py --quick > /workspace/log_pilot5_quick.txt 2>&1 \
      && python3 tests/test_mechanics.py --quick >> /workspace/log_pilot5_quick.txt 2>&1 \
      || { tail -30 /workspace/log_pilot5_quick.txt; echo "PILOT5 FAILED: quick test"; exit 1; }
    echo "PILOT5 QUICK OK"
  fi
  echo "PILOT5 REAL RUN $(date -u +%FT%TZ) (results hidden until both prediction sets are committed)"
  python3 pipeline.py 2>&1 | tee /workspace/log_pilot5.txt \
    | grep --line-buffered -E "===== stage|prompts \(|[0-9]+/[0-9]+ pairs$|Traceback|Error"
  rc=$?
  cp /workspace/log_pilot5.txt /workspace/log_pilot5_quick.txt results/real/ 2>/dev/null
  if [ $rc -ne 0 ] || [ ! -f results/real/report.md ]; then echo "PILOT5 FAILED: real run (see /workspace/log_pilot5.txt)"; exit 1; fi
  tar czf /workspace/pilot5_results.tar.gz -C $P5/results real
  echo "PILOT5 RUN FINISHED $(date -u +%FT%TZ): results packed ($(du -h /workspace/pilot5_results.tar.gz | cut -f1))"
fi

# Pre-registration: results are shown only once both prediction sets are in the repo.
for i in $(seq 1 360); do
  cd $REPO && git fetch -q origin main && git reset -q --hard origin/main
  if [ -f $P5/predictions/JASPER.md ] && [ -f $P5/predictions/CLAUDE.md ] && [ -f $P5/predictions/hashes.sha256 ]; then break; fi
  [ $((i % 10)) -eq 1 ] && echo "PILOT5 WAITING for the predictions before showing results ($(date -u +%H:%M) UTC)"
  sleep 30
done
cd $P5
sha256sum -c predictions/hashes.sha256 || { echo "PILOT5 FAILED: predictions missing or hashes do not match"; exit 1; }
mkdir -p /tmp/p5out && tar xzf /workspace/pilot5_results.tar.gz -C /tmp/p5out
echo "PILOT5 REPORT BEGIN"
cat /tmp/p5out/real/report.md
echo "PILOT5 REPORT END"
(cd /tmp/p5out/real && tar czf /tmp/p5_json.tar.gz *.json *.md log_pilot5.txt 2>/dev/null)
echo "PILOT5 BLOB BEGIN $(du -b /tmp/p5_json.tar.gz | cut -f1) bytes"
base64 -w 400 /tmp/p5_json.tar.gz
echo "PILOT5 BLOB END"
echo "PILOT5 DONE $(date -u +%FT%TZ)"
