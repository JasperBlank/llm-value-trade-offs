#!/bin/bash
# Pilot 5 without a notebook: the pod's start command copies this script to /tmp and runs it.
# Steps: install, quick test on Qwen3-0.6B, WAIT until both prediction sets are committed
# (pre-registration), the real run, pack the results, print the report to the pod's log.
# Progress lines start with "PILOT5" so they are easy to find in the log.
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
du -sh /workspace/acts /workspace/hf 2>/dev/null

echo "PILOT5 QUICK TEST (Qwen3-0.6B)"
python3 pipeline.py --quick > /workspace/log_pilot5_quick.txt 2>&1 \
  && python3 tests/test_mechanics.py --quick >> /workspace/log_pilot5_quick.txt 2>&1
rc=$?
grep -E "=====|Error|error|Traceback|^[1-4]\.|ALL OK|FAIL" /workspace/log_pilot5_quick.txt | tail -40
[ $rc -eq 0 ] || { tail -30 /workspace/log_pilot5_quick.txt; echo "PILOT5 FAILED: quick test"; exit 1; }
echo "PILOT5 QUICK OK"

# Pre-registration: the real run starts only when both prediction sets are in the repo.
for i in $(seq 1 360); do
  cd $REPO && git fetch -q origin main && git reset -q --hard origin/main
  if [ -f $P5/predictions/JASPER.md ] && [ -f $P5/predictions/CLAUDE.md ] && [ -f $P5/predictions/hashes.sha256 ]; then break; fi
  [ $((i % 10)) -eq 1 ] && echo "PILOT5 WAITING for the predictions ($(date -u +%H:%M) UTC)"
  sleep 30
done
cd $P5
sha256sum -c predictions/hashes.sha256 || { echo "PILOT5 FAILED: predictions missing or hashes do not match"; exit 1; }
git log -1 --format='PILOT5 real run on commit %h: %s'

echo "PILOT5 REAL RUN $(date -u +%FT%TZ)"
python3 pipeline.py 2>&1 | tee /workspace/log_pilot5.txt | grep --line-buffered -vE "Loading weights|it/s\]"
rc=$?
cp /workspace/log_pilot5.txt /workspace/log_pilot5_quick.txt /workspace/log_pilot5_pod.txt results/real/ 2>/dev/null
tar czf /workspace/pilot5_results.tar.gz -C $P5/results real && ls -la /workspace/pilot5_results.tar.gz
df -h /workspace | tail -1
echo "PILOT5 REPORT BEGIN"
cat results/real/report.md 2>/dev/null
echo "PILOT5 REPORT END"
[ $rc -eq 0 ] && echo "PILOT5 DONE $(date -u +%FT%TZ)" || echo "PILOT5 FAILED: real run (see /workspace/log_pilot5.txt)"
