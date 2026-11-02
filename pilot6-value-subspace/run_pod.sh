#!/bin/bash
# Pilot 6 on a Runpod pod. The pod's start command clones (or updates) the repo in /workspace,
# copies this script to /tmp and runs it. Steps: install; a quick test on Qwen3-0.6B with the
# mechanical checks; the REAL RUN straight away (no idle GPU), its output kept in
# /workspace/log_pilot6.txt and only progress lines (no numbers) shown in the pod's log; pack the
# results; then, once BOTH prediction sets are in the repo (pre-registration: nobody sees a result
# before that), print the report and the JSON files to the pod's log for Claude to read.
set -o pipefail
export HF_HOME=/workspace/hf
REPO=/workspace/llm-value-trade-offs
P6=$REPO/pilot6-value-subspace
exec > >(tee -a /workspace/log_pilot6_pod.txt) 2>&1
echo "PILOT6 START $(date -u +%FT%TZ)"
cd $P6 || { echo "PILOT6 FAILED: no repo"; exit 1; }
git log -1 --format='PILOT6 commit %h: %s'
REQ=$REPO/pilot5-value-circuit/requirements.txt
python3 -m pip install -q --break-system-packages -r $REQ || python3 -m pip install -q -r $REQ \
  || { echo "PILOT6 FAILED: pip install"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace | tail -1

# progress only: stage lines, which subspace finished training, errors (all numbers stripped)
progress() {
  awk '/===== stage|Traceback|Error|Killed|out of memory/ {print; fflush(); next}
       /  das L[0-9]+ [a-z]+_k[0-9]+:/ {sub(/: held-out.*/, ": trained"); print; fflush()}'
}

if ! python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "PILOT6 NO GPU on this start: no runs, only printing what is stored"
  NOGPU=1
fi
if [ -n "$NOGPU" ]; then
  [ -d results/real ] || { mkdir -p results && tar xzf /workspace/pilot6_results.tar.gz -C results 2>/dev/null; }
elif [ -f /workspace/pilot6_results.tar.gz ]; then
  echo "PILOT6 results already on this pod: not running again"
  [ -d results/real ] || tar xzf /workspace/pilot6_results.tar.gz -C results
else
  if grep -q "ALL OK" /workspace/log_pilot6_quick.txt 2>/dev/null; then
    echo "PILOT6 QUICK TEST already passed on this pod"
  else
    echo "PILOT6 QUICK TEST (Qwen3-0.6B) $(date -u +%FT%TZ)"
    python3 pilot6.py --quick > /workspace/log_pilot6_quick.txt 2>&1 \
      && python3 tests/test_mechanics6.py --quick >> /workspace/log_pilot6_quick.txt 2>&1 \
      || { tail -40 /workspace/log_pilot6_quick.txt; echo "PILOT6 FAILED: quick test"; exit 1; }
    grep -E "^(OK|BAD) " /workspace/log_pilot6_quick.txt
    echo "PILOT6 QUICK OK $(date -u +%FT%TZ)"
  fi
  echo "PILOT6 REAL RUN $(date -u +%FT%TZ) (results hidden until both prediction sets are committed)"
  python3 pilot6.py 2>&1 | tee /workspace/log_pilot6.txt | progress
  rc=$?
  python3 tests/test_mechanics6.py > /workspace/log_pilot6_checks.txt 2>&1
  grep -E "^(OK|BAD) .*(1a|1b|2\.|3a|3b|4\.)" /workspace/log_pilot6_checks.txt | sed 's/ max diff.*\| worst.*\| \[.*//'
  cp /workspace/log_pilot6.txt /workspace/log_pilot6_quick.txt /workspace/log_pilot6_checks.txt results/real/ 2>/dev/null
  if [ $rc -ne 0 ] || [ ! -f results/real/report.md ]; then echo "PILOT6 FAILED: real run (see /workspace/log_pilot6.txt)"; tail -5 /workspace/log_pilot6.txt | grep -v "^\[" ; exit 1; fi
  tar czf /workspace/pilot6_results.tar.gz -C $P6/results real
  echo "PILOT6 RUN FINISHED $(date -u +%FT%TZ): results packed ($(du -h /workspace/pilot6_results.tar.gz | cut -f1))"
fi

# Pre-registration: results are shown only once both prediction sets are in the repo.
for i in $(seq 1 360); do
  cd $REPO && git fetch -q origin main && git reset -q --hard origin/main
  if [ -f $P6/predictions/JASPER.md ] && [ -f $P6/predictions/CLAUDE.md ] && [ -f $P6/predictions/hashes.sha256 ]; then break; fi
  [ $((i % 10)) -eq 1 ] && echo "PILOT6 WAITING for the predictions before showing results ($(date -u +%H:%M) UTC)"
  sleep 30
done
cd $P6
sha256sum -c predictions/hashes.sha256 || { echo "PILOT6 FAILED: predictions missing or hashes do not match"; exit 1; }
echo "PILOT6 REPORT BEGIN"
python3 print_results6.py results/real
echo "PILOT6 REPORT END"
echo "PILOT6 DONE $(date -u +%FT%TZ)"
