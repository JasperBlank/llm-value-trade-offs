#!/bin/bash
# Pilot 8 (intrinsic values) on a Runpod pod: install; quick test on Qwen3-0.6B with the checks; the real run; then,
# if both prediction sets are in the repo (they are committed before the pod is created), print
# everything to the pod's log; finally the pod STOPS ITSELF (no idle GPU). The log stays readable.
set -o pipefail
export HF_HOME=/workspace/hf
REPO=/workspace/llm-value-trade-offs
P8=$REPO/pilot8-intrinsic-values
exec > >(tee -a /workspace/log_pilot8_pod.txt) 2>&1
stop_pod() {
  echo "PILOT8 STOPPING THE POD $(date -u +%FT%TZ)"
  python3 -c "import os,json,urllib.request as u; r=u.Request('https://api.runpod.io/v2/pods/'+os.environ['RUNPOD_POD_ID']+'/action', data=json.dumps({'action':'stop'}).encode(), method='POST', headers={'Authorization':'Bearer '+os.environ.get('RUNPOD_API_KEY',''),'Content-Type':'application/json'}); print('PILOT8 stop request', u.urlopen(r).status, flush=True)" \
    || echo "PILOT8 stop request FAILED"
  sleep infinity
}
echo "PILOT8 START $(date -u +%FT%TZ)"
cd $P8 || { echo "PILOT8 FAILED: no repo"; stop_pod; }
git log -1 --format='PILOT8 commit %h: %s'
REQ=$REPO/pilot5-value-circuit/requirements.txt
python3 -m pip install -q --break-system-packages -r $REQ || python3 -m pip install -q -r $REQ || { echo "PILOT8 FAILED: pip"; stop_pod; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
progress() { grep --line-buffered -E "=====|Traceback|Error|out of memory|Killed"; }
if [ ! -f /workspace/pilot8_results.tar.gz ]; then
  echo "PILOT8 QUICK TEST (Qwen3-0.6B) $(date -u +%FT%TZ)"
  python3 pilot8.py --quick > /workspace/log_pilot8_quick.txt 2>&1 \
    && python3 tests/test_mechanics8.py --quick --run >> /workspace/log_pilot8_quick.txt 2>&1 \
    || { tail -40 /workspace/log_pilot8_quick.txt; echo "PILOT8 FAILED: quick test"; stop_pod; }
  grep -E "^(OK|BAD) " /workspace/log_pilot8_quick.txt | cut -c1-70
  echo "PILOT8 REAL RUN $(date -u +%FT%TZ)"
  python3 pilot8.py 2>&1 | tee /workspace/log_pilot8.txt | progress
  rc=$?
  python3 tests/test_mechanics8.py --run > /workspace/log_pilot8_checks.txt 2>&1
  grep -E "^(OK|BAD) " /workspace/log_pilot8_checks.txt | cut -c1-70
  cp /workspace/log_pilot8.txt /workspace/log_pilot8_quick.txt /workspace/log_pilot8_checks.txt results/real/ 2>/dev/null
  if [ $rc -ne 0 ] || [ ! -f results/real/report.md ]; then echo "PILOT8 FAILED: real run"; grep -A8 Traceback /workspace/log_pilot8.txt | head -30; stop_pod; fi
  tar czf /workspace/pilot8_results.tar.gz -C $P8/results real
  echo "PILOT8 RUN FINISHED $(date -u +%FT%TZ)"
fi
cd $REPO && git fetch -q origin main && git reset -q --hard origin/main && cd $P8
if [ -f predictions/JASPER.md ] && [ -f predictions/CLAUDE.md ] && sha256sum -c predictions/hashes.sha256; then
  echo "PILOT8 REPORT BEGIN"
  python3 print_results8.py results/real
  echo "PILOT8 REPORT END"
else
  echo "PILOT8 predictions not in the repo: results kept on /workspace, not printed"
fi
stop_pod
