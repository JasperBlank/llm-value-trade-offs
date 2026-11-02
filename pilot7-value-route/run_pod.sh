#!/bin/bash
# Pilot 7 on a Runpod pod. The pod's start command clones (or updates) the repo in /workspace,
# copies this script to /tmp and runs it. HF_TOKEN comes from the Runpod secret huggingface_token
# (needed for Llama, which is gated). Steps: install; check Llama access (without it, Qwen only);
# quick test on Qwen3-0.6B with the mechanical checks; the REAL RUN straight away, with only stage
# lines (no numbers) in the pod's log; pack the results; once BOTH prediction sets are in the repo,
# print everything to the pod's log for Claude to read.
set -o pipefail
export HF_HOME=/workspace/hf
REPO=/workspace/llm-value-trade-offs
P7=$REPO/pilot7-value-route
exec > >(tee -a /workspace/log_pilot7_pod.txt) 2>&1
echo "PILOT7 START $(date -u +%FT%TZ)"
cd $P7 || { echo "PILOT7 FAILED: no repo"; exit 1; }
git log -1 --format='PILOT7 commit %h: %s'
REQ=$REPO/pilot5-value-circuit/requirements.txt
python3 -m pip install -q --break-system-packages -r $REQ || python3 -m pip install -q -r $REQ \
  || { echo "PILOT7 FAILED: pip install"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -n "$HF_TOKEN" ] && echo "PILOT7 HF_TOKEN is set (${#HF_TOKEN} characters)" || echo "PILOT7 HF_TOKEN is NOT set"
if python3 -c "from huggingface_hub import hf_hub_download as d; d('meta-llama/Llama-3.1-8B-Instruct', 'config.json')" >/dev/null 2>&1; then
  MODELS=qwen,llama; echo "PILOT7 Llama access OK: models $MODELS"
else
  MODELS=qwen; echo "PILOT7 no Llama access (not approved yet, or no token): Qwen only"
fi
progress() { grep --line-buffered -E "=====|Traceback|Error|out of memory|Killed"; }

if python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  if [ -f /workspace/pilot7_results.tar.gz ]; then
    echo "PILOT7 results already on this pod: not running again"
    [ -d results/real ] || { mkdir -p results && tar xzf /workspace/pilot7_results.tar.gz -C results; }
  else
    echo "PILOT7 QUICK TEST (Qwen3-0.6B) $(date -u +%FT%TZ)"
    python3 pilot7.py --quick > /workspace/log_pilot7_quick.txt 2>&1 \
      && python3 tests/test_mechanics7.py --quick >> /workspace/log_pilot7_quick.txt 2>&1 \
      || { tail -40 /workspace/log_pilot7_quick.txt; echo "PILOT7 FAILED: quick test"; exit 1; }
    grep -E "^(OK|BAD) " /workspace/log_pilot7_quick.txt | sed 's/ max diff.*\| worst.*\| \[.*//'
    echo "PILOT7 QUICK OK $(date -u +%FT%TZ)"
    echo "PILOT7 REAL RUN $(date -u +%FT%TZ), models $MODELS (results hidden until both prediction sets are committed)"
    python3 pilot7.py --models $MODELS 2>&1 | tee /workspace/log_pilot7.txt | progress
    rc=$?
    python3 tests/test_mechanics7.py --models $MODELS > /workspace/log_pilot7_checks.txt 2>&1
    grep -E "^(OK|BAD) " /workspace/log_pilot7_checks.txt | sed 's/ max diff.*\| worst.*\| \[.*\| [0-9]* bad$//'
    cp /workspace/log_pilot7.txt /workspace/log_pilot7_quick.txt /workspace/log_pilot7_checks.txt results/real/ 2>/dev/null
    if [ $rc -ne 0 ] || [ ! -f results/real/report.md ]; then echo "PILOT7 FAILED: real run"; grep -A5 Traceback /workspace/log_pilot7.txt | head -30; exit 1; fi
    tar czf /workspace/pilot7_results.tar.gz -C $P7/results real
    echo "PILOT7 RUN FINISHED $(date -u +%FT%TZ)"
  fi
else
  echo "PILOT7 NO GPU on this start: only printing what is stored"
  [ -d results/real ] || { mkdir -p results && tar xzf /workspace/pilot7_results.tar.gz -C results; }
fi

# Pre-registration: results are shown only once both prediction sets are in the repo.
for i in $(seq 1 360); do
  cd $REPO && git fetch -q origin main && git reset -q --hard origin/main
  if [ -f $P7/predictions/JASPER.md ] && [ -f $P7/predictions/CLAUDE.md ] && [ -f $P7/predictions/hashes.sha256 ]; then break; fi
  [ $((i % 10)) -eq 1 ] && echo "PILOT7 WAITING for the predictions before showing results ($(date -u +%H:%M) UTC)"
  sleep 30
done
cd $P7
sha256sum -c predictions/hashes.sha256 || { echo "PILOT7 FAILED: predictions missing or hashes do not match"; exit 1; }
echo "PILOT7 REPORT BEGIN"
python3 print_results7.py results/real
echo "PILOT7 REPORT END"
echo "PILOT7 DONE $(date -u +%FT%TZ)"
