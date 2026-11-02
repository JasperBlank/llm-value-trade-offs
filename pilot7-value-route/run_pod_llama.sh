#!/bin/bash
# Pilot 7, second start of the same pod (after the Qwen run): Llama 3.1 8B, once the Hugging Face
# token in the Runpod secret was corrected. Runs Llama's stages (results hidden), rebuilds the report
# for both models, then waits for both prediction sets and prints, as run_pod.sh does.
set -o pipefail
export HF_HOME=/workspace/hf
REPO=/workspace/llm-value-trade-offs
P7=$REPO/pilot7-value-route
exec > >(tee -a /workspace/log_pilot7_pod.txt) 2>&1
echo "PILOT7L START $(date -u +%FT%TZ)"
cd $REPO && git fetch -q origin main && git reset -q --hard origin/main && cd $P7 || { echo "PILOT7L FAILED: no repo"; exit 1; }
git log -1 --format='PILOT7L commit %h: %s'
REQ=$REPO/pilot5-value-circuit/requirements.txt
python3 -m pip install -q --break-system-packages -r $REQ || python3 -m pip install -q -r $REQ
[ -n "$HF_TOKEN" ] && echo "PILOT7L HF_TOKEN is set (${#HF_TOKEN} characters, starts with ${HF_TOKEN:0:3})"
[ -d results/real/qwen ] || { mkdir -p results && tar xzf /workspace/pilot7_results.tar.gz -C results; }
if [ -f results/real/llama/das.json ]; then
  echo "PILOT7L Llama already run on this pod"
elif python3 -c "from huggingface_hub import hf_hub_download as d; d('meta-llama/Llama-3.1-8B-Instruct', 'config.json')" > /workspace/log_llama_access.txt 2>&1; then
  echo "PILOT7L Llama access OK: running Llama $(date -u +%FT%TZ) (results hidden until both prediction sets are committed)"
  python3 pilot7.py --models llama --stages fit,map,das 2>&1 | tee /workspace/log_pilot7_llama.txt | grep --line-buffered -E "=====|Traceback|Error|out of memory|Killed"
  python3 pilot7.py --models qwen,llama --stages report >> /workspace/log_pilot7_llama.txt 2>&1
  python3 tests/test_mechanics7.py --models llama > /workspace/log_pilot7_llama_checks.txt 2>&1
  grep -E "^(OK|BAD) " /workspace/log_pilot7_llama_checks.txt | cut -c1-60
  cp /workspace/log_pilot7_llama.txt /workspace/log_pilot7_llama_checks.txt results/real/ 2>/dev/null
  tar czf /workspace/pilot7_results.tar.gz -C $P7/results real
  echo "PILOT7L RUN FINISHED $(date -u +%FT%TZ)"
else
  echo "PILOT7L still no Llama access:"; grep -o -E "(401|403|404|GatedRepo|Unauthorized|Access to model[^.]*)" /workspace/log_llama_access.txt | sort -u | head -3
fi
for i in $(seq 1 360); do
  cd $REPO && git fetch -q origin main && git reset -q --hard origin/main
  if [ -f $P7/predictions/JASPER.md ] && [ -f $P7/predictions/CLAUDE.md ] && [ -f $P7/predictions/hashes.sha256 ]; then break; fi
  [ $((i % 10)) -eq 1 ] && echo "PILOT7L WAITING for the predictions before showing results ($(date -u +%H:%M) UTC)"
  sleep 30
done
cd $P7
sha256sum -c predictions/hashes.sha256 || { echo "PILOT7L FAILED: hashes do not match"; exit 1; }
echo "PILOT7 REPORT BEGIN"
python3 print_results7.py results/real
echo "PILOT7 REPORT END"
echo "PILOT7L DONE $(date -u +%FT%TZ)"
