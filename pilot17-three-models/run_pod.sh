#!/bin/bash
# Pilot 17 (do Llama and a bigger Qwen price outcomes the same way?) on a Runpod pod with a 96 GB GPU: install; check
# Llama access; quick test on Qwen3-0.6B; the real run on Llama 3.1 8B (if accessible), Qwen3-14B and Qwen3-32B, one after
# the other (a failed model is skipped); print the report. With the GitHub_token secret (GH_PUSH_TOKEN) the results go
# to the branch pod-results/pilot17 and the pod stops at once; without it the pod waits READ_WAIT seconds, then stops.
set -o pipefail
export HF_HOME=/workspace/hf
REPO=/workspace/llm-value-trade-offs
P17=$REPO/pilot17-three-models
RUN_NAME=${RUN_NAME:-pilot17}
OUT=${OUT:-real2}
LOG=/workspace/log_pilot17_pod.txt
exec > >(tee -a $LOG) 2>&1
source $REPO/tools/pod_ship.sh
finish() {  # STATUS [PATH...]
  local status=$1; shift
  mkdir -p $P17/results/$OUT && cp $LOG $P17/results/$OUT/log_pod.txt
  ship_and_stop "$status" "$@"
}
fail() { echo "PILOT17 FAILED: $1"; [ -n "$2" ] && tail -40 "$2"; finish FAILED pilot17-three-models/results/$OUT/log_pod.txt; }
echo "PILOT17 START $(date -u +%FT%TZ)"
cd $P17 || fail "no repo"
REQ=$REPO/pilot5-value-circuit/requirements.txt
python3 -m pip install -q --break-system-packages -r $REQ || python3 -m pip install -q -r $REQ || fail pip
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace | tail -1
cd $REPO && git fetch -q origin main && git reset -q --hard origin/main && cd $P17
git log -1 --format='PILOT17 commit %h: %s'
if python3 -c "from huggingface_hub import hf_hub_download as d; d('meta-llama/Llama-3.1-8B-Instruct', 'config.json')" >/dev/null 2>&1; then
  MODELS=qwen32,qwen14,llama8; echo "PILOT17 Llama access OK: $MODELS"
else
  MODELS=qwen32,qwen14; echo "PILOT17 no Llama access: $MODELS"
fi
MODELS=${MODELS_ONLY:-$MODELS}; echo "PILOT17 models: $MODELS; results/$OUT"
echo "PILOT17 QUICK TEST (Qwen3-0.6B) $(date -u +%FT%TZ)"
python3 pilot17.py --model qwen06 --quick > /workspace/log_p17_quick.txt 2>&1 || fail "quick test" /workspace/log_p17_quick.txt
grep -E "price:|comparator:|done in|FAILED" /workspace/log_p17_quick.txt
echo "PILOT17 REAL RUN $(date -u +%FT%TZ)"
python3 pilot17.py --model $MODELS --out $OUT 2>&1 | tee /workspace/log_p17_real.txt | grep --line-buffered -E "blocks,|price:|comparator:|quantity:|weight:|method check|done in|FAILED|Error|Traceback"
[ -f results/$OUT/report.md ] || fail "real run" /workspace/log_p17_real.txt
cp /workspace/log_p17_real.txt results/$OUT/log_real.txt
echo "PILOT17 REPORT BEGIN"
cat results/$OUT/report.md
echo "PILOT17 REPORT END"
finish FINISHED pilot17-three-models/results/$OUT
