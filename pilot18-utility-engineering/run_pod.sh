#!/bin/bash
# Pilot 18 (Utility Engineering on Llama 3.1 8B, and is the utility used?) on a Runpod pod with an H100 (DESIGN.md):
#   1. our dependencies; their repo at commit 5e5966d; their environment (in the background)
#   2. the model at their revision; their released results and the paper's figure pages -> pod-results/pilot18-paper
#   3. a quick test of pilot18.py on synthetic utilities (fail fast)
#   4. their code, unmodified except the model path: utilities (K = 10, then logprobs mode), transitivity
#   5. pilot18.py: steps 1-3; the report
# With the GitHub_token secret (GH_PUSH_TOKEN) the results go to pod-results/pilot18 and the pod stops at once.
set -o pipefail
export HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=false
REPO=/workspace/llm-value-trade-offs
P18=$REPO/pilot18-utility-engineering
export UE=/workspace/emergent-values
UE_COMMIT=5e5966d
export VENV=/workspace/uevenv
VPY=$VENV/bin/python
RUN_NAME=${RUN_NAME:-pilot18}
OUT=${OUT:-real}
RES=$P18/results/$OUT
LOG=/workspace/log_pilot18_pod.txt
exec > >(tee -a $LOG) 2>&1
source $REPO/tools/pod_ship.sh
finish() {  # STATUS
  mkdir -p $RES && cp $LOG $RES/log_pod.txt
  for f in /workspace/log_ue_*.txt /workspace/log_p18_*.txt /workspace/log_fetch.txt; do
    [ -f "$f" ] && tail -c 2000000 "$f" > $RES/$(basename $f)
  done
  # their raw results files (every response) are large: compressed, and left out if still too big for git
  for f in $(find $RES -name "results_${KEY}.json" -o -name "triad_results_${KEY}.json" 2>/dev/null); do
    gzip -9f "$f"; [ $(stat -c %s "$f.gz") -gt 90000000 ] && rm -f "$f.gz" && echo "dropped $f.gz (too big)"
  done
  ship_and_stop "$1" pilot18-utility-engineering/results/$OUT pilot18-utility-engineering/results/theirs pilot18-utility-engineering/results/quick
}
fail() { echo "PILOT18 FAILED: $1"; [ -n "$2" ] && tail -60 "$2"; finish FAILED; }
KEY=llama-31-8b-instruct
echo "PILOT18 START $(date -u +%FT%TZ)"
cd $REPO && git fetch -q origin main && git reset -q --hard origin/main && cd $P18 || fail "no repo"
git log -1 --format='PILOT18 commit %h: %s'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h /workspace | tail -1
python3 --version

# ---- 1. dependencies, their repo, their environment
python3 -m pip install -q --break-system-packages -r $REPO/pilot5-value-circuit/requirements.txt pymupdf huggingface_hub \
  || python3 -m pip install -q -r $REPO/pilot5-value-circuit/requirements.txt pymupdf huggingface_hub || fail pip
[ -d $UE/.git ] || git clone -q https://github.com/centerforaisafety/emergent-values.git $UE || fail "their repo"
(cd $UE && git checkout -q -f $UE_COMMIT && git checkout -q -- .) || fail "their commit"
(cd $UE && git log -1 --format='PILOT18 their code %h: %s')
bash $P18/ue_setup.sh > /workspace/log_ue_setup.txt 2>&1 &
SETUP_PID=$!

# ---- 2. the model; their results; the paper's figures
MODEL=$(python3 $P18/fetch_theirs.py model 2>/workspace/log_fetch.txt | tail -1)
[ -d "$MODEL" ] || fail "model download" /workspace/log_fetch.txt
echo "PILOT18 model at $MODEL"
python3 $P18/fetch_theirs.py theirs --out $P18/results/theirs --paper /workspace/paper >> /workspace/log_fetch.txt 2>&1 \
  || echo "PILOT18 fetching their results had errors"
grep -E "their results|members|paper|Figure|revision|listing|download" /workspace/log_fetch.txt | head -30
if [ -n "$GH_PUSH_TOKEN" ] && [ "${PAPER_PUSH:-0}" = 1 ]; then   # the figure pages: a short-lived branch of their own (read once, 3 Nov)
  (rm -rf /workspace/paperpush && git clone -q --depth 1 "file://$REPO" /workspace/paperpush && cd /workspace/paperpush \
   && mkdir -p pilot18-paper && cp /workspace/paper/*.png pilot18-paper/ 2>/dev/null; cp -r $P18/results/theirs pilot18-paper/ 2>/dev/null
   git checkout -q -B pod-results/pilot18-paper && git add -f pilot18-paper \
   && git -c user.name="Runpod pod" -c user.email="pod@runpod.invalid" commit -q -m "pilot18: the paper's figure pages and their results" \
   && git -c credential.helper= -c credential.helper='!f() { echo username=x-access-token; echo "password=$GH_PUSH_TOKEN"; }; f' \
        push -q -f "$GH_REPO_URL" HEAD:refs/heads/pod-results/pilot18-paper) && echo "PILOT18 paper pages pushed" || echo "PILOT18 paper push failed"
fi

# ---- 3. quick test of our code (synthetic utilities, 40 outcomes)
echo "PILOT18 QUICK TEST $(date -u +%FT%TZ)"
python3 pilot18.py --model $MODEL --quick --ue_repo $UE > /workspace/log_p18_quick.txt 2>&1 || fail "quick test" /workspace/log_p18_quick.txt
grep -E "step|pipeline|cycles|done in" /workspace/log_p18_quick.txt

# ---- 4. their code
wait $SETUP_PID || fail "their environment" /workspace/log_ue_setup.txt
tail -3 /workspace/log_ue_setup.txt
python3 - "$UE/utility_analysis/models.yaml" "$MODEL" <<'EOF' || fail "models.yaml"
import sys
p, path = sys.argv[1], sys.argv[2]
s = open(p).read()
old = "/data/huggingface/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
assert s.count(old) == 1, "their Llama path is not in models.yaml"
open(p, "w").write(s.replace(old, path))
print("PILOT18 models.yaml now points at", path)
EOF
mkdir -p $RES
COMMON="--options_path ../../shared_options/options_hierarchical.json --compute_utilities_config_path ../../compute_utilities/compute_utilities.yaml --create_agent_config_path ../../compute_utilities/create_agent.yaml --create_agent_config_key default"
cd $UE/utility_analysis/experiments/compute_utilities
for cfg in thurstonian_active_learning thurstonian_active_learning_logprobs; do
  sub=ue_k10; [ $cfg = thurstonian_active_learning_logprobs ] && sub=ue_logprobs
  echo "PILOT18 THEIR UTILITIES ($cfg) $(date -u +%FT%TZ)"
  $VPY optimize_utility_model.py --model_key $KEY --save_dir $RES/$sub --save_suffix $KEY --compute_utilities_config_key $cfg $COMMON \
    > /workspace/log_ue_$sub.txt 2>&1 || fail "their utilities ($cfg)" /workspace/log_ue_$sub.txt
  grep -E -A3 "Holdout Set Metrics|Total time" /workspace/log_ue_$sub.txt | head -8
done
echo "PILOT18 THEIR TRANSITIVITY $(date -u +%FT%TZ)"
cd $UE/utility_analysis/experiments/transitivity
$VPY evaluate_transitivity.py --model_key $KEY --K 10 --save_suffix $KEY --save_dir $RES/ue_transitivity --system_message "You are a helpful assistant." \
  --sample_size 1000 --seed 42 --options_path ../../shared_options/options_hierarchical.json \
  --create_agent_config_path ../../compute_utilities/create_agent.yaml --create_agent_config_key default \
  > /workspace/log_ue_transitivity.txt 2>&1 || fail "their transitivity" /workspace/log_ue_transitivity.txt
grep -E "Saved|Total time" /workspace/log_ue_transitivity.txt

# ---- 5. our steps
echo "PILOT18 REAL RUN $(date -u +%FT%TZ)"
cd $P18
python3 pilot18.py --model $MODEL --ue $RES --out $RES --ue_repo $UE 2>&1 | tee /workspace/log_p18_real.txt \
  | grep --line-buffered -E "step|pipeline|cycles|sweep|done in|Error|Traceback"
[ -f $RES/report.md ] || fail "real run" /workspace/log_p18_real.txt
echo "PILOT18 REPORT BEGIN"
cat $RES/report.md
echo "PILOT18 REPORT END"
finish FINISHED
