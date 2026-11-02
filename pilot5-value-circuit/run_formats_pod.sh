#!/bin/bash
# Pilot 5's formats extension on a fresh pod (2 Nov): the first pod's GPU was taken and its disk
# (with the value directions) is gone, so the directions are fitted again from a new recording
# (same code, same prompts), the circuit pass is repeated for the all-pieces comparison, and the
# formats stage runs. The first run's steering.json and circuit.json (committed) are used for the
# A/B slope and the A/B top-20 writers. Everything prints to the pod's log at the end.
set -o pipefail
export HF_HOME=/workspace/hf
P5=/workspace/llm-value-trade-offs/pilot5-value-circuit
RES=results/formats_run
exec > >(tee -a /workspace/log_formats_pod.txt) 2>&1
echo "PILOT5F START $(date -u +%FT%TZ)"
cd $P5 || { echo "PILOT5F FAILED: no repo"; exit 1; }
git log -1 --format='PILOT5F commit %h: %s'
python3 -m pip install -q --break-system-packages -r requirements.txt || python3 -m pip install -q -r requirements.txt \
  || { echo "PILOT5F FAILED: pip install"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -f $RES/formats.json ]; then
  mkdir -p $RES && cp results/real/steering.json results/real/circuit.json $RES/
  python3 pipeline.py --results-dir $RES --stages record,directions,circuit,formats,analyze 2>&1 | tee /workspace/log_formats.txt \
    | grep --line-buffered -E "===== stage|prompts \(|option-level|Traceback|Error"
  [ -f $RES/formats.json ] || { echo "PILOT5F FAILED (see /workspace/log_formats.txt)"; exit 1; }
  cp /workspace/log_formats.txt $RES/
fi
echo "PILOT5F REPORT BEGIN"
python3 print_results.py $RES report.md formats.json directions.json record_check.json circuit_identity.json
echo "PILOT5F REPORT END"
echo "PILOT5F DONE $(date -u +%FT%TZ)"
