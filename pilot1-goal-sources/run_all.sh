#!/usr/bin/env bash
# Run every step in order. Meant for the two test runs:
#   bash run_all.sh --dry-run    no GPU needed, about a minute
#   bash run_all.sh --quick      small model on the pod, about 10-15 minutes
# For the real run, go step by step (see README) so you can check each output
# and lock the pre-registration before step 4. Without --dry-run or --quick this
# script stops at step 4 until the pre-registration is locked.
set -euo pipefail
cd "$(dirname "$0")"
for step in 00_check_setup 01_extract_direction 02_manipulation_check 03_calibrate_stakes 03b_cue_checks 04_run_main 05_analyze 06_harm_directions 07_compare_predictions 08_character_readout; do
  echo "================ $step $*"
  python "$step.py" "$@"
done
