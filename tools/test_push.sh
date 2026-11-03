#!/bin/bash
# A one-cent check that a pod can push results with the github_push_token secret (run on a small CPU pod).
# Pushes tools/pushtest/ok.txt to the branch pod-results/pushtest, then the pod stops itself.
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN_NAME=pushtest
source "$REPO/tools/pod_ship.sh"
mkdir -p "$REPO/tools/pushtest"
echo "pushed from pod ${RUNPOD_POD_ID:-?} at $(date -u +%FT%TZ)" > "$REPO/tools/pushtest/ok.txt"
READ_WAIT=${READ_WAIT:-300}
ship_and_stop FINISHED tools/pushtest/ok.txt
