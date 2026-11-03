#!/bin/bash
# Claude's side of tools/pod_ship.sh: wait until the pod has pushed pod-results/NAME, then fetch it
# and exit. Run it in the background; its exit is what wakes the session.
#   tools/wait_for_branch.sh NAME [SECONDS_BETWEEN_CHECKS]
name=${1:?usage: wait_for_branch.sh NAME [SECONDS]}
every=${2:-60}
ref=refs/heads/pod-results/$name
cd "$(dirname "$0")/.." || exit 1
until git ls-remote --exit-code origin "$ref" >/dev/null 2>&1; do sleep "$every"; done
git fetch -q origin "+$ref:refs/remotes/origin/pod-results/$name"
echo "pod-results/$name arrived: $(git log -1 --format='%s (%cI)' "origin/pod-results/$name")"
git show "origin/pod-results/$name:POD_STATUS" 2>/dev/null
