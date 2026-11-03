# Source this from a run_pod script. It replaces "print the results to the log and wait for Claude".
#
#   ship_and_stop STATUS PATH...
#
# STATUS is FINISHED or FAILED; PATHs are files or folders inside $REPO (results, logs).
# With the Runpod secret github_push_token passed in as GH_PUSH_TOKEN, it commits the paths to the
# branch pod-results/$RUN_NAME of the repo, pushes it, and stops the pod at once: no GPU sits idle.
# On Claude's side, tools/wait_for_branch.sh sees the branch appear and wakes the session.
# Without the token it falls back to the old way: wait READ_WAIT seconds for the log to be read, then stop.
#
# Needs: REPO (the clone on the pod), RUN_NAME (e.g. pilot9). The pod's own RUNPOD_POD_ID and
# RUNPOD_API_KEY are injected by Runpod. The token never appears in a command line or the log.

GH_REPO_URL=${GH_REPO_URL:-https://github.com/JasperBlank/llm-value-trade-offs.git}

pod_stop_now() {
  echo "POD STOPPING $(date -u +%FT%TZ)"
  python3 -c "import os,json,urllib.request as u; r=u.Request('https://api.runpod.io/v2/pods/'+os.environ['RUNPOD_POD_ID']+'/action', data=json.dumps({'action':'stop'}).encode(), method='POST', headers={'Authorization':'Bearer '+os.environ.get('RUNPOD_API_KEY',''),'Content-Type':'application/json'}); print('POD stop request', u.urlopen(r).status, flush=True)" \
    || echo "POD stop request FAILED"
  sleep infinity
}

ship_results() {  # STATUS PATH... ; returns non-zero if nothing was pushed
  local status=$1; shift
  [ -n "$GH_PUSH_TOKEN" ] || { echo "POD no github_push_token secret: not pushing"; return 1; }
  local branch=pod-results/$RUN_NAME
  (
    cd "$REPO" || exit 1
    git checkout -q -B "$branch" || exit 1
    printf '%s %s pod %s\n' "$status" "$(date -u +%FT%TZ)" "${RUNPOD_POD_ID:-?}" > POD_STATUS
    git add -f POD_STATUS
    for p in "$@"; do [ -e "$p" ] && git add -f "$p"; done
    git -c user.name="Runpod pod" -c user.email="pod@runpod.invalid" commit -q -m "$RUN_NAME $status on pod ${RUNPOD_POD_ID:-?}" || exit 1
    for try in 1 2 3; do
      git -c credential.helper= \
          -c credential.helper='!f() { echo username=x-access-token; echo "password=$GH_PUSH_TOKEN"; }; f' \
          push -q -f "$GH_REPO_URL" "HEAD:refs/heads/$branch" && exit 0
      sleep 20
    done
    exit 1
  ) && echo "POD PUSHED $branch ($status)" || { echo "POD PUSH FAILED"; return 1; }
}

ship_and_stop() {  # STATUS PATH...
  if ship_results "$@"; then
    pod_stop_now
  fi
  echo "POD WAITING ${READ_WAIT:-2400}s for the log to be read before stopping $(date -u +%FT%TZ)"
  sleep "${READ_WAIT:-2400}"
  pod_stop_now
}
