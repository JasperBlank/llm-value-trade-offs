# Tools

| File | What |
|---|---|
| `pod_ship.sh` | Sourced by a run_pod script. At the end of a run (or on failure), the pod commits its results to the branch `pod-results/NAME`, pushes it, and stops itself at once. Needs the Runpod secret `github_push_token` (a fine-grained GitHub token, Contents read and write, this repo only) passed to the pod as `GH_PUSH_TOKEN={{ RUNPOD_SECRET_github_push_token }}`. Without it, the pod falls back to waiting `READ_WAIT` seconds for its log to be read. |
| `wait_for_branch.sh` | Claude's side: run in the background, it checks GitHub every minute and exits when `pod-results/NAME` appears. Its exit wakes the session, which then pulls the results and terminates the stopped pod. |

Why: a pod cannot reach the Claude session, and the session cannot reach Runpod's servers, but both can reach GitHub. Before this, the GPU sat idle while waiting for a late check-in (pilot 7: about 80 minutes), or the pod stopped itself and its log was lost (pilot 8).
