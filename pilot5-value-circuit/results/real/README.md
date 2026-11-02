# Pilot 5 results (Qwen3-14B, RTX PRO 6000, Runpod US-NE-1, 2 Nov 2026)

Main run 09:23-09:36 UTC (code at commit 6b00e3e), outputs hidden on the pod until both prediction
sets were committed (314b636). The pod's GPU was then taken by another customer; it was started
without a GPU at ~10:00 UTC and printed the stored results to its log (`log_pod_print.txt`), from
which these JSON files and `report.md` were rebuilt exactly (no numbers retyped). The figures and
the `.npz` arrays (directions, per-prompt shifts) are still on the stopped pod's disk.

The formats extension (DESIGN.md, section 8) has not run yet: it needs a GPU.
`predictions_vs_results.md`: both prediction sets scored (`score_predictions.py`).
