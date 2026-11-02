# Pilot 5, formats extension (DESIGN.md, section 8)

Run on 2 Nov 2026, 10:12-10:23 UTC, on a fresh A100 pod (Runpod US-WA-1), code at commit 082beaf,
after both prediction sets (including F1-F6) were committed. The first pod's disk was gone, so the
value directions were fitted again from a new recording (same code, same prompts); they reproduce
the first run (option-level r 0.92 at layer 28, cos(w_A, w_B) 0.008). The A/B slope and the A/B
top-20 writers come from the first run (`../real/`).

`report_formats.md` is copied verbatim from the pod's log; `formats_summary.json` holds the key
numbers of the printed `formats.json`, copied digit for digit (the per-option estimates are left
out). The pod was terminated afterwards.
