"""Prints pilot 10's raw results to the pod's log in compact lines, so they can be rebuilt if nothing is pushed
(rebuild_from_log10.py). Menu: logit(P(A)) in steps of 0.1, clipped to [-12.8, 12.7], two hex digits each.
Ratings: each digit's -log P in steps of 0.25 (two hex digits each, 10 per prompt), and the exact expected rating
and log-scale score."""
import json
import sys
from pathlib import Path

import numpy as np

res = Path(sys.argv[1] if len(sys.argv) > 1 else "results/real")
p = np.clip(np.load(res / "menu_pA.npy").astype(float), 1e-9, 1 - 1e-9)
v = np.clip(np.round(np.log(p / (1 - p)) * 10) + 128, 0, 255).astype(int)
for i in range(0, len(v), 2000):
    print(f"PILOT10 MENU {i} " + "".join(f"{x:02x}" for x in v[i:i + 2000]))
R = json.load(open(res / "ratings10.json"))
q = np.clip(np.round(-np.array(R["logp"]) * 4), 0, 255).astype(int)
for i in range(0, len(q), 192):
    print(f"PILOT10 LOGP {i} " + " ".join("".join(f"{x:02x}" for x in row) for row in q[i:i + 192]))
for k in ("rating", "score"):
    for i in range(0, len(R[k]), 256):
        print(f"PILOT10 {k.upper()} {i} " + json.dumps(R[k][i:i + 256]))
print("PILOT10 META " + json.dumps({k: R[k] for k in ("model", "minutes", "n_menu", "n_ratings", "letter_mass_min", "letter_mass_median")}))
