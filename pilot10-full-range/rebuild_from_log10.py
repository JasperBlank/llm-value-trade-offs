"""Rebuilds results/real/menu_pA.npy and ratings10.json from a pod log printed by print_raw10.py.
  python rebuild_from_log10.py LOGFILE [results/real]
The menu is rebuilt at 0.1-logit resolution and the digit log-probabilities at 0.25 nats; the expected ratings
and log-scale scores are exact."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
log = open(sys.argv[1]).read().splitlines()
out = HERE / (sys.argv[2] if len(sys.argv) > 2 else "results/real")
out.mkdir(parents=True, exist_ok=True)
menu, logp, vals, meta = {}, {}, {"rating": {}, "score": {}}, None
for line in log:
    parts = line.split(" ", 3)
    if line.startswith("PILOT10 MENU "):
        menu[int(parts[2])] = [int(parts[3][j:j + 2], 16) for j in range(0, len(parts[3]), 2)]
    elif line.startswith("PILOT10 LOGP "):
        logp[int(parts[2])] = [[int(t[j:j + 2], 16) for j in range(0, 20, 2)] for t in parts[3].split(" ")]
    elif line.startswith("PILOT10 RATING ") or line.startswith("PILOT10 SCORE "):
        vals[parts[1].lower()][int(parts[2])] = json.loads(parts[3])
    elif line.startswith("PILOT10 META "):
        meta = json.loads(line[len("PILOT10 META "):])
v = np.array([x for k in sorted(menu) for x in menu[k]], dtype=float)
np.save(out / "menu_pA.npy", (1 / (1 + np.exp(-(v - 128) / 10))).astype(np.float32))
rows = [json.loads(l) for l in open(HERE / "ratings10.jsonl")]
LP = -np.array([x for k in sorted(logp) for x in logp[k]], dtype=float) / 4
R = dict(meta or {}, rating=[x for k in sorted(vals["rating"]) for x in vals["rating"][k]],
         score=[x for k in sorted(vals["score"]) for x in vals["score"][k]], logp=LP.tolist(),
         keys=[[r["s"], r["n"], r["wording"]] for r in rows],
         note="Rebuilt from the pod's log: menu at 0.1-logit resolution, digit log-probabilities at 0.25 nats.")
assert len(v) == R.get("n_menu", len(v)) and len(R["rating"]) == len(rows) == len(LP), (len(v), len(R["rating"]), len(LP))
json.dump(R, open(out / "ratings10.json", "w"))
print(f"rebuilt {len(v)} trade-offs and {len(rows)} ratings into {out}")
