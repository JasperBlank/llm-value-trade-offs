"""Print pilot 6's results to the pod's log (for Claude to read through the Runpod connection):
the report and the JSON files in compact form.
The learned single directions (the k = 1 runs, both layers) are printed as base64 (float16), so
they can be kept in the repo."""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

res = Path(sys.argv[1] if len(sys.argv) > 1 else "results/real")
print((res / "report.md").read_text())
for name in ["verdicts6.json", "control.json", "das.json", "directions6.json", "steer6.json"]:
    p = res / name
    if not p.exists():
        continue
    s = json.dumps(json.load(open(p)), separators=(",", ":"), default=float)
    print(f"PILOT6 FILE {name} ({len(s)} chars)")
    for i in range(0, len(s), 1000):
        print(s[i:i + 1000])
p = res / "das_subspaces.npz"
if p.exists():
    S = np.load(p)
    keep = {k: S[k].astype(np.float16) for k in S.files if k.endswith("_k1")}
    buf = io.BytesIO()
    np.savez_compressed(buf, **keep)
    s = base64.b64encode(buf.getvalue()).decode()
    print(f"PILOT6 NPZ k1_directions.npz ({len(s)} chars, float16)")
    for i in range(0, len(s), 1000):
        print(s[i:i + 1000])
print(f"PILOT6 FILE LIST: {sorted(x.name for x in res.iterdir())}")
