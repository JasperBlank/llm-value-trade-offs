"""Print pilot 7's results to the pod's log (for Claude to read through the Runpod connection):
the report, every JSON file in compact form, and the learned directions as base64 (float16)."""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

res = Path(sys.argv[1] if len(sys.argv) > 1 else "results/real")
print((res / "report.md").read_text())


def dump(label, s):
    print(f"PILOT7 {label} ({len(s)} chars)")
    for i in range(0, len(s), 1000):
        print(s[i:i + 1000])


for p in sorted(res.rglob("*.json")):
    dump(f"FILE {p.relative_to(res)}", json.dumps(json.load(open(p)), separators=(",", ":"), default=float))
for p in sorted(res.rglob("directions.npz")):
    Z = np.load(p)
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: Z[k].astype(np.float16) for k in Z.files})
    dump(f"NPZ {p.relative_to(res)}", base64.b64encode(buf.getvalue()).decode())
print(f"PILOT7 FILE LIST: {sorted(str(x.relative_to(res)) for x in res.rglob('*') if x.is_file())}")
