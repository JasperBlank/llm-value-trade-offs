"""Print pilot 8's results to the pod's log: the report, the JSON files, the worth direction (base64)."""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

res = Path(sys.argv[1] if len(sys.argv) > 1 else "results/real")
print((res / "report.md").read_text())


def dump(label, s):
    print(f"PILOT8 {label} ({len(s)} chars)")
    for i in range(0, len(s), 1000):
        print(s[i:i + 1000])


for p in sorted(res.rglob("*.json")):
    dump(f"FILE {p.relative_to(res)}", json.dumps(json.load(open(p)), separators=(",", ":"), default=float))
for p in sorted(res.rglob("*.npy")):
    buf = io.BytesIO()
    np.save(buf, np.load(p).astype(np.float16))
    dump(f"NPY {p.relative_to(res)}", base64.b64encode(buf.getvalue()).decode())
