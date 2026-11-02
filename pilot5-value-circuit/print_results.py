"""Print pilot 5's results to the pod's log (for Claude to read through the Runpod connection):
the report, then the main JSON files in compact form (long per-prompt lists left out)."""
import json
import sys
from pathlib import Path

res = Path(sys.argv[1] if len(sys.argv) > 1 else "results/real")
print((res / "report.md").read_text())
for name in ("verdicts.json", "steering.json", "directions.json", "formats.json", "patching.json", "readers.json",
             "objective.json", "record_check.json", "circuit_identity.json", "circuit.json"):
    p = res / name
    if not p.exists():
        continue
    d = json.load(open(p))
    if name == "steering.json":
        d.pop("prompt_ids", None)
    s = json.dumps(d, separators=(",", ":"), default=float)
    print(f"PILOT5 FILE {name} ({len(s)} chars)")
    for i in range(0, len(s), 1000):
        print(s[i:i + 1000])
