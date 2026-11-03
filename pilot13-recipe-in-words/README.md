# Pilot 13: the choice recipe in words

Pilots 10-12 showed that the model writes a price onto each outcome as it reads it, and that the choice partly uses that price. That price is a weighting of 5,120 numbers. Here each outcome is described on ten named features, as the model itself answers them, to ask *why* in words: do the features explain the price, how do the choice and the stated rating weigh them, and does the recipe price new outcomes? Also: does pilot 12's disagreement push beat 20 random directions? **Read [`DESIGN.md`](DESIGN.md) first.**

| File | What it does |
|---|---|
| `build13.py` | The ten feature questions and the 24 new sentences; writes `sentences13.json` |
| `pilot13.py` | The features, the new sentences' ratings and prices (13,824 new trade-offs), the states, the recipes (P1, P2, P4), the pushes at layer 20 (P3 and the feature pushes), the report (`--quick`: Qwen3-0.6B on 6 values) |
| `score_predictions13.py` | Scores both prediction sets |
| `run_pod.sh` | The pod's script: quick test, real run, scoring; pushes the results to `pod-results/pilot13` and stops the pod |
| `predictions/` | The questions, Claude's sealed predictions, Jasper's |
