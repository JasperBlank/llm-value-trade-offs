# Pilot 15: where the price is finished, and how it is compared

Pilot 14 found that single words move the choice by their meaning, but the price signal on the sentence's own words at layer 20 barely follows them. Here: does Qwen value each option on its own before comparing, does an option's worth funnel through one spot on its way to the answer, and is the comparison the generic "which is bigger" machine of pilots 5-6? Also pilot 14's Q4, redone with a working control. **Read [`DESIGN.md`](DESIGN.md) first.** No probability predictions for this pilot; the bars were fixed in advance.

| File | What it does |
|---|---|
| `pilot15.py` | The readout map (six positions, every 4th layer; trained on the intact sentences, tested on pilot 14's word deletions), the route (single-position swaps), the comparator (pilot 5's top-20 pieces), pilot 14's Q4 redone, the report (`--quick`: Qwen3-0.6B on 6 values) |
| `run_pod.sh` | The pod's script: quick test, real run; pushes the results to `pod-results/pilot15` and stops the pod |
| `predictions/` | The questions as first written, and Claude's predictions sealed before the prediction step was dropped |
