# Pilot 14: what writes the price

Pilots 10-13 found that Qwen writes a price onto each outcome as it reads it, and that the choice partly uses it. Here: which words of a sentence set its price, which attention heads and MLP layers write it onto the sentence, and whether the choice goes blind to the price when those parts are silenced. **Read [`DESIGN.md`](DESIGN.md) first.**

| File | What it does |
|---|---|
| `pilot14.py` | The price direction at layer 20; the exact split of its readout into the embedding and every head and MLP of blocks 1-20; word deletions; ablation of the top parts against matched random sets; the block-by-block map; what the top heads read; the report (`--quick`: Qwen3-0.6B on 6 values) |
| `score_predictions14.py` | Scores both prediction sets |
| `run_pod.sh` | The pod's script: quick test, real run, scoring; pushes the results to `pod-results/pilot14` and stops the pod |
| `predictions/` | The questions, Claude's sealed predictions, Jasper's |
