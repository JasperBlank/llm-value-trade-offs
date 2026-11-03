# Pilot 16: is the finished price absolute, or relative to the alternative?

Pilot 15 found that Qwen finishes an option's price at the line break after the second option, but it used one fixed alternative. Here the same 96 outcomes and their 994 word deletions are set against 8 different alternatives: does a word count the same whatever the alternative is, in the choice and in the finished price? **Read [`DESIGN.md`](DESIGN.md) first.** Results: [`results/real/README.md`](results/real/README.md) (run 3 Nov 2026). The bars were fixed in advance; Jasper's prediction is optional.

| File | What it does |
|---|---|
| `pilot16.py` | Every A against every B (8,720 menus); the choice's word effects and whole-outcome logits; a price readout at three positions and five layers; how much of each is common to all Bs; the report (`--quick`: Qwen3-0.6B on 6 values) |
| `run_pod.sh` | The pod's script: quick test, real run; pushes the results to `pod-results/pilot16` and stops the pod |
| `predictions/` | Claude's sealed predictions (hash), and Jasper's if he gives one |
