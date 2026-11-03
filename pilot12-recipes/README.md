# Pilot 12: one vibe, or shared ingredients with different recipes?

Price and stated rating overlap strongly (r 0.81), but pilot 11's single direction could not show whether they share one variable. Here: can the place where the rating disagrees with the price be read from the sentence itself, and does pushing along that disagreement move the rating beyond what it moves the choice? **Read [`DESIGN.md`](DESIGN.md) first.**

| File | What it does |
|---|---|
| `pilot12.py` | Reads the states (pilot 11's runner), Q1 with a permutation test, the PLS readouts, the four pushes (`--quick`: Qwen3-0.6B) |
| `probe12.py` | The disagreement, ridge with precomputed maps (fast permutations), PLS, the bootstrap of the excess |
| `score_predictions12.py` | Scores both prediction sets |
| `run_pod.sh` | The pod's script |
| `predictions/` | The questions, Claude's sealed predictions, Jasper's |
