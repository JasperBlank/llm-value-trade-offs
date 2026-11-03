# Pilot 11: one direction for both?

Pilot 10 found a stable price for each sentence, and a direct "how good" rating that overlaps with it (r 0.81 over the full range, about 0.5 among outcomes of similar size). Is there a direction inside the model, read from the sentence itself, that carries this overlap, and does the model use it for both the choice and the rating? **Read [`DESIGN.md`](DESIGN.md) first.**

| File | What it does |
|---|---|
| `pilot11.py` | Reads the states, finds the vibe direction (held-out), pushes along it (`--quick`: Qwen3-0.6B on 24 sentences) |
| `probe11.py` | The readouts: difference-of-means direction, ridge, partial correlation, all leave-one-value-out |
| `score_predictions11.py` | Scores both prediction sets |
| `run_pod.sh` | The pod's script |
| `predictions/` | The questions, Claude's sealed predictions, Jasper's |
