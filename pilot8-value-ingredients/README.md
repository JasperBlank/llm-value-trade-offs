# Pilot 8: why are the relative values what they are? Worth by domain and size

24 outcomes (money, free time and health, each at 8 sizes) are compared in every pair, at three
numbers of people and in four wordings, in Qwen3-14B. The fit gives each outcome's worth; curves over
log(size) per domain say how worth grows with size, whether the sensitivity to size is the same
across domains, and what the model's price list is (an hour of free time, a day of good health, in
euros). Inside the model, minimal pairs show where size and domain contributions sit, and a worth
direction on the outcome's own words tests whether the worth is already assembled there. **Read [`DESIGN.md`](DESIGN.md) first.**

| File | What it does |
|---|---|
| `make_prompts.py` | Builds `outcomes.json`, `behaviour.jsonl` (6,624 prompts) and `pairs.jsonl` (minimal pairs, outcome swaps) |
| `pilot8.py` | Stages: behave (fit, curves, price list; W1, W3), map (size and domain maps), das (worth direction, W2), report; reuses pilot 7's sites, patching and alignment search |
| `config.yaml` | Settings; `prereg` holds the pre-registered thresholds; `quick` is the Qwen3-0.6B test |
| `tests/test_mechanics8.py` | The fit recovers known worths, b and bias; the regression recovers known curves; a run's files and top-layer check |
| `run_pod.sh` | The pod's script: install, quick test, the real run, print, then the pod stops itself |
| `print_results8.py` | Prints the report and the result files to the pod's log |
| `predictions/` | The questions, Claude's sealed predictions, Jasper's |
