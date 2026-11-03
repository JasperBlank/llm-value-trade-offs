# Pilot 8: why are the relative values what they are? The ingredients of worth

48 outcomes built from five crossed ingredients (domain, gain vs prevented loss, size, children vs
adults, poverty) are compared in every pair, at three numbers of people and in two wordings, in
Qwen3-14B. The fit gives each outcome's worth in the model's own exchange rate; a regression says how
much each ingredient adds and how much of the differences the ingredients explain. Inside the model,
minimal pairs show where each ingredient's contribution sits, and a worth direction on the outcome's
own words tests whether the worth is already assembled there. **Read [`DESIGN.md`](DESIGN.md) first.**

| File | What it does |
|---|---|
| `make_prompts.py` | Builds `outcomes.json`, `behaviour.jsonl` (13,536 prompts) and `pairs.jsonl` (minimal pairs, outcome swaps) |
| `pilot8.py` | Stages: behave (fit, ingredient effects, W1), map (ingredient maps), das (worth direction, W2), report; reuses pilot 7's sites, patching and alignment search |
| `config.yaml` | Settings; `prereg` holds the pre-registered thresholds; `quick` is the Qwen3-0.6B test |
| `tests/test_mechanics8.py` | The fit recovers known worths, b and bias; the regression recovers known effects; a run's files and top-layer check |
| `run_pod.sh` | The pod's script: install, quick test, the real run, print, then the pod stops itself |
| `print_results8.py` | Prints the report and the result files to the pod's log |
| `predictions/` | The questions, Claude's sealed predictions, Jasper's |
