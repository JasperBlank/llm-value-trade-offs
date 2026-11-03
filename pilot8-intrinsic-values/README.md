# Pilot 8: trade-offs between intrinsic values

Jasper's four questions: how Qwen3-14B trades 24 intrinsic values off against each other (Clearer
Thinking's 22 categories plus Non-suffering and Truth); whether the same internal thing lights up
across different trade-offs; whether the menu is consistent; and whether the value the model reports
when asked about one value on its own is the same internal quantity it uses before comparing.
**Read [`DESIGN.md`](DESIGN.md) first.** Results: [`results/real/README.md`](results/real/README.md) (run 3 Nov 2026).

| File | What it does |
|---|---|
| `make_prompts.py` | Builds `values.json` and the prompt files (trade-offs, menus, ratings, swap pairs, rating pairs) |
| `pilot8.py` | Stages: behave (prices, menu checks, ratings), source (worth direction), compare (comparison direction), rating (the direction in the rating prompts), report |
| `config.yaml` | Settings; `prereg` holds the pre-registered thresholds; `quick` is the Qwen3-0.6B test |
| `tests/test_mechanics8.py` | The fit recovers known worths; the cycle count; rating-prompt sites; a run's files |
| `run_pod.sh` | The pod's script: install, quick test, real run, print, then the pod stops itself |
| `print_results8.py` | Prints the report and the result files to the pod's log |
| `predictions/` | The questions, Claude's sealed predictions, Jasper's |
| `score_predictions.py` | Scores both prediction sets against the results |
| `explore_sentences.py` | Exploratory, after the results: a price for each sentence instead of each value |
