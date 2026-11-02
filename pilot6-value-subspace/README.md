# Pilot 6: the direction Qwen3-14B actually uses, and is it a value circuit at all?

Pilot 5 found that the probe for option A's value is a weak lever (7% of the expected effect), while
swapping the whole last-position state, or just 20 pieces in blocks 24-27, carries most of a good's
effect. Pilot 6 searches for the direction the model actually uses (distributed alignment search)
and tests a rival hypothesis: that blocks 24-27 are a generic "which option has more" comparator,
using two comparisons with no value in them. **Read [`DESIGN.md`](DESIGN.md) first**; the
pre-registered criteria P1-P4 are in its section 4.

| File | What it does |
|---|---|
| `make_prompts.py` | Builds `pairs.jsonl` (run once, here): 4,797 recipient/donor pairs (value train and held-out, quantity, weight) |
| `pilot6.py` | All stages: control, record, das, directions, steer, report (reuses pilot 5's `common.py`) |
| `config.yaml` | Settings; `prereg` holds the pre-registered thresholds; `quick` is the Qwen3-0.6B test |
| `tests/test_mechanics6.py` | Checks: the swap of the whole space equals the whole-state swap, an empty swap changes nothing, the top-layer swap gives the donor's answer, learned subspaces are orthonormal |
| `tests/config_tiny.yaml` | The same pipeline on a tiny random Qwen3 (used here, before the pod) |
| `run_pod.sh` | The pod's script: install, quick test, the real run (results hidden), print once both prediction sets are committed |
| `print_results6.py` | Prints the report, the JSON files and the learned k = 1 directions to the pod's log |
| `predictions/` | The questions, Claude's sealed predictions, Jasper's |
| `values.json` | Pilot 3's fitted values (V = u + 1.46 log n) |

## On the pod

Claude creates the pod through the Runpod connection with a start command that clones the repo into
`/workspace` and runs `run_pod.sh`. The run starts at once (no idle GPU); its log shows progress
lines only. Once both prediction sets are in the repo, the pod prints the results, Claude rebuilds
the files from the log into `results/real/` and terminates the pod.

By hand: on any 80 GB GPU pod with the PyTorch template,
`cd /workspace && git clone https://github.com/JasperBlank/llm-value-trade-offs.git && bash llm-value-trade-offs/pilot6-value-subspace/run_pod.sh`.
