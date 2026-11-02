# Pilot 7: where does the model form the values it puts through the decision stage?

Pilot 6 found that the answer position receives the *outcome* of a comparison, from machinery shared
with non-value comparisons, so the values must be formed upstream. Pilot 7 follows option A's good
and number of people from A's own line to the answer, in Qwen3-14B and Llama 3.1 8B, and runs a
"monkey test": is there one number for option A that combines *which* good with *how many* people,
at the model's own exchange rate, and is it formed before the model has read option B? **Read
[`DESIGN.md`](DESIGN.md) first**; the pre-registered rule and criteria are in its section 4.

| File | What it does |
|---|---|
| `make_prompts.py` | Builds `pairs.jsonl` (run once, here): A-good, A-number and B-number swaps, apples and weight |
| `pilot7.py` | All stages, per model: fit (the model's own values), map (site x layer patching), das (the monkey test), report |
| `config.yaml` | Settings; `prereg` holds the pre-registered thresholds; `quick` is the Qwen3-0.6B test |
| `tests/test_mechanics7.py` | Checks: sites well formed and aligned, top-layer swap gives the donor's answer, empty swap changes nothing, whole-space subspace swap equals the site swap, sites before B's line carry nothing of a B swap |
| `tests/config_tiny.yaml` | The same pipeline on a tiny random Qwen3 (used here, before the pod) |
| `run_pod.sh` | The pod's script: install, Llama access check, quick test, the real run (results hidden), print once both prediction sets are committed |
| `print_results7.py` | Prints the report, the JSON files and the learned directions to the pod's log |
| `predictions/` | The questions, Claude's sealed predictions, Jasper's |
| `values.json` | Pilot 3's fitted values for Qwen3-14B (for reference; each model's own values are fitted in the run) |

On the pod: Claude creates it through the Runpod connection with `HF_TOKEN` taken from the Runpod
secret `huggingface_token` (Llama is gated) and a start command that clones the repo and runs
`run_pod.sh`.
