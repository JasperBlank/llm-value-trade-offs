# Pilot 3: the value gate with fair measurement

Pilot 2 asked whether Qwen3 trades goods off consistently on its immediate answers; no model
passed, partly because of how the choices were measured. Pilot 3 asks again on fresh goods,
with every comparison averaged over both option orders and six wordings, comparisons placed
around each pair's balance point, and criterion 2 judged on exchange rates. Qwen3-14B is the
main model, 8B the negative control, 32B the larger model. **Read [`DESIGN.md`](DESIGN.md) first.**

## What's in the box

| File | What it does |
|---|---|
| `DESIGN.md` | The whole design and why it differs from pilot 2 |
| `goods.yaml` | The 9 fresh test goods and their wordings |
| `prompts.yaml` | The six wordings and every other piece of prompt text |
| `config.yaml` | Models, quantities, and the pre-registered thresholds |
| `common.py` | Prompts, the stage-1 and stage-2 prompt lists, the goal module, model loading, a fake model for dry runs |
| `fit.py` | Cells (averaged over orders and wordings), the utility model, exchange rates, slopes, transitivity |
| `run.py` | One model: scouting, balance points, the test, the goal module |
| `analyze.py` | The gate, the extras and the goal module per model, plus a summary |
| `score_predictions.py` | Scores both prediction sets |
| `pod.ipynb` | Two notebook cells for the GPU pods; they beep when done |
| `predictions/` | Jasper's and Claude's predictions |
| `results/dryrun/` | The dry run: the analysis finding a fake model's known values again |

## Dry run (no GPU)

```bash
pip install -r requirements.txt
for m in 14b 8b 32b; do python run.py --model $m --dry-run; done
python analyze.py --dry-run
```

## On the pods

1. **Pod 1:** RunPod RTX A6000, PyTorch template, volume disk 80 GB. In JupyterLab: File →
   New → Terminal, then `cd /workspace && git clone https://github.com/JasperBlank/llm-value-trade-offs.git`.
   Open `llm-value-trade-offs/pilot3-value-gate/pod.ipynb` and run **cell 1** (about 30
   minutes). At the three beeps, download `/workspace/pilot3_results_a6000.tar.gz`, send it to
   Claude, and terminate the pod.
2. **Pod 2:** an A100 80 GB with a 100 GB volume disk. Same clone step, then **cell 2** only
   (about 30–40 minutes). Download `/workspace/pilot3_results_32b.tar.gz`, send it, terminate.

Results are never printed on the pod. Two low beeps mean a step failed (scroll up); every
step resumes where it left off, so run the same cell again after fixing the cause.
