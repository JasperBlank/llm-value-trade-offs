# Pilot 2: the value gate

Does a language model trade goods off consistently, the way monkeys trade juices in
Padoa-Schioppa's experiments? Qwen3-4B, 8B and 14B choose between two outcomes ("3 elderly
people each get a flu vaccination" or "2 pensioners are prevented from losing €100 to a
phone scam"), with reasoning off, and the answer is read as exact probabilities from the
first answer token. If one value per good explains the choices (quantity sensitivity,
held-out prediction, transitivity), the model passes the gate and the next step is to look
inside the network for the value code. A goal module asks whether an assigned objective
enters that valuation as extra value or overrides it like a rule.

**Read [`DESIGN.md`](DESIGN.md) first.** The predictions are in [`predictions/`](predictions/).

## What's in the box

| File | What it does |
|---|---|
| `DESIGN.md` | The whole design: question, prompt, goods, gate criteria, goal module, models, order of work |
| `goods.yaml` | The 18 goods (tune set and test set) and their wordings |
| `value_goods_split.json` | The seeded coin flip that split the goods into tune and test sets |
| `prompts.yaml` | Every other piece of prompt text |
| `config.yaml` | Models, grids, and the pre-registered thresholds |
| `common.py` | Prompt building, the list of prompts per phase, model loading, reading the letter probabilities, a fake model for dry runs |
| `fit.py` | The utility model (log or linear), held-out correlation, exchange rates, quantity slopes, transitivity |
| `run.py` | Runs one phase (`tune`, `test` or `goal`) on one model |
| `select_format.py` | After the tune phase: picks frame and value-function shape by the pre-registered rule |
| `analyze.py` | The gate, the extras, closeness, the goal module, the 32B rule; one report per model plus a summary |
| `pod.ipynb` | The notebook for the GPU pod: three cells that beep when done |
| `predictions/` | Jasper's predictions and Claude's (sealed until Jasper's are in; until then only their hash, in `SEALED.md`) |
| `results/dryrun/` | Reports from the dry run: the analysis finding the fake model's known values again |

## Test it without a GPU (a dry run)

A fake model with known values stands in for the real one; the analysis should find those
values again (it does: see `results/dryrun/report_14b.md`).

```bash
pip install -r requirements.txt
for m in 8b 14b; do python run.py --phase tune --model $m --dry-run; done
python select_format.py --dry-run
for m in 4b 8b 14b; do python run.py --phase test --model $m --dry-run; python run.py --phase goal --model $m --dry-run; done
python analyze.py --dry-run
```

## Running it on a pod

**Pod:** RunPod, one **RTX A6000** (48 GB), PyTorch template, **volume disk 80 GB** (the
three models take about 55 GB), container disk 20 GB. About $0.59 an hour; the whole run
takes about 1.5 to 2 hours, so about $1 to $1.50. An HF_TOKEN is optional (faster downloads).

1. Open JupyterLab. File → New → Terminal, and paste:
   ```bash
   cd /workspace && git clone https://github.com/JasperBlank/llm-value-trade-offs.git
   ```
2. In the file browser, open `llm-value-trade-offs/pilot2-value-gate/pod.ipynb`.
3. **Run cell 1** (setup, a quick check on a small model, the tune phase on 8B and 14B, and
   the format choice). About 45 to 60 minutes. At the three beeps, download
   `/workspace/pilot2_lock.tar.gz` and send it to Claude, who commits it to GitHub. That
   commit is the lock: the format is public before any test good is shown to a model.
   Leave the pod running.
4. **Run cell 2** once Claude says the lock is on GitHub. The cell checks that the format on
   GitHub matches this pod's before it starts. Test phase and goal module on 4B, 8B and 14B,
   then the analysis, without showing results. About 30 to 45 minutes. At the beeps,
   download `/workspace/pilot2_results.tar.gz` and send it to Claude. The last line says
   whether a 32B run is needed; if not, terminate the pod.
5. **Cell 3, only if cell 2 says so:** a new pod with an A100 80 GB (about $1.79 an hour) and
   a volume disk of 100 GB; repeat steps 1 and 2 there and run cell 3 only.

Two low beeps mean a step failed: scroll up for the error. Every step resumes where it left
off, so after fixing the cause, run the same cell again.

## Commits, in order

0. **Pre-lock:** design and code, tested; Claude's sealed predictions only as a SHA-256
   hash (`predictions/SEALED.md`), so they cannot influence Jasper's.
1. **The lock:** both prediction sets in full, before the pod starts.
2. **The format:** `results/real/locked_format.json` and `tune_report.md`, before the test phase.
3. **The results**, after Jasper has written down his expectations.
