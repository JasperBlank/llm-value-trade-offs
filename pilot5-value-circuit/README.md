# Pilot 5: does Qwen3-14B use its value code, and what circuit writes and reads it?

Pilot 4 found a value code at the last position of the prompt (layers 24-40). Pilot 5 tests,
pre-registered, whether pushing along it moves the choice by the predicted amount, and explores
which attention heads and MLPs write it and which turn it into the answer. **Read
[`DESIGN.md`](DESIGN.md) first.**

| File | What it does |
|---|---|
| `make_prompts.py` | Builds `prompts.jsonl` (run once, here): pilot 4's prompts, 1,440 fresh prompts in three new wordings, and their donors |
| `pipeline.py` | Runs the stages in order with the model loaded once |
| `record.py` | Last-position residual stream of every prompt, every layer |
| `directions.py` | A's and B's value directions, random and shuffled-values controls |
| `steer.py` | **The pre-registered test**: pushes at the last position |
| `circuit.py`, `circuit_analyze.py` | Who writes the value directions and the answer (exact decomposition) |
| `patch.py` | Donor-to-recipient patching at the last position |
| `readers.py` | Do the answer-writing heads look at the "A" label when A's value is pushed? |
| `objective.py` | Pilot 4's goal-module question, read with the last-position code |
| `analyze.py` | Verdicts and `report.md` |
| `tests/` | Mechanics checks on a small model; a tiny-model configuration |
| `predictions/` | Sealed predictions |
| `pod.ipynb` | One cell for the pod; it beeps when done |
| `run_pod.sh` | The same without a notebook, as the pod's start command |

## On the pod

**Through the Runpod connection (2 Nov):** Claude creates the pod with a start command that runs
`run_pod.sh`: install, the quick test, then it waits until both prediction sets are committed,
runs everything, packs `/workspace/pilot5_results.tar.gz` and prints the report to the pod's log.
Claude reads the log and terminates the pod. **By hand (as in pilot 4):**

1. Deploy a **48 GB GPU** pod (L40S or RTX A6000), Secure Cloud, in **US-TX-3** with the network
   volume attached (it mounts at `/workspace`), PyTorch template.
2. Open JupyterLab, File → New → Notebook is not needed: open
   `/workspace/llm-value-trade-offs/pilot5-value-circuit/pod.ipynb` (if the folder is missing, first
   run `cd /workspace && git clone https://github.com/JasperBlank/llm-value-trade-offs.git` in a terminal).
3. Run the cell (about 60-75 minutes). At the three rising beeps, download
   `/workspace/pilot5_results.tar.gz` and send it to Claude.
4. Terminate the pod.
