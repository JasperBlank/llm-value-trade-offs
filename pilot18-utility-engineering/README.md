# Pilot 18: Utility Engineering on Llama 3.1 8B, and is the utility used?

A replication of Utility Engineering (Mazeika et al. 2025) on Llama-3.1-8B-Instruct, with their code and outcomes:
- their consistency numbers;
- their linear readout of the utilities;
- one new test: push along the readout and see whether the choice moves, and patch pilot 17's decision writers (the
  parts that also compare piles of apples) on their menus.

**Read [`DESIGN.md`](DESIGN.md) first.** Results: pending.

| File | What it does |
|---|---|
| `DESIGN.md` | The question, the material, and the bars, fixed before the run |
| `run_pod.sh` | The pod's script: their environment and code, then ours |
| `ue_setup.sh` | Their pinned environment (vLLM 0.6.6.post1, torch 2.5.1, transformers 4.46.2) in a venv |
| `fetch_theirs.py` | The model at their revision, their released results (compact), the paper's figure pages |
| `pilot18.py` | Steps 1-3 on our side: the comparisons, the pipeline check against vLLM, the probe, the pushes, the patching (`--quick`: synthetic utilities) |
