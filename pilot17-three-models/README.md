# Pilot 17: do Llama and a bigger Qwen price outcomes the same way?

For Qwen3-14B, pilots 5-6 and 15-16 found a generic "which is bigger" comparator at the answer position, and a price that is partly relative to the alternative. Here the same tests run on Llama 3.1 8B Instruct and Qwen3-32B, with Qwen3-14B as the reference: does each give outcomes a fixed price, does a compact comparator decide, and is it the same machine that judges which pile has more apples? **Read [`DESIGN.md`](DESIGN.md) first.** Bars fixed in advance; no prediction gate.

| File | What it does |
|---|---|
| `pilot17.py` | Per model: every outcome and word deletion against 9 alternatives (9,810 menus); common shares, order flips, cycles, the readout inside; the comparator (direct contributions, top 20 chosen on half the values, swapped on the other half and on pilot 6's apples and weight pairs); the combined report (`--quick`: Qwen3-0.6B) |
| `run_pod.sh` | The pod's script (96 GB GPU, 100 GB volume; each model is deleted from disk after its run): Llama access check, quick test, the three models in turn; pushes the results to `pod-results/pilot17` and stops the pod |
