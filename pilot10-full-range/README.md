# Pilot 10: the full range of goodness

Pilot 9's ratings piled up at 7 because pilot 8's sentences are all moderately good outcomes. This pilot adds a small and a large version of each value (96 sentences), prices them all in one menu, and reads the ratings on a log scale too. **Read [`DESIGN.md`](DESIGN.md) first.** Results: [`results/real/README.md`](results/real/README.md) (run 3 Nov 2026).

| File | What it does |
|---|---|
| `build_prompts10.py` | The 96 sentences (`sentences10.json`), the 45,600 trade-offs (`menu_keys10.json`) and the 1,536 rating prompts (`ratings10.jsonl`) |
| `pilot10.py` | The GPU part: P(A) for every trade-off; the digits' log-probabilities for every rating prompt (`--quick`: Qwen3-0.6B) |
| `analyze10.py` | Prices, the pre-registered tests F1-F3 and the rest of the report, on CPU |
| `score_predictions10.py` | Scores both prediction sets |
| `print_raw10.py`, `rebuild_from_log10.py` | The raw results in compact log lines, and back, for when nothing is pushed |
| `run_pod.sh` | The pod's script |
| `predictions/` | The questions, Claude's sealed predictions, Jasper's |
