# Pilot 9: the vibe check

Pilot 8 found that Qwen3-14B prices each sentence, not each value. If the model does a per-sentence "vibe check" that then feeds the comparison circuit, asking it directly "How good would it be if [sentence]?" should give ratings that track the menu prices, including the quirks of the exact wording. **Read [`DESIGN.md`](DESIGN.md) first.** Results: [`results/real/README.md`](results/real/README.md) (run 3 Nov 2026).

| File | What it does |
|---|---|
| `build_prompts.py` | Builds `ratings9.jsonl`: pilot 8's 48 sentences at 2, 4 and 8 people, asked four ways (576 prompts) |
| `pilot9.py` | The GPU part: the probability of each digit 0-9 as the answer, and the expected rating (`--quick`: Qwen3-0.6B) |
| `analyze9.py` | The pre-registered tests T1-T3 and the rest of the report, on CPU, against pilot 8's per-sentence prices |
| `score_predictions9.py` | Scores both prediction sets |
| `run_pod.sh` | The pod's script: install, quick test, real run, analysis, then push the results (or print them) and stop |
| `pilot9_logscale.py`, `explore_logscale.py`, `run_pod_logscale.sh` | Follow-up (exploratory): the same prompts read on a log scale, so the answers piled up at 7 spread out. Folded into pilot 10 (its test F2), which reruns these prompts; these files were not run |
| `predictions/` | The questions, Claude's sealed predictions, Jasper's |
