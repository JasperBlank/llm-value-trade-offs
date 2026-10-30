# LLM value trade-offs

Do language models trade values off against each other consistently, like the monkeys in
Padoa-Schioppa's juice-choice experiments? And if they do, where in the network does the
trade-off happen? Research by Jasper Blank; code written with Claude.

| Folder | Question | Status |
|---|---|---|
| [`pilot1-goal-sources/`](pilot1-goal-sources/) | When an LLM agent faces a trade-off between an organisation's target and a third party's wellbeing, what decides its choice: the goal in its prompt, its trained persona, or whether it is observed? | Run on 29 Oct 2026, Qwen3-8B. Results in `results/qwen3-8b/` |
| [`pilot2-value-gate/`](pilot2-value-gate/) | Does the model hold consistent values (quantity sensitivity, one common scale, transitivity), judged on its immediate answers? And does an assigned objective enter that valuation as extra value or override it like a rule? | Pre-registered; not yet run |

## How pre-registration works here

For pilot 2, the design, the analysis code, the thresholds and both sets of predictions
(Jasper's and Claude's, Claude's sealed first) are committed before any model sees a
prompt. The format chosen in the tune phase is committed before the test phase. Results come
last. The commit history is the record.
