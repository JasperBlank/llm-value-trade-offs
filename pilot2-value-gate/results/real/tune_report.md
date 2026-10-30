# Tune phase: format selection

Tune models: 8b, 14b. Decided 2026-10-30T12:53+00:00 by the pre-registered rule.

| frame | probability on A/B (8b) | probability on A/B (14b) |
|---|---|---|
| A | 100.0% | 100.0% |
| B | 100.0% | 100.0% |

| frame and shape | held-out r | transitivity factor |
|---|---|---|
| A, log | 0.706 | 27257.35 |
| A, linear | 0.544 | 3.73 |
| B, log | 0.771 | 8.44 |
| B, linear | 0.657 | 1.46 |

**Chosen: frame B, log value function.**
Widened grids for test pairs (valence, domains): [('harm_prevented', 'health', 'money'), ('harm_prevented', 'health', 'time')]