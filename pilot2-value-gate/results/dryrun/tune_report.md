# Tune phase: format selection

Tune models: 8b, 14b. Decided 2026-10-30T00:24+00:00 by the pre-registered rule.

| frame | probability on A/B (8b) | probability on A/B (14b) |
|---|---|---|
| A | 97.0% | 97.0% |
| B | 99.5% | 99.5% |

| frame and shape | held-out r | transitivity factor |
|---|---|---|
| A, log | 0.969 | 1.04 |
| A, linear | 0.932 | 1.21 |
| B, log | 0.995 | 1.01 |
| B, linear | 0.960 | 1.18 |

**Chosen: frame B, log value function.**
Widened grids for test pairs (valence, domains): none