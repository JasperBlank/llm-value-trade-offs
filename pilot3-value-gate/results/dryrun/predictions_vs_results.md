# Pilot 3: predictions against results

Results: dry run (fake model). Brier score: lower is better; always saying 50% scores 0.25. Ranges are 80% intervals. Unless stated, 14B, strangers.

## Probabilities

| prediction | outcome | Jasper | Brier | Claude | Brier |
|---|---|---|---|---|---|
| gate passes, 14b | yes | 60% | 0.160 | 50% | 0.250 |
| gate passes, 8b | no | 10% | 0.010 | 10% | 0.010 |
| gate passes, 32b | yes | 70% | 0.090 | 60% | 0.160 |
| menu invariance passes | yes | – | – | 40% | 0.360 |
| description invariance passes | yes | – | – | 50% | 0.250 |
| closeness is uniform | yes | – | – | 40% | 0.360 |
| goal v1 (real good) rule-like | yes | – | – | 80% | 0.040 |
| goal v1 override holds at 1 against 32 | yes | – | – | 60% | 0.160 |
| goal v2 (files) rule-like | no | – | – | 20% | 0.040 |

| | Jasper | Claude |
|---|---|---|
| mean Brier score | 0.087 (3) | 0.181 (9) |

## Numbers

| prediction | outcome | Jasper: best (range) | in range | Claude: best (range) | in range |
|---|---|---|---|---|---|
| criterion 2: median held-out rate miss | 1.02 | 1.30 (1.10 to 2.00) | no | 1.40 (1.10 to 2.50) | no |
| criterion 2: pairs within ×2 (of 15) | 15 | – | – | 11.00 (7.00 to 15.00) | yes |
| criterion 3: transitivity factor, worse triplet | 1.03 | – | – | 1.40 (1.10 to 3.00) | no |
| loss aversion | 0.96 | – | – | 1.30 (0.80 to 2.00) | yes |
| closeness multiplier | 1.50 | – | – | 1.80 (1.10 to 3.00) | yes |
| goal v2 welfare price (people per file) | 2.05 | – | – | 4.00 (1.00 to above 32 / infinite) | yes |

| | Jasper | Claude |
|---|---|---|
| outcomes inside the 80% range | 0 of 1 | 4 of 6 |

## Which criterion breaks first

Failures per criterion: 1a: 0, 1b: 0, 2: 2, 3: 2.

- Jasper: 3 → hit (shared with 1 other)
- Claude: 2 → hit (shared with 1 other)
