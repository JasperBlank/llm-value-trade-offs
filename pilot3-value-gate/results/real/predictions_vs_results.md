# Pilot 3: predictions against results

Results: real run. Brier score: lower is better; always saying 50% scores 0.25. Ranges are 80% intervals. Unless stated, 14B, strangers.

## Probabilities

| prediction | outcome | Jasper | Brier | Claude | Brier |
|---|---|---|---|---|---|
| gate passes, 14b | no | 40% | 0.160 | 40% | 0.160 |
| gate passes, 8b | no | 10% | 0.010 | 10% | 0.010 |
| gate passes, 32b | – | 60% | – | 50% | – |
| menu invariance passes | no | 50% | 0.250 | 30% | 0.090 |
| description invariance passes | no | 60% | 0.360 | 55% | 0.303 |
| closeness is uniform | yes | 10% | 0.810 | 35% | 0.423 |
| goal v1 (real good) rule-like | yes | 95% | 0.003 | 85% | 0.023 |
| goal v1 override holds at 1 against 32 | yes | 80% | 0.040 | 65% | 0.122 |
| goal v1 override holds at 1 against 128 | yes | 50% | 0.250 | – | – |
| goal v2 (files) rule-like | no | 15% | 0.022 | 15% | 0.022 |

| | Jasper | Claude |
|---|---|---|
| mean Brier score | 0.212 (9) | 0.144 (8) |

## Numbers

| prediction | outcome | Jasper: best (range) | in range | Claude: best (range) | in range |
|---|---|---|---|---|---|
| criterion 2: median held-out rate miss | 1.38 | 1.50 (1.30 to 1.80) | yes | 1.35 (1.10 to 2.20) | yes |
| criterion 2: pairs within ×2 (of 15) | 13 | 10.00 (7.00 to 13.00) | yes | 11.00 (7.00 to 15.00) | yes |
| criterion 3: transitivity factor, worse triplet | 1.88 | 1.40 (1.20 to 1.60) | no | 1.35 (1.10 to 2.50) | yes |
| loss aversion | 0.54 | 1.58 (1.18 to 1.98) | no | 1.50 (0.80 to 3.00) | no |
| loss aversion, family | 0.25 | 2.45 (2.05 to 2.85) | no | – | – |
| closeness multiplier | 2.57 | 2.08 (1.68 to 2.48) | no | 2.00 (1.20 to 4.00) | yes |
| goal v2 welfare price (people per file) | above 128 / infinite | 4.40 (2.00 to 10.00) | no | 3.00 (0.70 to above 128 / infinite) | yes |

| | Jasper | Claude |
|---|---|---|
| outcomes inside the 80% range | 2 of 7 | 5 of 6 |

## Which criterion breaks first

Failures per criterion: 1a: 0, 1b: 1, 2: 3, 3: 4.

- Jasper: 3 → hit
- Claude: 2 → miss
