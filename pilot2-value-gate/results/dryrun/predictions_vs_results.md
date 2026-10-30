# Pilot 2: predictions against results

Results: dry run (fake model). Probabilities are scored with the Brier score (lower is better; always saying 50% scores 0.25). Ranges are 80% intervals.

## Probabilities

| prediction | outcome | Jasper | Brier | Claude | Brier |
|---|---|---|---|---|---|
| frame B selected | yes | 70% | 0.090 | 55% | 0.202 |
| log shape selected | yes | 60% | 0.160 | 60% | 0.160 |
| log and linear tied (Jasper only) | no | 20% | 0.040 | – | – |
| gate passes, 4b | no | 30% | 0.090 | 10% | 0.010 |
| gate passes, 8b | yes | 45% | 0.303 | 25% | 0.562 |
| gate passes, 14b | yes | 50% | 0.250 | 35% | 0.423 |
| 32B rule triggers | no | 15% | 0.022 | 35% | 0.122 |
| menu invariance passes | yes | 60% | 0.160 | 35% | 0.423 |
| description invariance passes | yes | 60% | 0.160 | 40% | 0.360 |
| closeness is uniform | yes | 50% | 0.250 | 45% | 0.303 |
| goal v1 (real good) rule-like | no | 80% | 0.640 | 35% | 0.122 |
| goal v2 (boxes) rule-like | yes | 50% | 0.250 | 50% | 0.250 |

| | Jasper | Claude |
|---|---|---|
| mean Brier score | 0.201 (12 predictions) | 0.267 (11 predictions) |

## Numbers

| prediction | outcome | Jasper: best (range) | in range | Claude: best (range) | in range |
|---|---|---|---|---|---|
| criterion 2, held-out r | 1.00 | 0.90 (0.50 to 0.96) | no | 0.85 (0.60 to 0.95) | no |
| criterion 3, transitivity factor (worse triplet) | 1.03 | 1.50 (1.20 to 4.00) | no | 1.60 (1.20 to 3.00) | no |
| real goods with a quantity slope (of 6) | 6.00 | 6.00 (3.00 to 6.00) | yes | 5.00 (3.00 to 6.00) | yes |
| loss aversion | 0.92 | 1.50 (0.80 to 3.00) | yes | 1.30 (0.80 to 2.00) | yes |
| closeness multiplier (family / strangers) | 1.28 | 1.00 (0.30 to 3.00) | yes | 1.30 (0.90 to 2.50) | yes |
| goal v1 value shift | 2.99 | – | – | 4.00 (2.00 to 20.00) | yes |
| goal v1 partner slope ratio | 1.00 | – | – | 0.60 (0.30 to 1.00) | yes |
| goal v2 welfare price (people per box) | above 8 / undefined | 8.00 (– to above 8 / undefined) | yes | 3.00 (1.00 to above 8 / undefined) | yes |

| | Jasper | Claude |
|---|---|---|
| outcomes inside the 80% range | 4 of 6 | 6 of 8 |

## Which criterion breaks first

Failures per criterion over the model-and-group combinations: 1a: 0, 1b: 0, 2: 2, 3: 2.

- Jasper: 2 → hit (shared with 1 other)
- Claude: 1a → miss
