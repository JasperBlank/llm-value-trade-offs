# Pilot 2: predictions against results

Results: real run. Probabilities are scored with the Brier score (lower is better; always saying 50% scores 0.25). Ranges are 80% intervals.

## Probabilities

| prediction | outcome | Jasper | Brier | Claude | Brier |
|---|---|---|---|---|---|
| frame B selected | yes | 70% | 0.090 | 55% | 0.202 |
| log shape selected | yes | 60% | 0.160 | 60% | 0.160 |
| log and linear tied (Jasper only) | no | 20% | 0.040 | – | – |
| gate passes, 4b | no | 30% | 0.090 | 10% | 0.010 |
| gate passes, 8b | no | 45% | 0.203 | 25% | 0.062 |
| gate passes, 14b | no | 50% | 0.250 | 35% | 0.122 |
| 32B rule triggers | no | 15% | 0.022 | 35% | 0.122 |
| menu invariance passes | no | 60% | 0.360 | 35% | 0.122 |
| description invariance passes | no | 60% | 0.360 | 40% | 0.160 |
| closeness is uniform | no | 50% | 0.250 | 45% | 0.203 |
| goal v1 (real good) rule-like | yes | 80% | 0.040 | 35% | 0.423 |
| goal v2 (boxes) rule-like | no | 50% | 0.250 | 50% | 0.250 |

| | Jasper | Claude |
|---|---|---|
| mean Brier score | 0.176 (12 predictions) | 0.167 (11 predictions) |

## Numbers

| prediction | outcome | Jasper: best (range) | in range | Claude: best (range) | in range |
|---|---|---|---|---|---|
| criterion 2, held-out r | 0.75 | 0.90 (0.50 to 0.96) | yes | 0.85 (0.60 to 0.95) | yes |
| criterion 3, transitivity factor (worse triplet) | 1.58 | 1.50 (1.20 to 4.00) | yes | 1.60 (1.20 to 3.00) | yes |
| real goods with a quantity slope (of 6) | 6.00 | 6.00 (3.00 to 6.00) | yes | 5.00 (3.00 to 6.00) | yes |
| loss aversion | 1.58 | 1.50 (0.80 to 3.00) | yes | 1.30 (0.80 to 2.00) | yes |
| closeness multiplier (family / strangers) | 2.08 | 1.00 (0.30 to 3.00) | yes | 1.30 (0.90 to 2.50) | yes |
| goal v1 value shift | above 8 / undefined | – | – | 4.00 (2.00 to 20.00) | no |
| goal v1 partner slope ratio | 0.00 | – | – | 0.60 (0.30 to 1.00) | no |
| goal v2 welfare price (people per box) | 4.40 | 8.00 (– to above 8 / undefined) | yes | 3.00 (1.00 to above 8 / undefined) | yes |

| | Jasper | Claude |
|---|---|---|
| outcomes inside the 80% range | 6 of 6 | 6 of 8 |

## Which criterion breaks first

Failures per criterion over the model-and-group combinations: 1a: 0, 1b: 4, 2: 6, 3: 6.

- Jasper: 2 → hit (shared with 1 other)
- Claude: 1a → miss
