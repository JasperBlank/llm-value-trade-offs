# Pilot 2 results: 8b

Value function: log. Probability on the answer letters: 100.0% (prompts under 90%: 0.0%).

| criterion | strangers | family |
|---|---|---|
| 1a quantity sensitivity, all real goods | pass | pass |
| 1b real over neutral (mean probability) | 0.82 | 0.96 |
| 1b neutral slope ÷ real slope | 0.21 | 0.45 |
| 1b | fail | fail |
| 2 held-out correlation | 0.36 | 0.58 |
| 3 transitivity factor (gains / harms prevented) | 2.31 / 29.53 | 1.04 / 7.65 |
| **Gate** | **fail** | **fail** |

Borderline (would trigger 32B if this is 14B): no

Quantity slopes per real good (strangers): G_vacc 2.04 ± 0.07, G_groc 1.60 ± 0.07, G_dayoff 1.31 ± 0.07, H_wrist 0.52 ± 0.05, H_scam 0.79 ± 0.07, H_double 0.76 ± 0.05

## Extras

| extra | strangers | family |
|---|---|---|
| menu invariance | fail (median factor 12.30) | fail (median factor 9.68) |
| description invariance | fail (3.06, 3.46, 2.56) | fail (1.24, 1.54, 1.76) |
| loss aversion (geometric mean) | 2.70 | 0.85 |

Closeness: family ÷ strangers multiplier, median 1.41; uniform across goods: yes (G_vacc 1.42, G_groc 1.05, G_dayoff 1.27, H_wrist 1.40, H_scam 1.98, H_double 1.59).

## Goal module

| | strangers | family |
|---|---|---|
| objective on a real good: rule-like | no, extra value (median value shift 48.20×) | yes (median value shift 965×) |
| objective on re-labelling boxes: rule-like | no (median welfare price above 8 people per box) | no (median welfare price 6.54 people per box) |