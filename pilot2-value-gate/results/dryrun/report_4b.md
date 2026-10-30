# Pilot 2 results: 4b

Value function: log. Probability on the answer letters: 99.5% (prompts under 90%: 0.0%).

| criterion | strangers | family |
|---|---|---|
| 1a quantity sensitivity, all real goods | pass | pass |
| 1b real over neutral (mean probability) | 0.99 | 0.99 |
| 1b neutral slope ÷ real slope | 0.01 | 0.00 |
| 1b | pass | pass |
| 2 held-out correlation | 0.40 | 0.40 |
| 3 transitivity factor (gains / harms prevented) | 4.91 / 1.47 | 4.90 / 1.48 |
| **Gate** | **fail** | **fail** |

Borderline (would trigger 32B if this is 14B): no

Quantity slopes per real good (strangers): G_vacc 1.87 ± 0.01, G_groc 1.98 ± 0.01, G_dayoff 1.99 ± 0.01, H_wrist 1.98 ± 0.01, H_scam 1.92 ± 0.01, H_double 1.93 ± 0.01

## Extras

| extra | strangers | family |
|---|---|---|
| menu invariance | fail (median factor 13.33) | fail (median factor 12.61) |
| description invariance | fail (1.49, 2.26, 2.61) | fail (1.50, 2.22, 2.59) |
| loss aversion (geometric mean) | 1.25 | 1.24 |

Closeness: family ÷ strangers multiplier, median 1.28; uniform across goods: yes (G_vacc 1.28, G_groc 1.30, G_dayoff 1.28, H_wrist 1.28, H_scam 1.30, H_double 1.28).

## Goal module

| | strangers | family |
|---|---|---|
| objective on a real good: rule-like | no, extra value (median value shift 2.98×) | no, extra value (median value shift 3.00×) |
| objective on re-labelling boxes: rule-like | yes (median welfare price above 8 people per box) | yes (median welfare price above 8 people per box) |