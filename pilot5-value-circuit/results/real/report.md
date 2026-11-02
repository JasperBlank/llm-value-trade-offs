# Pilot 5 results

## The pre-registered test (steering at the last position)

300 fresh prompts. Slope = change in log-odds of A per value unit added (predicted: +1 for A's direction, -1 for B's, about 0 for the controls).

| layer | A's direction | B's direction | random | shuffled 1 | shuffled 2 | verdict |
|---|---|---|---|---|---|---|
| 24 | +0.02 [+0.02, +0.03] | -0.02 [-0.02, -0.01] | +0.01 [+0.00, +0.01] | -0.01 [-0.02, -0.01] | -0.02 [-0.03, -0.02] | an effect, but not specific to value (controls failed) |
| 28 (primary) | +0.07 [+0.06, +0.07] | -0.08 [-0.08, -0.07] | -0.01 [-0.01, -0.00] | -0.02 [-0.03, -0.02] | -0.04 [-0.05, -0.04] | an effect, but not specific to value (controls failed) |
| 32 | +0.06 [+0.05, +0.06] | -0.07 [-0.07, -0.06] | +0.01 [+0.00, +0.01] | -0.01 [-0.02, -0.01] | -0.04 [-0.04, -0.03] | an effect, but not specific to value (controls failed) |
| 36 | +0.08 [+0.07, +0.08] | -0.07 [-0.08, -0.07] | +0.01 [+0.01, +0.02] | +0.01 [+0.01, +0.02] | -0.01 [-0.01, -0.00] | used, but not at full size |

**Main verdict (layer 28): an effect, but not specific to value (controls failed).** Full size at any layer: no.

One currency (A's slope by kind of good, layer 28; within a factor 1.5):

- gain worded vs harm worded: slopes +0.07 and +0.07 (n = 247, 53), ratio 0.98 [0.85, 1.22]: one currency
- money vs nonmoney: slopes +0.07 and +0.07 (n = 90, 210), ratio 1.02 [0.89, 1.16]: one currency

## The directions

| layer | option-level r, A | option-level r, B | cos(w_A, w_B) | steering length / residual length |
|---|---|---|---|---|
| 24 | 0.62 | 0.09 | -0.25 | 0.419 / 169 |
| 28 | 0.92 | 0.53 | +0.01 | 1.68 / 270 |
| 32 | 0.93 | 0.62 | +0.03 | 3.01 / 484 |
| 36 | 0.92 | 0.60 | +0.06 | 4.4 / 813 |

## Exploratory: the circuit

Writers of A's value at layer 28: heads 55%, MLPs 45%, embedding 0%; half of it carried by 7 of 1149 pieces (80% by 25). Overall number-to-good ratio 1.38 (b = 1.46).
Writers of B's value: heads 52%, MLPs 48%; half by 6 pieces.

Top writers of A's value:

| piece | share | number part | good part | ratio |
|---|---|---|---|---|
| MLP 26 | 17.1% | +0.191 | +0.198 | 0.97 |
| MLP 27 | 9.9% | +0.126 | +0.097 | 1.3 |
| MLP 25 | 8.0% | +0.091 | +0.092 | 0.99 |
| head 26.2 | 4.6% | +0.072 | +0.030 | 2.42 |
| head 26.11 | 4.5% | +0.050 | +0.054 | 0.93 |
| head 27.11 | 4.4% | +0.069 | +0.029 | 2.4 |
| head 26.35 | 3.1% | +0.057 | +0.010 | 5.52 |
| head 24.22 | 2.7% | +0.044 | +0.015 | 2.87 |
| head 27.23 | 2.6% | +0.034 | +0.025 | 1.34 |
| head 26.37 | 2.0% | +0.018 | +0.028 | 0.63 |

Where the heads read A's value from (share of the total): template_end 19%, separator 18%, reply 9%, a_text 6%

Readers (direct share of the log-odds): heads 30%, MLPs 70%. Letter heads among the top 10: head 38.24, head 38.22, head 34.6, head 34.35, head 28.21, head 38.35.

| head | share | attention to A label | to B label | corr(label attention, log-odds) |
|---|---|---|---|---|
| head 38.24 | 10.0% | 0.00 | 0.01 | +0.59 |
| head 39.10 | 8.4% | 0.01 | 0.01 | -0.02 |
| head 39.25 | 6.0% | 0.00 | 0.00 | -0.33 |
| head 38.22 | 5.8% | 0.01 | 0.02 | +0.72 |
| head 34.6 | 4.5% | 0.14 | 0.07 | +0.75 |
| head 34.35 | 3.7% | 0.05 | 0.04 | +0.90 |
| head 28.21 | 3.2% | 0.07 | 0.05 | +0.61 |
| head 39.12 | 3.2% | 0.01 | 0.01 | -0.19 |
| head 38.35 | 2.8% | 0.04 | 0.04 | +0.79 |
| head 35.17 | 1.7% | 0.07 | 0.12 | -0.05 |

Patching (195 pairs; transfer of the swapped good, 0 = none, 1 = all):

- resid@4: -0.01 [-0.02, +0.01]
- resid@8: -0.01 [-0.02, +0.00]
- resid@12: -0.00 [-0.01, +0.01]
- resid@16: +0.01 [-0.00, +0.03]
- resid@20: +0.00 [-0.01, +0.02]
- resid@24: +0.03 [+0.01, +0.05]
- resid@28: +0.77 [+0.71, +0.83]
- resid@32: +0.95 [+0.94, +0.96]
- resid@36: +0.97 [+0.96, +0.98]
- resid@40: +1.00 [+1.00, +1.00]
- top-5: +0.30 [+0.26, +0.33]
- top-10: +0.50 [+0.47, +0.54]
- top-20: +0.65 [+0.61, +0.70]
- top-50: +0.65 [+0.60, +0.71]
- random-5: -0.00 [-0.01, +0.01]
- random-10: -0.01 [-0.02, +0.00]
- random-20: +0.01 [-0.01, +0.02]
- random-50: +0.00 [-0.01, +0.01]

Readers under the push (A's direction at layer 28, -2 to +2; log-odds shift -0.14 to +0.13):

- head 38.24: attention to A label 0.00 -> 0.00, B label 0.01 -> 0.01
- head 39.10: attention to A label 0.02 -> 0.02, B label 0.01 -> 0.01
- head 39.25: attention to A label 0.00 -> 0.00, B label 0.00 -> 0.00
- head 38.22: attention to A label 0.01 -> 0.01, B label 0.03 -> 0.03
- head 34.6: attention to A label 0.12 -> 0.13, B label 0.09 -> 0.09
- head 34.35: attention to A label 0.05 -> 0.05, B label 0.03 -> 0.04
- head 28.21: attention to A label 0.05 -> 0.05, B label 0.08 -> 0.08
- head 39.12: attention to A label 0.01 -> 0.01, B label 0.01 -> 0.01
- head 38.35: attention to A label 0.04 -> 0.04, B label 0.04 -> 0.04
- head 35.17: attention to A label 0.07 -> 0.07, B label 0.12 -> 0.13

## Exploratory: the objective, read at the last position

- goal1: the choice moves +13.7 log-odds towards the target; at layer 28 the target's value readout moves +1.55, the partner's -2.82, their difference +4.38 (32% of the choice's move).
- goal2: the choice moves +14.7 log-odds towards the target; at layer 28 the target's value readout moves +4.77, the partner's -1.87, their difference +6.64 (45% of the choice's move).

