# Unseen goods, option by option

Correlation, over the 9 real options, between each option's held-out value estimate
(probe trained without that option, number of people removed) and its true value from
pilot 3. With 9 points, |r| needs to be above about 0.67 for p < 0.05 (two-sided).

| target | position | model | L0 | L4 | L8 | L12 | L16 | L20 | L24 | L28 | L32 | L36 | L40 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v_a | a_end | real | +0.32 | -0.04 | -0.10 | -0.17 | +0.24 | +0.42 | +0.34 | +0.28 | +0.40 | +0.46 | +0.49 |
| v_a | a_end | random | +0.36 | -0.34 | -0.23 | -0.27 | -0.02 | +0.13 | +0.22 | +0.11 | +0.15 | +0.23 | +0.15 |
| v_a | reply_end | real | – | +0.50 | +0.11 | +0.09 | +0.11 | +0.18 | +0.55 | +0.65 | +0.67 | +0.52 | +0.62 |
| v_a | reply_end | random | – | +0.25 | +0.22 | +0.29 | +0.37 | +0.27 | +0.29 | +0.22 | +0.22 | +0.26 | +0.29 |
| v_a | t-1 | real | – | +0.20 | +0.26 | +0.06 | +0.07 | +0.12 | +0.54 | +0.92 | +0.93 | +0.91 | +0.91 |
| v_a | t-1 | random | – | +0.28 | -0.12 | -0.03 | -0.14 | -0.14 | -0.21 | -0.24 | -0.27 | -0.24 | -0.24 |
| v_b | b_end | real | +0.33 | +0.12 | +0.19 | +0.01 | +0.24 | +0.32 | +0.40 | +0.29 | +0.43 | +0.43 | +0.71 |
| v_b | b_end | random | +0.37 | +0.02 | +0.01 | -0.06 | +0.03 | +0.15 | +0.16 | +0.04 | -0.04 | +0.02 | -0.05 |
| v_b | reply_end | real | – | -0.12 | +0.17 | +0.37 | +0.26 | +0.41 | +0.20 | +0.29 | +0.42 | +0.41 | +0.42 |
| v_b | reply_end | random | – | -0.28 | -0.41 | -0.41 | -0.53 | -0.55 | -0.57 | -0.56 | -0.57 | -0.62 | -0.57 |
| v_b | t-1 | real | – | +0.03 | +0.17 | +0.03 | +0.14 | +0.20 | +0.16 | +0.53 | +0.63 | +0.60 | +0.59 |
| v_b | t-1 | random | – | -0.26 | -0.02 | -0.02 | +0.07 | +0.10 | +0.02 | -0.00 | -0.07 | -0.05 | -0.10 |

## Option A's value at the final position, layer 29, each option held out

| option | true value u | estimate, Qwen3-14B | estimate, random weights |
|---|---|---|---|
| H_fine~gainword | -3.08 | +0.28 | +2.03 |
| H_shift | -1.63 | +1.63 | +1.18 |
| H_burn | -1.59 | +1.66 | -0.93 |
| H_shift~gainword | -1.30 | +1.70 | +0.94 |
| H_burn~gainword | -0.84 | +1.83 | +0.26 |
| G_respite | -0.14 | +3.01 | +0.77 |
| G_dental | +0.00 | +2.76 | +0.09 |
| G_energy | +0.49 | +4.26 | +1.81 |
| H_fine | +0.78 | +3.20 | -0.11 |

Estimates are on an arbitrary offset (the probe's training mean is removed per fold);
only their order and spacing matter.
