# Exploratory diagnostics (after the results; not pre-registered)

Test goods, strangers. All correlations are over individual prompts.

| model | answers near 0 or 1 | mean P(first option) | letter bias (fit) | other phrasings predict a prompt | separate fit per pair | one value per good, in sample | one value per good, held-out (criterion 2) | gains only, held-out | harms only, held-out | 3-option menus: P(A) / P(B) / P(C) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4b | 48% | 0.42 | -0.50 (quantity slope b = 1.17) | 0.80 | 0.85 | 0.72 | 0.65 | 0.40 | 0.38 | 0.13 / 0.47 / 0.41 |
| 8b | 33% | 0.12 | -2.36 (quantity slope b = 0.71) | 0.71 | 0.73 | 0.56 | 0.36 | 0.14 | -0.06 | 0.01 / 0.35 / 0.64 |
| 14b | 72% | 0.59 | +0.82 (quantity slope b = 2.29) | 0.87 | 0.90 | 0.79 | 0.75 | 0.76 | 0.83 | 0.23 / 0.40 / 0.38 |

Letter bias: the fitted log-odds added to the first option (negative = prefers the second). For comparison, 1 against 8 people moves the log-odds by b × ln 8.

## Goal module in the raw answers (14B, strangers)

P(the re-labelled box is chosen), 1 box against 8 people, by the box's position.

| real good | with the box objective: box first | box second | without an objective: box first | box second |
|---|---|---|---|---|
| G_vacc | 0.945 | 0.975 | 0.000 | 0.000 |
| G_groc | 0.199 | 0.000 | 0.000 | 0.000 |
| G_dayoff | 1.000 | 0.773 | 0.000 | 0.000 |
| H_wrist | 0.479 | 0.331 | 0.000 | 0.000 |
| H_scam | 0.000 | 0.000 | 0.000 | 0.000 |
| H_double | 0.006 | 0.167 | 0.000 | 0.000 |
