# Pilot 16 results (Qwen/Qwen3-14B)

96 sentences and their 994 single-word deletions as option A, each against 8 alternatives B (prices -7.3, -3.1, -2.6, -2.2, -1.1, +0.2, +1.0, +4.3): 8720 menus at 8:8. The median word effect is 1.75 logits.

## Pre-registered questions (bar 0.9 for "absolute")

- **Q1** inside: is A's finished price the same whatever B is? Common share of the readout's word effects at the line break after B, layer 24: 0.752: **relative**
- **Q2** choice: does a word count the same whatever B is? Common share of the choice's word effects: 0.747: **relative**
- **Q3** choice: are whole outcomes priced the same whatever B is? Common share of the 96 × 8 logits: 0.762: **relative**
- **Verdict:** relative, already in the finished price

## Common share of the readout's word effects, by position and layer

One ridge readout per position and layer, trained on the intact menus of all Bs to predict the choice logit. The line break after A comes before B and must give 1.

| position | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|
| A_nl | 1.000 | 0.999 | 0.999 | 0.999 | 0.999 |
| B_nl | 0.722 | 0.789 | 0.752 | 0.729 | 0.679 |
| END | 0.599 | 0.715 | 0.804 | 0.739 | 0.745 |

Without the two extreme Bs (-7.3 and +4.3): choice words 0.786, whole outcomes 0.805; readout:

| position | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|
| A_nl | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |
| B_nl | 0.751 | 0.809 | 0.769 | 0.753 | 0.709 |
| END | 0.613 | 0.730 | 0.822 | 0.782 | 0.789 |

With a separate readout for each B (line break after B): B_nl@16 0.585, B_nl@20 0.707, B_nl@24 0.721, B_nl@28 0.677, B_nl@32 0.628.

## The shape of the dependence on B

| B (price) | mean logit of the 96 | slope: whole outcomes | slope: word effects, choice | r with the mean word effect | slope: word effects, readout |
|---|---|---|---|---|---|
| each spend a week travelling somewhere they have never been (-7.3) | +16.06 | 0.61 | 1.05 | 0.88 | 0.89 |
| each get to decide where to go for lunch (-3.1) | +15.47 | 1.02 | 1.28 | 0.90 | 1.11 |
| each finally learn the truth about what happened to a parent they never knew (-2.6) | +10.82 | 0.49 | 0.91 | 0.89 | 0.70 |
| each turn the activity they love into their full-time job (-2.2) | +4.66 | 1.15 | 1.18 | 0.92 | 1.23 |
| each have a decade of unequal pay made up to them in full (-1.1) | +2.93 | 1.24 | 0.95 | 0.89 | 1.03 |
| are each spared a month of back pain (+0.2) | +4.23 | 1.55 | 1.20 | 0.87 | 1.23 |
| each have a month in which they feel truly content (+1.0) | -2.81 | 1.23 | 0.88 | 0.86 | 1.03 |
| each see the company that poisoned their town held fully responsible (+4.3) | -7.88 | 0.71 | 0.55 | 0.78 | 0.79 |

Transfer at the line break after B, layer 24: a readout trained with one B predicts the choice's word effects with that B at r 0.83 (mean) and with the other Bs at r 0.77.

Run time 1.9 min.
