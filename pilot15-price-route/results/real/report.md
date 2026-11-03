# Pilot 15 results (Qwen/Qwen3-14B)

96 sentences as option A against B = "each help a teenager get through a difficult school year" (8:8), and their 994 single-word deletions. The choice's logit correlates 0.88 with pilot 10's price; the median deletion moves it 2.50 logits.

## Pre-registered questions

- **Q1** does Qwen value each option on its own before comparing? Best readout on A's own positions, chosen on half the values: A_nl@20 (r 0.57 there); on the other half r = 0.53: **fail** (bar 0.6)
- **Q2** does an option's worth funnel through one spot on its way to the answer? Best single-position swap: B_nl@24, transfer 0.43 [0.38, 0.48]: **fail** (bar 0.5)
- **Q3** is the comparison the generic "which is bigger" machine (pilots 5-6)? Its top-20 pieces carry 0.60 [0.58, 0.62]; random sets up to 0.07: **pass** (≥ 0.5, random ≤ 0.1)
- **Q4** (pilot 14's Q4 redone) silencing pilot 14's top 10 MLPs at the sentence loses 32% of the choice's price slope; 5 random sets of 10 other MLPs (no block 1) lose 5%, 6%, 5%, 6%, 32%: **pass**

## The readout map

r between the readout's predicted change and the choice's change over all deletions. The readout was trained on the intact sentences only (T).

| position | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 36 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_mean | +0.04 | +0.07 | +0.22 | +0.23 | +0.27 | +0.30 | +0.31 | +0.25 | +0.22 | +0.24 | +0.27 |
| A_end | +0.01 | +0.06 | +0.12 | +0.14 | +0.23 | +0.26 | +0.25 | +0.22 | +0.18 | +0.17 | +0.19 |
| A_nl | +nan | +0.12 | +0.25 | +0.32 | +0.41 | +0.55 | +0.54 | +0.45 | +0.39 | +0.38 | +0.41 |
| B_end | +nan | +0.19 | +0.22 | +0.28 | +0.39 | +0.53 | +0.57 | +0.49 | +0.43 | +0.41 | +0.36 |
| B_nl | +nan | +0.15 | +0.20 | +0.18 | +0.42 | +0.76 | +0.85 | +0.82 | +0.72 | +0.70 | +0.64 |
| END | +nan | +0.20 | +0.23 | +0.20 | +0.27 | +0.35 | +0.82 | +0.99 | +0.99 | +0.99 | +1.00 |

Upper bound: a readout trained on the deletions themselves, one value left out (U).

| position | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 36 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_mean | 0.31 | 0.44 | 0.49 | 0.51 | 0.52 | 0.62 | 0.60 | 0.59 | 0.55 | 0.51 | 0.52 |
| A_end | 0.08 | 0.27 | 0.28 | 0.36 | 0.48 | 0.58 | 0.57 | 0.56 | 0.51 | 0.47 | 0.49 |
| A_nl | -0.23 | 0.37 | 0.44 | 0.50 | 0.61 | 0.72 | 0.72 | 0.70 | 0.65 | 0.60 | 0.59 |
| B_end | -0.23 | 0.35 | 0.40 | 0.40 | 0.53 | 0.69 | 0.76 | 0.77 | 0.74 | 0.74 | 0.74 |
| B_nl | -0.23 | 0.36 | 0.43 | 0.46 | 0.59 | 0.82 | 0.90 | 0.89 | 0.87 | 0.86 | 0.84 |
| END | -0.23 | 0.32 | 0.38 | 0.38 | 0.46 | 0.55 | 0.85 | 0.99 | 1.00 | 1.00 | 1.00 |

The 87 deletions that move the choice more than 10 logits: the share whose sign the readout (T) gets right.

| position | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 36 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_mean | 71% | 77% | 84% | 83% | 85% | 78% | 84% | 82% | 78% | 79% | 91% |
| A_end | 5% | 82% | 82% | 86% | 84% | 86% | 89% | 87% | 86% | 89% | 98% |
| A_nl | 0% | 85% | 93% | 89% | 94% | 94% | 94% | 92% | 91% | 89% | 87% |
| B_end | 0% | 82% | 89% | 84% | 89% | 97% | 94% | 93% | 90% | 92% | 86% |
| B_nl | 0% | 84% | 95% | 95% | 99% | 100% | 100% | 100% | 100% | 100% | 99% |
| END | 0% | 83% | 89% | 83% | 92% | 93% | 100% | 100% | 100% | 100% | 100% |

The readout's held-out r on the intact sentences (one value left out).

| position | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 36 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_mean | 0.41 | 0.63 | 0.70 | 0.72 | 0.76 | 0.76 | 0.74 | 0.71 | 0.70 | 0.67 | 0.69 |
| A_end | 0.00 | 0.28 | 0.48 | 0.61 | 0.66 | 0.69 | 0.68 | 0.68 | 0.67 | 0.68 | 0.73 |
| A_nl | -0.56 | 0.67 | 0.67 | 0.74 | 0.78 | 0.80 | 0.82 | 0.82 | 0.78 | 0.76 | 0.76 |
| B_end | -0.56 | 0.64 | 0.52 | 0.70 | 0.82 | 0.79 | 0.77 | 0.79 | 0.77 | 0.77 | 0.74 |
| B_nl | -0.56 | 0.60 | 0.59 | 0.71 | 0.78 | 0.87 | 0.92 | 0.90 | 0.87 | 0.85 | 0.85 |
| END | -0.56 | 0.44 | 0.45 | 0.62 | 0.52 | 0.63 | 0.88 | 0.99 | 1.00 | 1.00 | 1.00 |

## The route

Transfer of a swap of A's sentence (96 pairs, donors of another value priced ≥ 2 apart; SD of the donors' effect 15.40 logits) when only one position's state at one layer is swapped.

| position | 8 | 16 | 20 | 24 | 28 | 32 | 36 |
|---|---|---|---|---|---|---|---|
| A_end | +0.26 | +0.26 | +0.08 | +0.01 | +0.00 | +0.00 |  |
| A_nl | +0.01 | +0.10 | +0.10 | +0.10 | +0.00 | -0.00 |  |
| B_end | +0.01 | +0.02 | +0.02 | +0.00 | +0.00 | +0.00 |  |
| B_nl | +0.00 | +0.05 | +0.29 | +0.43 | +0.02 | +0.00 |  |
| END |  | +0.00 | +0.00 | +0.02 | +0.75 | +0.91 | +0.95 |

## The comparator (pilot 5's top-20 pieces, blocks counted from 0, at the answer position)

| pieces | transfer (95% interval) |
|---|---|
| top-20 | +0.60 [+0.58, +0.62] |
| top-20 MLPs | +0.31 [+0.29, +0.33] |
| top-20 heads | +0.47 [+0.44, +0.50] |
| random-20 #1 | +0.07 [+0.06, +0.08] |
| random-20 #2 | -0.00 [-0.01, +0.00] |
| random-20 #3 | -0.00 [-0.01, +0.00] |

Q4 random sets: MLP 21, MLP 26, MLP 5, MLP 24, MLP 25, MLP 29, MLP 4, MLP 3, MLP 7, MLP 27; MLP 9, MLP 29, MLP 27, MLP 11, MLP 24, MLP 3, MLP 28, MLP 7, MLP 26, MLP 21; MLP 27, MLP 4, MLP 23, MLP 7, MLP 24, MLP 30, MLP 25, MLP 5, MLP 8, MLP 9; MLP 4, MLP 21, MLP 22, MLP 6, MLP 24, MLP 25, MLP 27, MLP 5, MLP 23, MLP 30; MLP 9, MLP 24, MLP 2, MLP 3, MLP 5, MLP 11, MLP 21, MLP 22, MLP 6, MLP 23

Run time 4.5 min.
