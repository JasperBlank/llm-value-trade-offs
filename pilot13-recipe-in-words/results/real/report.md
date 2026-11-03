# Pilot 13 results (Qwen/Qwen3-14B)

96 old sentences (pilot 10's prices and ratings) and 24 new ones, priced here against the old (13824 trade-offs). Features: the model's own 0-9 answers to ten descriptive questions (expected digit).

## Pre-registered questions

- **P1** the price from the nine features in words (F2-F10), held out by value: r = 0.79: **fail** (bar 0.8). With F1 (size) added: 0.80; F1 alone: 0.79. The rating from the same nine: 0.86.
- **P2** 'putting right something morally wrong' weighs -0.04 more (standardised) in the rating than in the price, 95% interval [-0.22, +0.15]: **fail** (the interval must lie above 0)
- **P3** pilot 12's disagreement push: excess +0.0064 rating points per residual unit [+0.0049, +0.0079] over sentences; the 20 random directions -0.0062 to +0.0050 (rank p = 0.048): **pass** (must beat all 20)
- **P4** the 24 new sentences, the recipe's predicted prices against their measured prices: r = 0.71: **pass** (bar 0.7)

## The two recipes

Standardised weights of F2-F10, fitted together on the old sentences; r is each feature's plain correlation.

| feature | price (choice) | rating (stated) | rating minus price (95% interval) | r with price | r with rating |
|---|---|---|---|---|---|
| severity | -0.20 | -0.05 | +0.15 [-0.12, +0.35] | +0.59 | +0.67 |
| duration | +0.36 | -0.11 | -0.47 [-0.65, -0.20] | +0.75 | +0.66 |
| health_safety | +0.03 | +0.01 | -0.02 [-0.13, +0.12] | +0.35 | +0.40 |
| moral | +0.10 | +0.05 | -0.04 [-0.22, +0.15] | +0.59 | +0.64 |
| relationships | +0.08 | +0.16 | +0.07 [-0.11, +0.26] | +0.61 | +0.71 |
| meaning | +0.65 | +0.65 | +0.01 [-0.35, +0.33] | +0.80 | +0.88 |
| pleasure | +0.02 | +0.15 | +0.13 [+0.01, +0.28] | +0.41 | +0.56 |
| legacy | -0.14 | +0.14 | +0.28 [-0.03, +0.58] | +0.75 | +0.84 |
| concreteness | +0.01 | -0.06 | -0.07 [-0.20, +0.03] | +0.08 | +0.04 |
| size (F1; not in the recipe) | | | | +0.81 | +0.83 |

With F1 added (F1-F10 together), price: size +0.38, severity -0.22, duration +0.29, health_safety +0.03, moral +0.06, relationships +0.12, meaning +0.31, pleasure +0.04, legacy -0.09, concreteness +0.02.
Rating: size +0.29, severity -0.07, duration -0.17, health_safety +0.01, moral +0.02, relationships +0.18, meaning +0.40, pleasure +0.17, legacy +0.18, concreteness -0.06.

## Pushes at layer 20

Every direction is a unit vector pushed by ±6.33 residual units (2 price units along the price direction), on the sentence's tokens: as option A against its 4 partners for the choice, in the W0 prompt for the rating. The price direction moves the rating 0.145 points per logit of choice; excess = rating change minus that ratio times the choice change (per residual unit; multiply by 6.33 for per push).

| direction | cos with price direction | choice: logit P(A) per push | rating: points per push | excess |
|---|---|---|---|---|
| price | +1.00 | +0.771 | +0.112 | +0.0000 |
| rating | +0.91 | +0.729 | +0.126 | +0.0033 |
| disagreement | +0.05 | +0.103 | +0.055 | +0.0064 |
| feature:size | +0.85 | +0.613 | +0.112 | +0.0036 |
| feature:severity | +0.72 | +0.510 | +0.087 | +0.0020 |
| feature:duration | +0.86 | +0.564 | +0.099 | +0.0028 |
| feature:health_safety | +0.46 | +0.512 | +0.052 | -0.0035 |
| feature:moral | +0.71 | +0.628 | +0.099 | +0.0012 |
| feature:relationships | +0.71 | +0.547 | +0.093 | +0.0022 |
| feature:meaning | +0.86 | +0.554 | +0.101 | +0.0033 |
| feature:pleasure | +0.56 | +0.620 | +0.106 | +0.0026 |
| feature:legacy | +0.83 | +0.564 | +0.100 | +0.0029 |
| feature:concreteness | +0.18 | +0.453 | +0.036 | -0.0047 |
| 20 random directions (range) | | -0.334 to +0.326 | -0.033 to +0.060 | -0.0062 to +0.0050 |

- The random benchmark: the price direction moves the choice +0.771 logits per push; the largest random direction 0.334.
- Do the feature pushes move the choice in line with each feature's correlation with the price? Spearman +0.44 (10 features).
- Does each feature's rating excess follow how much more the rating recipe weighs it? Spearman +0.37 (9 features).

## The new sentences

Predictors of their measured prices (r): recipe_in_words 0.71, with_F1 0.67, F1_alone 0.55, states_ridge_layer20 0.74, how_good_rating 0.74. The states readout's held-out r on the old sentences: 0.84.

| value | sentence | measured price | from the words | from the states | how good (W0-W2) |
|---|---|---|---|---|---|
| Caring | are each able to give a seriously ill family member the care they need to recover | +4.60 | +4.14 | +3.03 | 8.66 |
| Justice | each see the company that poisoned their town held fully responsible | +4.32 | +1.52 | -0.26 | 7.15 |
| Longevity | each gain ten more years of healthy life | +4.31 | +2.96 | +1.68 | 7.09 |
| Legacy | each plant a tree that will stand for a hundred years | +4.23 | +2.32 | +2.54 | 7.21 |
| Happiness | each have a month in which they feel truly content | +1.01 | -1.39 | +0.35 | 6.71 |
| Protection | are each protected from a burglary | +0.51 | -0.61 | +0.75 | 7.06 |
| Nurturing | each help a teenager get through a difficult school year | +0.35 | +0.87 | +1.17 | 7.36 |
| Non-suffering | are each spared a month of back pain | +0.19 | -0.12 | +0.64 | 6.99 |
| Pleasure | each enjoy a month-long holiday doing everything they love | +0.14 | +2.71 | +1.92 | 7.21 |
| Preserving | each see a favourite old shop stay open another year | -0.24 | -1.61 | -0.92 | 6.44 |
| Fairness | each have a decade of unequal pay made up to them in full | -1.11 | -0.32 | +0.35 | 6.84 |
| Respect | each be honoured publicly for a lifetime of quiet work | -1.33 | +1.63 | +0.98 | 7.59 |
| Beauty | each watch a spectacular sunset over the sea | -1.93 | -1.73 | -1.23 | 6.52 |
| Doing | each turn the activity they love into their full-time job | -2.23 | +0.78 | +0.85 | 6.85 |
| Spirituality | each feel a moment of peace during a quiet prayer or meditation | -2.53 | -2.52 | -1.58 | 5.90 |
| Loyalty | each get a birthday call from an old friend | -2.59 | -1.54 | -3.62 | 5.92 |
| Truth | each finally learn the truth about what happened to a parent they never knew | -2.65 | +1.60 | -1.42 | 6.46 |
| Achievement | each solve a tricky puzzle they had been stuck on | -2.77 | -2.07 | -1.62 | 6.57 |
| Diversity | each spend a month living with a family from another culture | -3.09 | +0.39 | -0.95 | 6.93 |
| Freedom | each get to decide where to go for lunch | -3.11 | -5.25 | -1.81 | 5.84 |
| Defending | each speak up when a waiter is treated rudely | -4.26 | -1.84 | -2.97 | 7.19 |
| Embodying | each stick to a small resolution they made | -6.45 | -3.27 | -3.81 | 5.85 |
| Experiencing | each spend a week travelling somewhere they have never been | -7.29 | -1.00 | -0.14 | 6.06 |
| Knowing | each find out how a magic trick works | -8.07 | -4.12 | -2.70 | 4.68 |

Run time 13.3 min.
