# Pilot 11 results (Qwen/Qwen3-14B)

96 sentences; the vibe direction (difference of means, price thirds) learned on all but one value and read on that value. Layer chosen by Q1 (among 4-32): **20**.

## Pre-registered questions

- **Q1** readable before option B: held-out r = 0.78: **pass** (bar 0.7)
- **Q2** the same direction read in the rating prompt: r with the rating = 0.64; price and rating correlate 0.81, and 0.65 once the reading is accounted for: **fail** (bars 0.6 and 0.3)
- **Q3** pushing along it by ±2 price units at layer 20: menu transfer 0.38, rating transfer 0.17; a random direction: -0.02 and -0.02: **fail** (bars 0.5; random at most 0.1)

**Verdict: readable but not shared.**

## Within stakes level (each level's mean removed)

Price and rating r = 0.54; the reading against the price 0.37; the rating-prompt projection against the rating 0.04; price and rating once it is accounted for 0.57.
Cosine between the price direction (menu prompt) and a rating direction (rating prompt) at layer 20: 0.87.

## Every layer (held-out r; vibe direction unless marked ridge)

| layer | menu, mean: price | menu, last: price | rating ctx, mean: price | rating ctx: rating | ridge menu: price | ridge rating ctx: rating | menu direction read in rating ctx: rating | partial price-rating | disagreement |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.46 | 0.17 | 0.46 | 0.42 | 0.52 | 0.47 | 0.44 | 0.76 | 0.13 |
| 4 | 0.59 | 0.26 | 0.62 | 0.58 | 0.67 | 0.61 | 0.48 | 0.75 | 0.18 |
| 8 | 0.72 | 0.41 | 0.72 | 0.56 | 0.78 | 0.65 | 0.59 | 0.69 | 0.23 |
| 12 | 0.74 | 0.55 | 0.74 | 0.60 | 0.81 | 0.70 | 0.62 | 0.66 | 0.26 |
| 16 | 0.77 | 0.63 | 0.77 | 0.64 | 0.83 | 0.72 | 0.62 | 0.67 | 0.32 |
| 20 | 0.78 | 0.66 | 0.77 | 0.65 | 0.84 | 0.76 | 0.64 | 0.65 | 0.34 |
| 24 | 0.78 | 0.67 | 0.76 | 0.65 | 0.84 | 0.76 | 0.64 | 0.64 | 0.32 |
| 28 | 0.76 | 0.65 | 0.76 | 0.63 | 0.81 | 0.75 | 0.65 | 0.64 | 0.29 |
| 32 | 0.72 | 0.65 | 0.74 | 0.58 | 0.78 | 0.69 | 0.58 | 0.70 | 0.23 |
| 36 | 0.70 | 0.68 | 0.73 | 0.56 | 0.77 | 0.65 | 0.53 | 0.72 | 0.19 |
| 40 | 0.66 | 0.65 | 0.68 | 0.49 | 0.76 | 0.64 | 0.47 | 0.75 | 0.15 |

## Pushing along the direction (±2 price units)

| layer | kind | menu transfer (small / medium / large) | rating transfer | rating points per price unit | log-scale score per price unit |
|---|---|---|---|---|---|
| 4 | vibe | 0.10 (0.09 / 0.10 / 0.10) | 0.03 | 0.008 | 0.032 |
| 8 | vibe | 0.11 (0.11 / 0.13 / 0.05) | 0.03 | 0.008 | 0.052 |
| 12 | vibe | 0.27 (0.28 / 0.30 / 0.22) | 0.05 | 0.018 | 0.063 |
| 16 | vibe | 0.41 (0.41 / 0.42 / 0.38) | 0.12 | 0.038 | 0.188 |
| 20 | random | -0.02 (-0.03 / -0.01 / -0.02) | -0.02 | -0.007 | -0.031 |
| 20 | vibe | 0.38 (0.37 / 0.40 / 0.38) | 0.17 | 0.056 | 0.243 |
| 24 | vibe | 0.08 (0.08 / 0.08 / 0.09) | 0.06 | 0.019 | 0.084 |
| 28 | vibe | 0.00 (0.00 / 0.00 / -0.00) | 0.00 | 0.001 | 0.007 |
| 32 | vibe | -0.00 (0.00 / -0.01 / 0.00) | 0.00 | 0.000 | 0.002 |
| 36 | vibe | -0.00 (-0.00 / -0.00 / 0.00) | 0.00 | 0.000 | -0.002 |

Base menu prompts: median |logit P(A)| 4.50. Run time 6.0 min.
