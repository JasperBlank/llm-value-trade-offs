# Pilot 12 results (Qwen/Qwen3-14B)

96 sentences. Rating = mean of W0-W2 at 8 people; it correlates 0.83 with the price. Disagreement = the rating minus a cubic in the price (SD 0.63 rating points).

## Pre-registered questions

- **Q1** the disagreement read from the sentence as option A (ridge, held out by value): r = 0.35 at layer 28; permutation p = 0.010 (95th percentile of the null 0.19): **pass** (bars r 0.3, p 0.05)
- **Q2** pushing along the disagreement direction at layer 20: excess = +0.0064 rating points per residual unit [+0.0049, +0.0080]: **pass** (the interval must lie above 0)

**Verdict: different recipes.**

## Pushes (each by the residual distance of 2 price units along the price direction)

| direction | choice: change in logit P(A) | rating: change in points | excess over the price direction's ratio (95% interval) |
|---|---|---|---|
| price | +0.771 | +0.112 | (the reference) |
| rating | +0.726 | +0.127 | +0.0035 [+0.0024, +0.0047] |
| disagreement | +0.103 | +0.055 | +0.0064 [+0.0049, +0.0080] |
| random | +0.126 | -0.008 | -0.0042 [-0.0053, -0.0032] |

The price direction moves the rating 0.145 points per logit of choice. Excess is in rating points per residual unit; multiply by 6.33 for points per push.

## Q1 by layer (held-out r of the disagreement)

| layer | ridge, menu context | ridge, rating context | difference of means, menu context |
|---|---|---|---|
| 8 | 0.01 | 0.04 | 0.13 |
| 12 | 0.26 | 0.25 | 0.15 |
| 16 | 0.22 | 0.25 | 0.20 |
| 20 | 0.28 | 0.35 | 0.24 |
| 24 | 0.33 | 0.41 | 0.21 |
| 28 | 0.35 | 0.37 | 0.16 |
| 32 | 0.34 | 0.33 | 0.06 |

## A small set of directions (PLS, menu context, layer 28, held out by value)

Price and rating correlate 0.83. With noisy held-out readouts this partial correlation cannot reach 0 even if the overlap lives in these directions.

| components | price r | rating r | partial r of price and rating given both readouts |
|---|---|---|---|
| 1 | 0.71 | 0.59 | 0.85 |
| 2 | 0.77 | 0.64 | 0.65 |
| 3 | 0.78 | 0.69 | 0.63 |
| 5 | 0.80 | 0.73 | 0.65 |

## Where the held-out disagreement readout puts each value (mean of its 4 sentences; actual in brackets)

Rated below its price: Preserving -0.13 (-0.06), Doing -0.13 (+0.10), Freedom -0.08 (-0.42), Truth -0.07 (+0.07), Beauty -0.06 (+0.13).
Rated above its price: Defending +0.31 (+0.72), Respect +0.19 (+0.58), Loyalty +0.16 (+0.05), Experiencing +0.14 (-0.18), Justice +0.12 (+0.21).

Run time 2.3 min.
