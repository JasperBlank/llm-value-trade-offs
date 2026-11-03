# Pilot 13, exploratory: the features rated blind by Claude

Plan and set-up: [`EXTERNAL_RATINGS.md`](../../EXTERNAL_RATINGS.md). Two fresh Claude instances rated the 120 sentences on the ten questions without seeing any price, rating or result. Qwen's features are its own answers (expected digit).

| | Qwen's own features | Claude's blind ratings |
|---|---|---|
| price from F2-F10, held out by value (P1) | 0.79 | 0.84 |
| ... with F1 (size) added | 0.80 | 0.83 |
| ... F1 alone | 0.79 | 0.77 |
| the stated rating from F2-F10, held out | 0.86 | 0.62 |
| the 24 new sentences' prices from the recipe (P4) | 0.71 | 0.68 |
| ... with F1 added | 0.67 | 0.68 |
| mean correlation among F2-F10 (halo check; 120 sentences) | 0.46 | 0.02 |

## Feature by feature

| feature | Qwen and Claude agree (r, 120 sentences) | r with price, Qwen's | r with price, Claude's | rating minus price weight, Qwen's | rating minus price weight, Claude's |
|---|---|---|---|---|---|
| size | +0.82 | +0.81 | +0.78 | (not in the recipe) | (not in the recipe) |
| severity | +0.61 | +0.59 | +0.43 | +0.15 [-0.12, +0.35] | -0.10 [-0.33, +0.13] |
| duration | +0.86 | +0.75 | +0.77 | -0.47 [-0.65, -0.20] | +0.09 [-0.18, +0.34] |
| health_safety | +0.54 | +0.35 | +0.16 | -0.02 [-0.13, +0.12] | -0.07 [-0.26, +0.10] |
| moral | +0.46 | +0.59 | +0.00 | -0.04 [-0.22, +0.15] | +0.15 [+0.01, +0.29] |
| relationships | +0.34 | +0.61 | +0.16 | +0.07 [-0.11, +0.26] | +0.18 [+0.03, +0.33] |
| meaning | +0.77 | +0.80 | +0.75 | +0.01 [-0.35, +0.33] | -0.03 [-0.24, +0.20] |
| pleasure | +0.35 | +0.41 | -0.21 | +0.13 [+0.01, +0.28] | +0.00 [-0.16, +0.14] |
| legacy | +0.49 | +0.75 | +0.46 | +0.28 [-0.03, +0.58] | -0.12 [-0.27, +0.01] |
| concreteness | +0.08 | +0.08 | -0.63 | -0.07 [-0.20, +0.03] | +0.14 [-0.02, +0.34] |

Weights are standardised and fitted together on the 96 old sentences; intervals are 95% bootstraps over sentences (2,000).
