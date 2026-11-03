# Pilot 17 results

## The questions, per model

| | qwen32 (Qwen3-32B) | qwen14 (Qwen3-14B) |
|---|---|---|
| **Q1** fixed price? (words / outcomes; bar 0.9) | no: 0.71 / 0.82 | no: 0.75 / 0.77 |
| **Q2** compact comparator? (top 20 on held-out sentence swaps; bar 0.5) | yes: 0.75 (random ≤ -0.00) | no: 0.96 (random ≤ 0.10) |
| **Q3** generic? (apples / weight; bar 0.3) | yes: 0.83 / 0.81 | yes: 0.94 / 0.96 |
| layer where the decision reaches the answer position (whole-state swap ≥ 0.5) | 51 | 28 |
| **Q2b** decision writers before that layer (held-out sentence swaps; bar 0.4) | yes: 0.46 (random ≤ -0.00) | yes: 0.71 (random ≤ 0.01) |
| **Q3b** the same on apples / weight (bar 0.3) | yes: 0.47 / 0.38 (random ≤ 0.01) | yes: 0.81 / 0.76 (random ≤ 0.00) |

## Also

| | qwen32 | qwen14 |
|---|---|---|
| without the two extreme alternatives (words / outcomes) | 0.72 / 0.82 | 0.79 / 0.81 |
| order flips (clear pairs; within ±8 logits) | 0.18 (4442); 0.15 | 0.34 (4559); 0.38 |
| cycles (all gaps > 2 logits) | 12 | 96 |
| menus with \|logit\| > 10 | 11% | 55% |
| inside: common share at the line break after B, by layer | 26: 0.74, 32: 0.71, 38: 0.70, 45: 0.72, 51: 0.66 | 16: 0.73, 20: 0.79, 24: 0.75, 28: 0.73, 32: 0.68 |
| check: line break after A (must be 1) | 1.000, 1.000, 1.000, 1.000, 0.999 | 1.000, 0.999, 0.999, 0.999, 0.998 |
| price before B is read: best line break after A (held out) | A_nl@45 r 0.41 | A_nl@20 r 0.53 |
| price at the line break after B (mid B), by layer | 26: 0.44, 32: 0.44, 38: 0.48, 45: 0.83, 51: 0.81 | 16: 0.43, 20: 0.76, 24: 0.85, 28: 0.82, 32: 0.73 |
| decision at the answer position: whole-state swap by layer (sentences) | 26: 0.00, 32: 0.00, 38: 0.01, 42: 0.01, 45: 0.00, 48: 0.03, 51: 0.51, 58: 0.91 | 16: 0.00, 20: 0.00, 24: 0.01, 26: 0.20, 28: 0.73, 30: 0.80, 32: 0.90, 36: 0.95 |
| the same for apples | 26: 0.00, 32: 0.00, 38: 0.00, 42: -0.00, 45: 0.00, 48: 0.02, 51: 0.57, 58: 0.94 | 16: -0.00, 20: -0.00, 24: 0.00, 26: 0.04, 28: 0.79, 30: 0.86, 32: 0.93, 36: 0.95 |
| top 20 pieces | MLP 59 (+0.315); head 62.39 (+0.228); MLP 62 (+0.212); head 63.21 (+0.102); head 63.19 (+0.094); MLP 60 (+0.093); head 62.37 (+0.066); head 63.43 (+0.055) … | MLP 35 (+0.317); MLP 38 (+0.195); MLP 39 (+0.122); head 38.24 (+0.107); head 39.10 (+0.095); head 38.22 (+0.060); MLP 36 (+0.058); head 39.25 (+0.056) … |
| pilot 5's 20 pieces (qwen14 only): transfer on sentences / apples / weight; shared with the logit writers / decision writers |  | 0.57 / 0.55 / 0.39; 1 / 14 |
| decision writers | MLP 50 (+2.736); head 50.61 (+0.619); head 49.24 (+0.417); head 50.60 (+0.358); head 50.56 (+0.332); head 50.58 (+0.244); head 48.31 (+0.173); MLP 49 (+0.154) … | MLP 27 (+1.079); MLP 26 (+0.826); head 27.11 (+0.434); head 26.37 (+0.218); head 26.36 (+0.115); head 27.3 (+0.098); head 27.10 (+0.078); MLP 25 (+0.058) … |
| run time (min) | 4.8 | 2.7 |
