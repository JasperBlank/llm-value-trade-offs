# Pilot 17 results

## The questions, per model

| | llama8 (Llama-3.1-8B-Instruct) |
|---|---|
| **Q1** fixed price? (words / outcomes; bar 0.9) | no: 0.78 / 0.70 |
| **Q2** compact comparator? (top 20 on held-out sentence swaps; bar 0.5) | yes: 0.82 (random ≤ 0.06) |
| **Q3** generic? (apples / weight; bar 0.3) | yes: 0.87 / 0.88 |
| layer where the decision reaches the answer position (whole-state swap ≥ 0.5) | 19 |
| **Q2b** decision writers before that layer (held-out sentence swaps; bar 0.4) | yes: 0.86 (random ≤ 0.02) |
| **Q3b** the same on apples / weight (bar 0.3) | yes: 0.90 / 0.95 (random ≤ 0.06) |

## Also

| | llama8 |
|---|---|
| without the two extreme alternatives (words / outcomes) | 0.79 / 0.73 |
| order flips (clear pairs; within ±8 logits) | 0.21 (3956); 0.16 |
| cycles (all gaps > 2 logits) | 0 |
| menus with \|logit\| > 10 | 3% |
| inside: common share at the line break after B, by layer | 13: 0.79, 16: 0.76, 19: 0.73, 22: 0.70, 26: 0.64 |
| check: line break after A (must be 1) | 1.000, 0.999, 0.999, 0.999, 0.999 |
| price before B is read: best line break after A (held out) | A_nl@13 r 0.45 |
| price at the line break after B (mid B), by layer | 13: 0.80, 16: 0.78, 19: 0.70, 22: 0.67, 26: 0.70 |
| decision at the answer position: whole-state swap by layer (sentences) | 13: 0.01, 16: 0.43, 19: 0.89, 21: 0.89, 22: 0.90, 24: 0.93, 26: 0.93, 29: 0.95 |
| the same for apples | 13: -0.02, 16: 0.58, 19: 0.90, 21: 0.91, 22: 0.92, 24: 0.94, 26: 0.94, 29: 0.96 |
| top 20 pieces | MLP 29 (+0.499); head 30.27 (+0.374); MLP 28 (+0.167); head 30.26 (+0.160); MLP 26 (+0.136); head 31.1 (+0.134); MLP 22 (+0.110); head 24.24 (+0.101) … |
| pilot 5's 20 pieces (qwen14 only): transfer on sentences / apples / weight; shared with the logit writers / decision writers |  |
| decision writers | head 17.24 (+0.084); MLP 18 (+0.052); MLP 17 (+0.038); head 16.1 (+0.027); MLP 16 (+0.025); head 18.20 (+0.021); head 18.28 (+0.018); head 18.16 (+0.016) … |
| run time (min) | 2.6 |
