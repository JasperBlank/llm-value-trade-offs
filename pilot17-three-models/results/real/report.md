# Pilot 17 results

## The questions, per model

| | llama8 (Llama-3.1-8B-Instruct) | qwen14 (Qwen3-14B) |
|---|---|---|
| **Q1** fixed price? (words / outcomes; bar 0.9) | no: 0.78 / 0.70 | no: 0.75 / 0.77 |
| **Q2** compact comparator? (top 20 on held-out sentence swaps; bar 0.5) | yes: 0.82 (random ≤ 0.07) | yes: 0.96 (random ≤ 0.04) |
| **Q3** generic? (apples / weight; bar 0.3) | yes: 0.86 / 0.88 | yes: 0.94 / 0.96 |

## Also

| | llama8 | qwen14 |
|---|---|---|
| without the two extreme alternatives (words / outcomes) | 0.79 / 0.73 | 0.79 / 0.81 |
| order flips (clear pairs; within ±8 logits) | 0.22 (3945); 0.17 | 0.35 (4558); 0.40 |
| cycles (all gaps > 2 logits) | 0 | 93 |
| menus with \|logit\| > 10 | 4% | 55% |
| inside: common share at the line break after B, by layer | 13: 0.77, 16: 0.76, 19: 0.72, 22: 0.70, 26: 0.64 | 16: 0.73, 20: 0.79, 24: 0.76, 28: 0.73, 32: 0.68 |
| check: line break after A (must be 1) | 1.000, 1.000, 0.999, 0.999, 0.999 | 1.000, 0.999, 0.999, 0.999, 0.999 |
| price before B is read: best line break after A (held out) | A_nl@13 r 0.45 | A_nl@20 r 0.53 |
| price at the line break after B (mid B), by layer | 13: 0.81, 16: 0.78, 19: 0.70, 22: 0.68, 26: 0.71 | 16: 0.42, 20: 0.76, 24: 0.85, 28: 0.82, 32: 0.73 |
| decision at the answer position: whole-state swap by layer (sentences) | 13: 0.03, 16: 0.47, 19: 0.90, 21: 0.91, 22: 0.92, 24: 0.94, 26: 0.94, 29: 0.96 | 16: -0.00, 20: -0.00, 24: 0.01, 26: 0.20, 28: 0.73, 30: 0.81, 32: 0.90, 36: 0.95 |
| the same for apples | 13: -0.03, 16: 0.58, 19: 0.90, 21: 0.91, 22: 0.93, 24: 0.95, 26: 0.94, 29: 0.96 | 16: -0.00, 20: -0.00, 24: 0.00, 26: 0.04, 28: 0.79, 30: 0.86, 32: 0.93, 36: 0.95 |
| top 20 pieces | MLP 29 (+0.486); head 30.27 (+0.366); MLP 28 (+0.167); head 30.26 (+0.154); head 31.1 (+0.135); MLP 26 (+0.134); MLP 22 (+0.101); head 24.24 (+0.097) … | MLP 35 (+0.314); MLP 38 (+0.199); MLP 39 (+0.123); head 38.24 (+0.106); head 39.10 (+0.095); head 38.22 (+0.060); MLP 36 (+0.059); head 39.25 (+0.056) … |
| pilot 5's 20 pieces (qwen14 only) |  | 0.58 on sentences; 1 shared with the top 20 |
| run time (min) | 2.0 | 2.6 |
