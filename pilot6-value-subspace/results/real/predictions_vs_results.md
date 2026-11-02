# Pilot 6: predictions against results

| question | result | Claude | Jasper |
|---|---|---|---|
| q1_quantity_half_layer | 28 | 28 [24, 32] hit, off by 0 | 20 [10, 30] hit, off by 8 |
| q2_weight_half_layer | 28 | 28 [24, 32] hit, off by 0 | 22 [12, 32] hit, off by 6 |
| q3_quantity_top20_transfer | 0.545 | 0.4 [0.05, 0.8] hit, off by 0.14 | 0.65 [0.4, 0.8] hit, off by 0.11 |
| q4_weight_top20_transfer | 0.385 | 0.4 [0.05, 0.8] hit, off by 0.015 | 0.65 [0.4, 0.8] miss, off by 0.26 |
| q5_das_k1_share_28 | 0.985 | 0.65 [0.3, 0.9] miss, off by 0.34 | 0.5 [0.1, 0.9] miss, off by 0.49 |
| q6_smallest_k_for_0.9 | 1 | 4 wrong | 8 wrong |
| q7_probe_swap_share_28 | 0.00908 | 0.08 [0.01, 0.3] miss, off by 0.071 | 0.5 [0.1, 0.9] miss, off by 0.49 |
| q8_abs_cos_learned_probe_28 | 0.0924 | 0.15 [0.03, 0.5] hit, off by 0.058 | 0.0 [0.0, 0.5] hit, off by 0.092 |
| q9_corr_learned_va_heldout_28 | 0.519 | 0.45 [0.15, 0.8] hit, off by 0.069 | 0.5 [0.3, 0.8] hit, off by 0.019 |
| q10_push_ratio_learned_over_probe_28 | 8.59 | 5.0 [1.0, 20.0] hit, off by 3.6 | 3 [0.5, 4] miss, off by 5.6 |
| q11a_p1_verdict | generic | 40% on it; Brier 0.54 | 30% on it; Brier 0.78 |
| q11b_p2_pass | True | 70%; Brier 0.09 | 50%; Brier 0.25 |
| q11c_p3_pass | True | 50%; Brier 0.25 | 50%; Brier 0.25 |
| q11d_p4_pass | True | 40%; Brier 0.36 | 50%; Brier 0.25 |

Mean Brier over the probability questions (lower is better): Claude 0.311 (4 questions), Jasper 0.382 (4 questions)
- Claude: 7 of 9 80% ranges contain the result
- Jasper: 5 of 9 80% ranges contain the result
