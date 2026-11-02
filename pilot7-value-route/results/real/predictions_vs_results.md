# Pilot 7: predictions against results

| question | result | Claude | Jasper |
|---|---|---|---|
| q1_after_A_half_layer | 20 | 16 [8, 24] hit, off by 4 | – |
| q2_a_site_share | 0.357 | 0.2 [0.03, 0.6] hit, off by 0.16 | – |
| q3_b_site | after_B | after_B right | – |
| q4_a_site_number_share | 0.295 | 0.35 [0.05, 0.8] hit, off by 0.055 | – |
| q5_b_site_number_share | 0.765 | 0.7 [0.3, 0.95] hit, off by 0.065 | – |
| q6_b_site_exchange_ratio | 2.06 | 1.0 [0.6, 1.6] miss, off by 1.1 | – |
| q7_b_site_rho | -2.47 | -0.6 [-1.2, 0.3] miss, off by 1.9 | – |
| q8_b_site_apples_share | 0.115 | 0.5 [0.1, 0.9] hit, off by 0.39 | 0.5 [0.1, 0.9] hit, off by 0.39 |
| q11a_qwen_a_site_one_currency | False | 15%; Brier 0.02 | 40%; Brier 0.16 |
| q11b_qwen_b_site_one_currency | False | 50%; Brier 0.25 | 40%; Brier 0.16 |
| q11c_qwen_b_site_v5 | comparison | 50% on it; Brier 0.38 | 30% on it; Brier 0.68 |
| q9_llama_usable | True | 60%; Brier 0.16 | – |
| q10_llama_arrival_fraction | 0.5 | 0.65 [0.45, 0.85] hit, off by 0.15 | – |
| q11d_llama_same_b_headline | True | 60%; Brier 0.16 | 70%; Brier 0.09 |

Mean Brier over the probability questions each answered (lower is better): Claude 0.195 (5 questions), Jasper 0.273 (4 questions)
On the probability questions both answered (4): Claude 0.203, Jasper 0.273
- Claude: 6 of 8 80% ranges contain the result
- Jasper: 1 of 1 80% ranges contain the result
