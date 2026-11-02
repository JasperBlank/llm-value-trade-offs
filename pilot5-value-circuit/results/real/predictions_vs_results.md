# Pilot 5: predictions against results (main set)

| question | result | Claude | Jasper |
|---|---|---|---|
| q1_a_slope_28 | 0.0693 | 0.55 [0.15, 1.3] miss, off by 0.48 | 0.1 [0.02, 0.2] hit, off by 0.031 |
| q2_b_slope_28 | -0.0792 | -0.45 [-1.2, -0.1] miss, off by 0.37 | -0.5 [-0.9, -0.25] miss, off by 0.42 |
| q3_controls_specific_28 | False | 65%; Brier 0.42 | 80%; Brier 0.64 |
| q4a_layer_largest_a_slope | 36 | 32 wrong | 36 right |
| q4b_full_size_any_layer | False | 45%; Brier 0.20 | 50%; Brier 0.25 |
| q5_cos_wa_wb_28 | 0.00956 | 0.3 [-0.4, 0.8] hit, off by 0.29 | 0.7 [0.5, 1.0] miss, off by 0.69 |
| q6a_currency_gain_vs_harm | True | 55%; Brier 0.20 | 40%; Brier 0.36 |
| q6b_currency_money_vs_other | True | 50%; Brier 0.25 | 40%; Brier 0.36 |
| q7_mlp_share_a_28 | 0.445 | 0.5 [0.2, 0.8] hit, off by 0.055; above half 50%, Brier 0.25 | 0.5 [0.1, 0.9] hit, off by 0.055; above half 50%, Brier 0.25 |
| q8_pieces_for_half | 7 | 12 [4, 60] hit, off by 5 | 50 [3, 200] hit, off by 43 |
| q9_first_layer_half_transfer | 28 | 28 right | 35 [20, 40] hit, off by 7 |
| q10_top20_transfer | 0.654 | 0.25 [0.05, 0.6] miss, off by 0.4 | 0.5 [0.2, 0.9] hit, off by 0.15 |
| q11_letter_head | True | 60%; Brier 0.16 | 30%; Brier 0.49 |
| q12_number_and_good_in_one_piece | True | 60%; Brier 0.16 | 40%; Brier 0.36 |
| q13_goal1_fraction_28 | 0.32 | 0.12 [0.02, 0.45] hit, off by 0.2 | 0.1 [0.05, 0.8] hit, off by 0.22 |
| q14_verdict_28 | not_specific | 18% on it; Brier 0.93 | 40% on it; Brier 0.48 |

Mean Brier over the probability questions (lower is better): Claude 0.323 (8 questions), Jasper 0.399 (8 questions)
- Claude: 4 of 7 80% ranges contain the result
- Jasper: 6 of 8 80% ranges contain the result

# The formats extension

| question | result | Claude | Jasper |
|---|---|---|---|
| f1_rating_behaviour_corr | 0.614 | 0.55 [0.2, 0.85] hit, off by 0.064 | 0.5 [0.1, 0.9] hit, off by 0.11 |
| f2_rating_option_r | 0.549 | 0.55 [0.0, 0.9] hit, off by 0.00096 | 0.2 [0.05, 0.6] hit, off by 0.35 |
| f3_numbered_relative_to_ab | 0.815 | 0.7 [0.2, 1.3] hit, off by 0.12 | 0.2 [0.1, 0.5] miss, off by 0.62 |
| f4_should_push_raises_yes | True | 70%; Brier 0.09 | 30%; Brier 0.49 |
| f5_rating_push_raises_rating | False | 60%; Brier 0.36 | 50%; Brier 0.25 |
| f6_writers_overlap_top20 | 15 | 13 [7, 18] hit, off by 2 | 5 [1, 18] hit, off by 10 |

Mean Brier (F4, F5): Claude 0.225, Jasper 0.370
