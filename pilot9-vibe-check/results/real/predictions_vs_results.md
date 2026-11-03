# Pilot 9: predictions against results

| question | result | Claude | Jasper |
|---|---|---|---|
| t1_rating_vs_menu_r | 0.55 | 0.55 [0.3, 0.78]: hit, off by 0.00067 | 0.8 [0.4, 0.9]: hit, off by 0.25 |
| t2_same_sentence_count | 20 | 15 [11, 19]: miss, off by 5 | 15 [12, 22]: hit, off by 5 |
| t3_size_at_least_half_point | False | 35%; Brier 0.122 | 60%; Brier 0.360 |
