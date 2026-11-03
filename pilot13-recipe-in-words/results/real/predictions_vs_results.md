# Pilot 13: predictions against results

| question | result | Claude | Jasper |
|---|---|---|---|
| p1_features_r | 0.79 | 0.75 [0.55, 0.88]: hit, off by 0.04 | 0.5 [0.1, 0.9]: hit, off by 0.29 |
| p2_moral_rating_prob | False | 55%; Brier 0.303 | 80%; Brier 0.640 |
| p3_beats_random_prob | True | 50%; Brier 0.250 | 80%; Brier 0.040 |
| p4_new_sentences_r | 0.71 | 0.65 [0.35, 0.85]: hit, off by 0.064 | 0.5 [0.1, 0.9]: hit, off by 0.21 |
