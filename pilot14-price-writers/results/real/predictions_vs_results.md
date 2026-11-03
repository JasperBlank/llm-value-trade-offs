# Pilot 14: predictions against results

| question | result | Claude | Jasper |
|---|---|---|---|
| q1_few_words_prob | False | 40%; Brier 0.160 | 61%; Brier 0.372 |
| q2_readout_tracks_r | 0.28 | 0.5 [0.25, 0.75]: hit, off by 0.22 | 0.8 [0.5, 0.9]: miss, off by 0.52 |
| q3_top10_share | 0.52 | 0.6 [0.35, 1.0]: hit, off by 0.085 | 1.0 [0.0, 2.0]: hit, off by 0.48 |
| q4_writers_used_prob | False | 45%; Brier 0.203 | 62%; Brier 0.384 |
