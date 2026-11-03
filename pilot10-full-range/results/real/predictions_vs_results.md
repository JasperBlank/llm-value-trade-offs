# Pilot 10: predictions against results

| question | result | Claude | Jasper |
|---|---|---|---|
| q1_full_range_r | 0.81 | 0.82 [0.65, 0.92]: hit, off by 0.0097 | 0.7 [0.4, 0.9]: hit, off by 0.11 |
| q2_logscale_r | 0.50 | 0.68 [0.5, 0.85]: hit, off by 0.18 | 0.4 [0.2, 0.8]: hit, off by 0.1 |
| q3_size_at_least_one_point | False | 40%; Brier 0.160 | 10%; Brier 0.010 |
