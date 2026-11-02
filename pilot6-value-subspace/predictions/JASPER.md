# Jasper's predictions for pilot 6

Given one question at a time on 2 Nov 2026, 12:37-12:51 Amsterdam time, after Claude's predictions
were sealed (hash in `SEALED.md`) and while the run's outputs were hidden on the pod. The final
values are in `predictions.yaml`.

| Question | First answer | Final |
|---|---|---|
| 1. Quantity, half-transfer layer | 5 (2-20); changed before question 2 | 20 (10-30) |
| 2. Weight, half-transfer layer | 22 (12-32) | same |
| 3. Quantity, top-20 transfer | 0.65 (0.4-0.8) | same |
| 4. Weight, top-20 transfer | 0.65 (0.4-0.8) | same |
| 5. One learned direction, share of the whole swap | 0.5 (0.1-0.9) | same |
| 6. Smallest k for 0.9 | 8 | same |
| 7. The probe as a direction, share | 0.5 (0.1-0.9) | same |
| 8. Cosine learned vs probe (size) | 0 (-0.5 to 0.5, i.e. 0-0.5 in size) | same |
| 9. Correlation with A's value, held out | 0 (-0.5 to 0.5) | 0.5 (0.3-0.8) |
| 10. Push ratio learned / probe | 2 (0.5-4) | 3 (0.5-4) |
| 11a. P1: generic / value-specific / mixed | 30 / 40 / 30% | 30 / 20 / 50% |
| 11b-d. P2 / P3 / P4 pass | 60 / 60 / 60% | 50 / 50 / 50% |

**Consistency check.** At Jasper's request, after all answers Claude pointed out where they did not
fit together (P4 vs question 9, P3 vs question 10, the P1 split vs questions 1-4, questions 5/7/8,
P2 vs question 5, questions 1 and 3). Jasper then changed questions 9, 10 and 11 as above and kept
the rest. Both versions are kept here; scoring uses the final values.
