# Predictions versus results

Effects in relative change of the care rate. ✓ = the result fell inside that person's 80% range.

| effect | result | jasper (best, range) | claude (best, range) |
|---|---|---|---|
| Goal in prompt | -80% | -30% (-70 to -7) ✗ | -30% (-60 to -6) ✗ |
| Unmonitored | -5% | -50% (-80 to -20) ✗ | -8% (-24 to +6) ✓ |
| Away from Assistant | -17% | -10% (-50 to +10) ✓ | -16% (-50 to +10) ✓ |
| Toward Assistant | -35% | – | +6% (-8 to +20) ✗ |
| Random push | -41% | -10% (-50 to +10) ✓ | +0% (-12 to +12) ✗ |
| Named individual vs group | +100% | +0% (-40 to +40) ✗ | +10% (-10 to +30) ✗ |
| Versions stuck at floor or ceiling | 4 | 3 (0 to 10) ✓ | 3 (1 to 5) ✓ |
| 10. Harm ladders where care rises with the harm | 6 | 2 (0 to 10) ✓ | 9 (6 to 12) ✓ |
| 10b. Average rise from mildest to most serious harm (points) | +27.4 | 10 (0 to 30) ✓ | 30 (15 to 45) ✓ |

| | jasper | claude |
|---|---|---|
| Results inside the 80% range | 5 of 8 | 5 of 9 |

| question | what happened | jasper: probability given | claude: probability given |
|---|---|---|---|
| The steering manipulation worked as pre-specified | yes | 70% | 70% |
| Same harm type more alike inside the network (6a) | yes | 70% | 65% |
| One shared harm signal (6b) | no | 50% | – |
| Steering away moves harm types differently (9) | yes | 40% | 15% |
| Largest effect | goal | 30% on goal | 60% on goal |

| | jasper | claude |
|---|---|---|
| Brier score (lower is better; 0.25 = always saying 50%) | 0.210 over 5 | 0.254 over 4 |