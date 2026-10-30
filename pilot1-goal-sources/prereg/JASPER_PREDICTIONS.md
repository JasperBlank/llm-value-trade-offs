# Jasper's predictions

Recorded during the prediction session on 2026-10-28, before any real model was run.
Effects are RELATIVE: percentage change of the care rate compared with the reference
condition (e.g. -30% means a care rate of 60% becomes 42%). Ranges are 80% intervals.

| Quantity | Reference condition | Best guess | 80% range |
|---|---|---|---|
| Goal effect (unsteered) | no goal in prompt | -30% | -7% to -70% |
| Unmonitored effect (unsteered) | monitored | -50% | -20% to -80% |
| Away from the Assistant | no steering | -10% | +10% to -50% (widened from -5% to -50% after discussion) |
| Random direction, same size | no steering | -10% | +10% to -50% |
| Individual (named Tomas) vs group, supplier versions | group | 0% | -40% to +40% |

Setup:
- 7. The steering manipulation works as pre-specified (away lowers AI self-identification by at least 20 points while answers stay readable): 70%.
- 8. Versions stuck near 0% or 100% care after calibration: 3 of 13 (80% range: 0 to 10).

Harm directions inside the network:
- 6a. Versions of the same harm type are more similar to each other than to other harm types: 70%.
- 6b. All harm types share one signal (average similarity above 0.3): 50%.

Which source has the largest effect: monitoring 50%, goal 30%, character 20%.

Implication noted at the time: with random as large as away, Jasper expects the
pre-registered specificity rule for the Assistant direction to fail (any drop in care
comes from disturbing the model in general, not from removing its assistant character).

Baseline care rates: no confident guess (very wide uncertainty).

Values or behaviours (added 2026-10-29, recorded at 07:11 and 07:15 Europe/Amsterdam, before
step 3 was run):
- 9. Steering away from the Assistant moves some harm types more than others (the
  pre-registered permutation test is supported): 40%. Jasper expects a real value profile,
  separate from a single care dial; the 40% allows for the test's low power. His own
  uncertainty about that probability: 10% to 70%.
- 10. Of the 13 harm ladders, how many the test classifies as rising: 2 (80% range 0 to 10).
  In his words: care will rise, but only slightly, not a lot.
- 10b. Average rise in care from the mildest to the most serious harm level: +10 points
  (80% range 0 to +30).

What would change my mind (in Jasper's words):
1. A result that would make me drop this direction: if I cannot understand any part of what
   happened by the end, and if every scenario seemed not to impact the "care" at all.
2. A result that would make me most excited to continue: if after this we have clear evidence
   against one of the theories of causal drives behind care/goal-seeking.
3. A result I would find hard to explain: if every answer fell outside my ranges.
