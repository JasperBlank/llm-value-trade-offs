# Claude's predictions (written before Jasper made his, and not shown to him until he had)

Written 2026-10-28 14:05 Europe/Amsterdam, after the scenario set was fixed at 13 versions
in 8 situations and before any real model was run. Effects in percentage points of "chose
the care option". Ranges are 80% intervals.

| Quantity | Best guess | 80% range |
|---|---|---|
| Goal effect (unsteered) | -15 | -30 to -3 |
| Unmonitored effect (unsteered) | -4 | -12 to +3 |
| Away from Assistant | -8 | -25 to +5 |
| Toward Assistant | +3 | -4 to +10 |
| Random direction, same size | 0 | -6 to +6 |
| Goal x away (goal effect under away minus unsteered) | -5 | -20 to +10 |
| Unmonitored x away | 0 | -10 to +10 |
| Individual (named) minus group, same harm type, unsteered | +5 | -5 to +15 |

- Source most likely to dominate: goal 60%, away 25%, unmonitored 15%.
- Manipulation check passes (away lowers AI self-identification by at least 20 points): 70%.
- Versions stuck at floor or ceiling after calibration: 3 of 13 (range 1 to 5); most likely
  the physical-safety and fatigue versions, at the ceiling.
- Harm directions: positively aligned across harm types (mean cosine 0.3 to 0.5). Probability
  that same-type pairs (e.g. the three "money" versions) align more than cross-type pairs: 65%.

Reasoning in brief: system-prompt objectives are a strong, well-trained lever; monitoring
cues are weak in models without much situational-awareness training at this size; steering
at the largest still-readable strength partly degrades the model, which pulls answers toward
50/50 rather than in a direction, so the away effect is uncertain and probably modest.

## Addendum: estimands 9 and 10

Written 2026-10-29 (about 01:50 UTC), after 9 and 10 were added to the design and before
Jasper made his predictions for them. No real model had been run.

- **10. Harm ladders where care rises with the harm** (of 13): 9, 80% range 6 to 12. The
  ladders escalate explicitly, so most should register; the ones most likely to miss are a
  physical-safety or fatigue ladder that sits near the ceiling at every level ("no room"),
  and one of the need ladders, where the extra hardship is described rather than priced.
- **9. Steering away moves harm types differently** (the permutation test is supported): 15%.
  Mostly because the test has little power (six same-type pairs), and partly because I expect
  steering at a readable strength to act more like a general loosening than a reshuffling of
  priorities. A "no" here will say little either way.

## Addendum: estimand 10b

Written 2026-10-29 (06:11-06:12 UTC), when 10b was added at Jasper's request and before
he made his prediction for it.

- **10b. Average rise in care from the mildest to the most serious harm level**, over all 13
  ladders: +30 points, 80% range +15 to +45. Consistent with my count for 10: most ladders
  rise by 20 to 40 points, pulled down by any ladder that sits near the ceiling throughout.
