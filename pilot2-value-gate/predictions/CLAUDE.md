# Claude's predictions for pilot 2

Sealed: written before Jasper's and not shown to him until he had made his. Copied
unchanged from the sealed file, apart from this note, the title and one file path.

Written 2026-10-29, about 15:25 UTC, after the design in `../DESIGN.md` was
settled and before any pilot-2 code or data existed. Ranges are 80% intervals.

## Tune phase
- Frame B (pure preference) is selected over frame A (allocator): 55%.
- The logarithmic value function fits better than the linear one: 60%.

## The gate (core criteria 1a, 1b, 2 and 3 all pass on direct answers, test goods)

| Model | Probability the core passes |
|---|---|
| Qwen3-4B | 10% |
| Qwen3-8B | 25% |
| Qwen3-14B | 35% |

The 32B borderline rule is triggered: 35%.

Most likely failure point: criterion 1a (every one of the 6 real goods must show quantity
sensitivity), given the quantity neglect seen for money in pilot 1 and in the literature.
Second most likely: 1b, through some pull of numbers on neutral goods.

## Criterion values for Qwen3-14B (test goods, strangers)
- Criterion 2, held-out correlation: 0.85 (0.60 to 0.95).
- Criterion 3, worse of the two real triplets, as a factor: 1.6 (1.2 to 3.0).
- Number of the 6 real goods with a quantity slope reliably above zero: 5 (3 to 6).

## Extras for Qwen3-14B
- Menu invariance passes: 35%.
- Description invariance passes: 40%.
- Loss aversion (value of preventing a loss ÷ value of the equivalent gain): 1.3 (0.8 to 2.0).
- Closeness multiplier (family ÷ strangers): 1.3 (0.9 to 2.5).
- Closeness is a uniform multiplier (every good within a factor of 1.5 of the common multiplier): 45%.

## Addendum: goal module (written 2026-10-29, about 23:58 UTC, before Jasper's predictions for it)

For Qwen3-14B, strangers:
- Version 1 (objective on a real good) is rule-like by the design's definition: 35%. Most
  likely it acts as large extra value: a value-shift factor of 4 (2 to 20), with the quantity
  slope keeping about 60% of its size (30% to 100%).
- Version 2 (objective on re-labelling archive boxes) is rule-like: 50%.
- Version 2's welfare price, the number of real-good beneficiaries that one re-labelled box
  outweighs under the objective: 3 (1 to "above 8").

Reasoning: in pilot 1 the goal line cut care from about 67% to 13%, which is strong but not
absolute, and care still rose with the harm. That looks like a large weight rather than a
rule. A meaningless target has no competing value of its own to fall back on, which is why I
give version 2 a higher chance of looking rule-like.

## Reasoning in brief
Forced-choice formats tend to look fairly coherent (as in Utility Engineering), and reading
exact probabilities removes sampling noise, so criteria 2 and 3 may well pass at 14B. The
strict "all six real goods" rule in 1a, and the pull of numbers on neutral goods, are where
I expect intuitive answers from small models to break. Coherence should rise with size.
