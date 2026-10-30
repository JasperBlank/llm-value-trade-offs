# Claude's predictions for pilot 3

Sealed: written before Jasper's and not shown to him until he has made his. Written on
30 Oct 2026, about 14:15 UTC, after the pilot-3 design and code were final and before any
pilot-3 data existed. Ranges are 80% intervals. Unless stated: Qwen3-14B, strangers.

## The gate (all four criteria, in both beneficiary groups)
- Qwen3-14B passes: 40%.
- Qwen3-32B passes: 50%.
- Qwen3-8B (negative control) passes: 10%.
- Criterion most likely to break first (most failures over 3 models × 2 groups): 2.

## Criterion values (14B, strangers)
- Criterion 2, median held-out exchange-rate miss: ×1.35 (×1.10 to ×2.20).
- Criterion 2, pairs within ×2: 11 of 15 (7 to 15).
- Criterion 3, transitivity factor of the worse triplet: 1.35 (1.10 to 2.50).

## Extras (14B, strangers)
- Menu invariance passes: 30%.
- Description invariance passes: 55%.
- Loss aversion: 1.5 (0.8 to 3.0).
- Closeness multiplier (family ÷ strangers): 2.0 (1.2 to 4.0); uniform: 35%.

## Goal module (14B, strangers)
- Version 1 (objective on a real good) is rule-like: 85%.
- Version 1's override holds even at 1 against 32: 65%.
- Version 2 (objective on colour-tagging files) is rule-like: 15%.
- Version 2's median welfare price: 3 people per file (0.7 to "above 32").

## Reasoning in brief
Re-analysing pilot 2 with order averaging, 14B passed criteria 1a, 1b and 3 in both groups
and criterion 2 for strangers (median ×1.35) but not for family (×1.70). The two-stage grid
should make the pair rates more precise, which removes noise but not real inconsistency, and
the goods are new. So 14B is close to a coin flip, a bit below because both groups must pass.
32B should be somewhat more coherent. 8B answered mostly by position; averaging turns that
into near-50% cells with noisy rates, so it should fail. The goal results follow pilot 2: an
objective on a real good was a total override (12 of 12 pairs); an objective on a meaningless
good won against some goods (vaccinations, days off) but not others (scam prevention), giving
a median price of about 4 people; the money harm (a wrongful fine) may resist again.
