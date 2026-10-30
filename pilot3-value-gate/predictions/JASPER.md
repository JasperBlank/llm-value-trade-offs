# Jasper's predictions for pilot 3

Written before the run and before seeing Claude's predictions, question by question, on
30 Oct 2026. Ranges are 80% intervals. Unless stated: Qwen3-14B, strangers. At Jasper's
request the gate-pass probabilities come last, after the parts.

## Criterion values
1. Criterion 2, median held-out exchange-rate miss: best guess ×1.5, range ×1.3 to ×1.8 (first given
   as ×1.64; revised while answering the gate questions)
2. Criterion 2, pairs within ×2: best guess 10 of 15, range 7 to 13
3. Criterion 3, transitivity factor of the worse triplet: best guess 1.4, range 1.2 to 1.6
4. Criterion most likely to break first (most failures over 14B, 32B, 8B × both groups): 3; runner-up 2

## Extras
5. Menu invariance passes: 50%
6. Description invariance passes: 60%
7. Loss aversion: strangers best guess 1.58, range 1.18 to 1.98; family best guess 2.45, range 2.05 to 2.85 (pilot 2's values, ranges of width 0.8 centred on them)
8. Closeness multiplier (family ÷ strangers): best guess 2.08, range 1.68 to 2.48 (pilot 2's value, range of width 0.8); uniform: 10%

## Goal module
9. Version 1 (objective on a real good) is rule-like (at 1 against 8): 95% (first given as 80%,
   corrected when question 10 made clear that question 9 was about 1 against 8)
10. Version 1's override holds at 1 against 32: 80%
11. Version 1's override holds at 1 against 128: 50%
12. Version 2 (objective on colour-tagging files) is rule-like: 15%; median welfare price: best guess 4.4 people per file, range 2 to 10 ("the same as last time"; Jasper agreed to this translation)

## The gate (asked last, at Jasper's request)
13. Qwen3-14B passes (all four criteria, both groups): 40%
14. Qwen3-32B passes: 60%
15. Qwen3-8B (negative control) passes: 10%
