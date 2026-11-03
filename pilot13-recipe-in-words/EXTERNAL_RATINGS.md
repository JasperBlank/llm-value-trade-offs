# Exploratory: the features rated by a different model

Added on 3 Nov 2026 at about 10:36 Amsterdam time, while the pilot-13 run was in progress and before any of its results were seen. Jasper's suggestion.

## Why

The ten features in pilot 13 are Qwen's own answers. If Qwen rates an outcome as more serious or more meaningful simply because it likes it (a halo), then the recipe in words partly explains the price with the price.

## The check

Two fresh Claude instances rate the same 120 sentences on the same ten questions, with the same 0-9 scale and labels. One instance takes F1-F5, the other F6-F10.

They see only a shuffled list of the sentences: no prices, no ratings, no values, no results, and nothing else in the repo. They work one question at a time across all sentences, and give whole numbers.

**Difference from Qwen's prompt:** the raters are told to answer each question literally, so that how good an outcome is does not affect a rating unless the question asks about it. Qwen was not told this.

## Analysis (fixed now, before the results)

1. **P1 and P4 with Claude's ratings:**
   - the held-out r (one value left out) of the price from F2-F10 on the 96 sentences;
   - the r for the 24 new sentences' prices, from the recipe fitted on the 96.
   - Also with F1 added, and F1 alone.
2. **The halo check:** the mean correlation among F2-F10, in Qwen's ratings and in Claude's.
3. **Agreement:** Qwen's and Claude's ratings correlated feature by feature, over the 120 sentences.
4. **The two recipes** (the standardised weights of F2-F10 in the price and in the rating) with Claude's ratings, set against Qwen's. This includes the moral weight difference, with the same bootstrap as P2.

None of this has a pass bar. It is reported as exploratory, alongside the pre-registered results.

## Files

- `external/outcomes_to_rate.json`: the blind list the raters saw.
- `external/ratings_part1.csv` and `external/ratings_part2.csv`: their ratings, as written.
- `external13.py`: the analysis above, which writes `results/real/external_ratings.md`.
