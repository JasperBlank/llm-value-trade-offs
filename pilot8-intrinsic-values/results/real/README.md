# Pilot 8 results (Qwen3-14B, run 3 Nov 2026)

**The run.**

- It used one GPU pod on Runpod. Both prediction sets were committed before any result was printed (hashes checked on the pod).
- The pod stopped itself after printing. A stopped pod's log can no longer be read, so the pod was restarted without a GPU to print the saved log. The files here were rebuilt from that log; `log_pod.txt` is the log itself.
- `behaviour_pA.npy` holds the model's P(A) for all 13,248 trade-offs, in the order of `behaviour.jsonl` (float16). The menu answers were not saved, so the menu effect cannot be re-analysed (see the cautions).

| File | What |
|---|---|
| `report.md` | The report from the pod |
| `verdicts8.json` | The pre-registered verdicts |
| `qwen/behave.json` | Q1, Q3, Q4a: prices, menu checks, stated ratings |
| `qwen/source.json`, `qwen/worth_direction.npy` | Q2a: the worth direction on option A's own words |
| `qwen/compare.json`, `qwen/comparison_direction.npy` | Q2b: the comparison direction after option B |
| `qwen/rating.json` | Q4b, Q4c: the worth direction in the "how much does it matter?" prompts |
| `qwen/behaviour_pA.npy` | The model's answer to every trade-off |
| `predictions_vs_results.md` | Both prediction sets scored (`score_predictions.py`) |
| `exploratory_sentences.md`, `.json` | Exploratory, after the results: prices per sentence (`explore_sentences.py`) |

Figures: [`fig5_price_list.png`](../../../figures/fig5_price_list.png), [`fig6_sentences_not_values.png`](../../../figures/fig6_sentences_not_values.png), [`fig7_pilot8_tests.png`](../../../figures/fig7_pilot8_tests.png).

## Jasper's four questions

**1. How does the model trade the values off?** It has a clear price list.

- Fitted from 13,248 two-option choices, the prices order the answers with r = 0.76. One extra log-person is worth b = 1.59 logits.
- Top: Freedom, Protection, Nurturing (one person counts as 3.0, 2.9 and 2.4 people of an average value).
- Bottom: Happiness (0.31), Longevity (0.44), Truth (0.46), Pleasure, Justice.
- It leans towards answering A by 0.98 logits.

**2. Does the same thing light up across trade-offs?** Yes for the comparison, no for a worth code before it.

- **Q2b passes.** After option B (layer 24; the answer position takes over at layer 28), one direction learned on trade-offs among 12 values carries 0.82 of the site's effect on trade-offs among the 12 other values. It carries 0.70 of the whole effect of swapping one value for another.
- **Q2a fails.** On option A's own words, the best site carries only 0.12 of a value swap. A worth direction learned there orders the 12 held-out values with r = 0.26 (the training values: 0.65).

**3. Is the menu consistent?** Partly.

- Transitive: 61 of 2,024 triads are cyclic (3.0%; pass).
- One scale fails narrowly: prices fitted on one question wording predict the other with r = 0.73 (bar 0.8).
- Menu invariance fails badly: adding a third option changes the A:B odds by a median factor of 70 (bar 1.5). See the cautions.

**4. When asked how much one value matters, does the same representation show up?** No.

- **Q4a fails.** Stated ratings and trade-off prices agree r = 0.54. Longevity is the clearest match (rated 2.0 of 9, priced 23rd). Defending (5.9) and Justice (5.6) are rated highly but priced in the middle and near the bottom.
- **Q4b fails.** In the rating prompts, the trade-off worth direction orders the 12 held-out values with r = 0.12.
- **Q4c fails.** Swapping only that direction carries 0.10 of the rating site's effect (bar 0.3).

## Exploratory: prices of sentences, not values

This was not pre-registered. It uses the model's saved answers.

- Each value was written as two sentences. Priced separately, **the two sentences of one value agree only r = 0.20**.
- **The same sentence is priced consistently**: fitted separately on the two question wordings, r = 0.91.
- So the model's prices are stable, but they belong to the specific situations, not to the value categories. For example, Diversity as "become part of a community made up of many different kinds of people" is priced +2.2; as "get to know someone very different from themselves", −2.6. Non-suffering as "spared a week of pain" is +2.5; as "relieved of a painful condition for a week", −0.7.
- The value-level price list above averages two sentences per value. It should be read as the prices of those 48 sentences, not of "Freedom" or "Happiness" in general.

## Cautions

- **Menu effect.** The factor of 70 is too large to be a real decoy effect. Most likely the three-option format itself moves the answer: the mean shift is −3.9 log-odds against A. The menu answers were not saved, so this cannot be checked against option order.
- **The A site is weak.** Option A's own words carry only 0.12 of a value swap at the best site. This is consistent with pilot 7: most of the information gathers after option B. So Q2a tested a direction at a site that carries little. A per-value worth code could exist elsewhere before the comparison, but not at A's own words.
- **Q2a and Q4b use value-level prices** as the target. Since the two sentences of a value disagree, a direction tracking sentence prices would score low here even if it existed. A sentence-level test needs the GPU again.
- One model (Qwen3-14B), one answer format, two sentences per value, two question wordings.

## Reading

The model behaves as if it has a stable price for each situation: transitive, and consistent across question wordings. But the intrinsic-value categories are not what sets those prices. Two descriptions of the same value can sit at opposite ends of the list.

Inside the model, the clearest shared structure is the comparison. After both options are read, one direction carries most trade-offs, including between values it was never trained on. There is no transferable per-value worth code on option A's own words. The value the model states when asked directly ("how much does it matter?") is only loosely related to its trade-off prices and does not use the trade-off worth direction.
