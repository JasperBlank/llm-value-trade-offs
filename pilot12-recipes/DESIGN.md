# Pilot 12: one vibe, or shared ingredients with different recipes?

## 1. The question

In pilot 11, one direction on a sentence's own words reads its menu price (held-out r 0.78). Read in the rating prompt, it predicts the rating r 0.64. That is about what a noisy copy of the price would give (0.79 × 0.81), so pilot 11 could not tell whether price and rating share one variable. Pushing along it moved the choice by about 40% of the predicted amount and the rating by about 15%.

Jasper's question: if there is no shared direction, why do price and rating overlap so strongly (r 0.81)? Two pictures:

- **One vibe.** The model computes one variable per sentence. The choice and the rating both read it, each with its own noise and its own bends (the rating's pile-up at 7, its ceiling at 9). Where they disagree is noise.
- **Shared ingredients, different recipes.** Both are built from the same features of the sentence (above all, how big the outcome is) but weigh them differently. Pilot 10's ratings lean towards moral and protective acts; its prices lean towards lasting and meaningful ones. Where they disagree is systematic.

Under "one vibe", the disagreement cannot be read from the sentence, and pushing along any direction moves the rating only as far as it moves the choice. Under "different recipes", the disagreement can be read, and pushing along it moves the rating beyond what the choice moves.

## 2. Material

- **Model:** Qwen3-14B (bf16, thinking off), with pilot 10's 96 sentences.
- **Price u:** pilot 10's menu price.
- **Rating R:** the mean of pilot 10's three "goodness" wordings (W0-W2: how good, how much better the world, how bad or good) at 8 people. The mean is more reliable than one wording and averages out each wording's own quirks.
- **The disagreement d:** the rating minus a smooth function of the price. This is the residual of R on a cubic in u, fitted over the 96 sentences. Removing a smooth function, not just a straight line, matters: the rating bends (7, 9) would otherwise look like disagreement that can be read from the price itself.
- **States:** as in pilot 11. The residual stream after blocks 0, 4, …, 40, averaged over the sentence's own tokens, in the **menu context** (the sentence as option A, before B; the model knows nothing about any rating question) and the rating context (W0 prompt).

## 3. Pre-registered questions

| | Question | Measure | Pass |
|---|---|---|---|
| **Q1** | Is the disagreement systematic: can it be read from the sentence? | A ridge readout of d from the **menu-context** states, leave one value out (penalty chosen by an inner leave-one-value-out), at the best of layers 8-32. Significance from a permutation test (d shuffled across sentences, 200 times, the whole procedure including the choice of layer repeated each time) | held-out r ≥ 0.3 **and** permutation p < 0.05 |
| **Q2** | Does pushing along the disagreement move the rating beyond the choice? | At layer 20, push along four directions (difference of means between top and bottom thirds, menu context, all 96) by the same amount: price (by u), rating (by R), disagreement (by d), random. Each push is ±δ, where δ = the residual distance for 2 price units along the price direction (as in pilot 11). For each direction: C = change in logit P(A) (each sentence as option A against its 4 nearest-priced partners, 8 vs 8) and R = change in the W0 expected rating (8 people), both per unit of push. The price direction gives the rating-per-choice ratio s = R_price / C_price. The disagreement direction's **excess** is R_dis − s × C_dis: how far it moves the rating beyond what its effect on the choice would carry through a shared variable | excess > 0 with the 95% bootstrap interval (2,000 resamples of sentences) above 0 |

**Verdict:**
- **"Different recipes"** if Q1 and Q2 pass.
- **"One vibe"** if neither passes.
- **"Mixed"** otherwise.

## 4. Also reported (not pre-registered verdicts)

- Q1 in the rating context, and with the difference-of-means direction.
- **A small set of directions (PLS):** partial-least-squares readouts with k = 1, 2, 3 and 5 components, learned jointly on price and rating (menu context, leave one value out).
  - For each k: how well price and rating are predicted, and the partial correlation of price and rating once both predictions are accounted for.
  - Caution: with 96 sentences, every held-out readout is noisy, so this partial correlation cannot reach zero even if the overlap does live in those k directions.
- The excess for the rating direction and the random direction. All pushes' C and R, in both units.
- Which values the disagreement readout puts highest and lowest (held out).

## 5. Running it

- One A40 pod, about 10 minutes, about $0.10.
- With the `github_push_token` secret, the states and results are pushed to `pod-results/pilot12`. Otherwise the report is printed to the log.
- Predictions: Claude's sealed first, then Jasper's (`predictions/`).
