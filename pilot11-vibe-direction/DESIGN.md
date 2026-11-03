# Pilot 11: one direction for both?

## 1. The question

Pilot 10 found that Qwen3-14B has a stable price for each sentence: the same sentences priced in two very different menus agree r = 0.98. The model's direct "How good would it be if …?" rating overlaps with that price, r = 0.81 over the full range, but only about 0.5 among outcomes of similar size.

Jasper's question: **is there a direction inside the model that predicts this overlap?** If the model does one "vibe check" of each sentence that feeds both the choice (the menu) and the stated rating, there should be a direction that:

1. can be read from the sentence itself, before the model has seen the other option;
2. predicts both the menu price and the rating, and accounts for what they share;
3. moves both the choice and the rating when the model is pushed along it.

Gilg et al. (2026) found such a direction for choices in other models (a linear readout at the end of a task predicts choice-based utilities, and steering along it flips choices). They did not test stated ratings.

## 2. Material

- **Model:** Qwen3-14B (bf16, thinking off).
- **Sentences:** pilot 10's 96 sentences, with their menu prices (pilot 10: 45,600 trade-offs, split-half reliability 1.00) and their W0 ratings at 8 people ("How good would it be if the following happened?", expected rating; pilot 10).
- **Two contexts for each sentence, both at 8 people:**
  - **Menu context:** the sentence as option A of a pilot 10 trade-off prompt. The model has not seen option B at A's words (it reads left to right), so B does not matter for reading A's state.
  - **Rating context:** pilot 10's W0 rating prompt.
- **Reading the state:** the residual stream after blocks 0 (embeddings), 4, 8, …, 40, averaged over the sentence's own tokens (the words after "8 people"). The sentence's last token is also reported.

## 3. The direction

- **The vibe direction at a layer** is the difference of means between the 32 highest-priced and the 32 lowest-priced sentences (thirds by menu price), in the menu context. This is a "mass-mean" direction, chosen because such directions tend to be the ones a model actually uses (better for steering than regression weights).
- **A sentence's reading** is its state projected onto that direction, scaled to price units by a straight-line fit on the training sentences.
- **Held-out:** leave one value out. The direction is learned on 23 values (92 sentences) and read on the 4 sentences of the 24th, all the way round.
- **Reported alongside:** a ridge readout (standardised features, penalty chosen by an inner leave-one-value-out), and readouts trained on the rating instead of the price.
- **The layer for Q2 and Q3** is the one where the held-out reading of the price (menu context, mean over the sentence) is best (Q1).

## 4. Pre-registered questions

| | Question | Measure | Pass |
|---|---|---|---|
| **Q1** | Can the price be read from the sentence, before option B? | held-out r between the vibe reading (menu context) and the menu price, best layer | r ≥ 0.7 |
| **Q2** | Does the same direction carry the overlap? | read in the **rating** context with the direction learned in the menu context (held-out): r with the rating, and the partial correlation of price and rating once the reading is accounted for (raw r = 0.81) | r ≥ 0.6 **and** partial r ≤ 0.3 |
| **Q3** | Does the model use it for both? | push along the direction (learned on all 96, at the Q1 layer) over the sentence's words, by ±2 price units | see below |

**Q3 in detail:**
- **Menu:** each sentence as option A against its 4 nearest-priced sentences of other values (8 vs 8 people). The change in logit P(A) per price unit pushed is the **menu transfer** (1 = the full predicted amount).
- **Rating:** each sentence's W0 rating prompt (8 people). The change in the expected rating, as a share of 0.327 × the push (0.327 rating points per price unit is pilot 10's slope of rating on price), is the **rating transfer**.
- **Control:** a random direction pushed by the same amount.
- **Pass:** menu transfer ≥ 0.5 **and** rating transfer ≥ 0.5, while both are at most 0.1 in size for the random direction.

**Verdict:**
- **"One vibe direction feeds both"** if Q1-Q3 all pass.
- **"Readable but not shared"** if Q1 passes and Q2 fails.
- **"Shared but not used"** if Q1 and Q2 pass and Q3 fails.
- **"No readable price on the sentence"** if Q1 fails.

## 5. Also reported (not pre-registered verdicts)

- Every layer: held-out r for price and rating, in both contexts, mean over the sentence and last token, vibe direction and ridge.
- The cosine between the price direction and a rating direction (difference of means by rating thirds, rating context).
- Within stakes level (small, medium, large): the same correlations and partial correlation.
- A "disagreement" direction (difference of means by standardised rating minus price): is it readable on held-out values?
- Steering at other layers (4 to 36), and the effect on the log-scale score.

## 6. Running it

- One A40 pod. Reading the states takes seconds; steering takes about 4,000 forward passes. About 10-15 minutes, about $0.15.
- With the `github_push_token` secret, the results are pushed to `pod-results/pilot11`. Otherwise the report is printed to the log.
- Predictions: Claude's sealed first, then Jasper's (`predictions/`).
