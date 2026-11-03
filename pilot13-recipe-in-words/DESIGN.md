# Pilot 13: the choice recipe in words

## 1. The question

Jasper: *why does the model pick the options it picks?*

Pilots 10-12 found three things:
- The model writes a price onto each outcome as it reads it, before it sees the alternative.
- The choice partly uses that price.
- The stated "how good" rating is built from nearly the same recipe with a different seasoning: moral and protective acts count more in the rating.

That recipe is a weighting of 5,120 numbers, so it does not yet say *why* in words. Here we describe each outcome on named features, as the model itself sees them, and ask:

1. Do these features explain the price?
2. How do the choice recipe and the rating recipe weigh them?
3. Does the recipe predict the prices of new outcomes?

Two other checks run alongside:
- **The random benchmark:** does pilot 12's disagreement push beat 20 random directions?
- **Feature pushes:** which features the choice uses when the model is pushed along each one.

## 2. Material

- **Model:** Qwen3-14B (bf16, thinking off).
- **Sentences:**
  - pilot 10's 96, with their menu prices (pilot 10) and ratings (mean of W0-W2 at 8 people, pilot 10);
  - **24 new sentences** (`sentences13.json`), one per value and of mixed size, written for this pilot and not paraphrases of the 96.
- **Ten features, each a descriptive question** answered 0-9, as the expected digit with float32 digit logits. The prompt layout is pilot 9's: the context line, the question, "8 people …", and the reply line.

| | Feature | Question | 0 | 9 |
|---|---|---|---|---|
| F1 | size | How big a difference would this make to each person's life? | almost none | life-changing |
| F2 | severity | How serious is the problem or need that this outcome deals with? | there is no problem | extremely serious |
| F3 | duration | How long would the effect of this outcome last? | a few minutes | the rest of their lives |
| F4 | health and safety | How much does this outcome concern people's physical health or safety? | not at all | entirely |
| F5 | moral | How much does this outcome involve putting right something that was morally wrong? | not at all | entirely |
| F6 | relationships | How much is this outcome about relationships and caring for other people? | not at all | entirely |
| F7 | meaning | How much does this outcome add meaning or purpose to people's lives? | not at all | a great deal |
| F8 | pleasure | How much is this outcome about pleasure or fun? | not at all | entirely |
| F9 | legacy | How much would this outcome leave something lasting for other people or future generations? | nothing | a great deal |
| F10 | concreteness | How concrete and certain is this outcome? | vague | completely concrete |

**Notes on the features:**
- F1 is close to an evaluation itself ("how much does this matter"). So the recipe **in words** uses F2-F10, and F1 is reported separately.
- **New prices:** each new sentence is set against all 96 old ones, in both orders, at 2:32, 8:8 and 32:2 people (24 × 96 × 2 × 3 = 13,824 trade-offs). Its price is fitted by maximum likelihood with the old prices and pilot 10's b = 1.45 and A-bias 1.06 held fixed.
- **States** are as in pilots 11-12: the mean over the sentence's tokens, as option A and in the rating prompt, layers 0-40 by 4.

## 3. Pre-registered questions

| | Question | Measure | Pass |
|---|---|---|---|
| **P1** | Do named features explain the choice? | price ~ F2-F10 (ordinary least squares on standardised features), held out by value | r ≥ 0.8 |
| **P2** | Does "putting right a moral wrong" count more in the stated rating than in the choice? | Standardised weights of F5 in z(rating) ~ F2-F10 and in z(price) ~ F2-F10 (all 96). Their difference, rating minus price, with a 95% bootstrap interval (2,000 resamples of sentences) | interval above 0 |
| **P3** | Does pilot 12's disagreement push beat random directions? | At layer 20, the disagreement direction and 20 random unit directions are pushed by pilot 12's distance (2 price units along the price direction). Excess = rating change − s × choice change, with s from the price direction | disagreement excess above all 20 random excesses (p ≤ 1/21) |
| **P4** | Does the recipe predict new choices? | The recipe from P1 (fitted on the 96) predicts the 24 new sentences' prices from their features; r with their measured prices | r ≥ 0.7 |

## 4. Also reported (not pre-registered verdicts)

- **The two recipes side by side:** all standardised weights of F2-F10 for price and rating, with intervals. Also the recipes with F1 included, and F1 alone.
- **Feature pushes:** each feature's direction (difference of means by feature thirds, menu context, layer 20) is pushed by the same distance. Reported per feature:
  - the change in choice and in rating;
  - whether the choice changes track each feature's correlation with the price (Spearman over features);
  - whether each feature's rating excess tracks how much more the rating recipe weighs it.
- **New sentences:** the prices predicted from the model's states (a ridge readout trained on the 96, layer 20), from F1 alone, and from the "how good" rating.
- **The random benchmark:** the price direction's choice change against the 20 random ones.

## 5. Running it

- One GPU pod (L40S, $1.09/h): about 20 minutes, about $0.40.
- The pod pushes its results to `pod-results/pilot13` and stops itself (`tools/pod_ship.sh`).
- Predictions: Claude's sealed first, then Jasper's (`predictions/`).
