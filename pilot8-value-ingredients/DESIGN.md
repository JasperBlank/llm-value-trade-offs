# Pilot 8: why are the relative values what they are? The ingredients of worth

Written 3 Nov 2026, before any pilot-8 result. Section 4 is pre-registered; the thresholds are also
in `config.yaml` under `prereg`.

## 1. Question

Pilots 3-7 measured how much Qwen3-14B values 9 goods and traced how a good reaches the decision. But
the 9 goods differ in many ways at once (domain, who benefits, how much, need), so their values
cannot say *which* of these the model pays for. Pilot 8 builds outcomes from controlled ingredients
and asks: (1) how much does each ingredient add to an outcome's worth, and how much of the
differences do the ingredients explain? (2) Is the worth already assembled on the outcome's own words,
with each ingredient weighted as the choices weigh it?

## 2. Outcomes and prompts (`make_prompts.py`)

48 outcomes, all five ingredients crossed (3 × 2 × 2 × 2 × 2):

| ingredient | levels |
|---|---|
| domain | money / free time / health |
| frame | a gain ("each gain X") / a prevented loss ("are each prevented from losing X") |
| size | small / large: €20 / €200; an hour / a week of free time; a day / a month of good health |
| who | adults / children |
| need | none / "living in poverty" |

Example: "8 children living in poverty are each prevented from losing a week of free time".

**Behaviour.** Every unordered pair of outcomes (1,128), both orders, three numbers of people
(A 2 / B 8, 4 / 4, 8 / 2) and two question wordings (pilot 3's wordings 0 and 1, strangers
frame): 13,536 prompts. The answer is P(A) from the letter probabilities.

**Fit.** u (one per outcome, the first fixed at 0), b (the weight of log people) and a position bias
are fitted by maximum likelihood: P(A) = σ(u_A − u_B + b·log(n_A / n_B) + c). An ingredient's effect
is reported as an exchange rate in people: exp(β / b) ("one child counts as x adults").

## 3. Inside the model (Qwen3-14B)

- **Ingredient maps** (descriptive): minimal pairs of outcomes that differ in one ingredient (in
  option A, against 4 fixed option-B outcomes); the recipient's states at one site and layer
  replaced by the donor's (sites: A's last token, the line break after A, the line break after B,
  the text after B, the answer position; layers 4, 8, ..., 40); share of the ingredient's effect
  carried. This shows where each ingredient's contribution sits on its way to the decision.
- **The worth direction:** at the A site chosen by the rule below, one direction (distributed alignment
  search, k = 1) trained on swaps of option A's outcome among 32 training outcomes (the other 16 are
  held out). Then each outcome's position along it is read off its own state at that site (in option
  A, before option B has been read).

## 4. Pre-registered criteria

**Site rule:** among A's last token and the line break after A, and layers 4-24, the site and layer
where swapping the whole site carries the most of an outcome swap (ties: the earlier layer).

- **W1, the ingredients explain the worth:** a model with the five ingredients' main effects only
  (domain as two contrasts), fitted on 47 outcomes, predicts the 48th; over all 48, held-out R² ≥
  **0.7**: "worth is (mostly) a sum of ingredients"; 0.4-0.7: "partly"; below 0.4: "not explained by
  these ingredients".
- **W2, assembled on the outcome's own words:** along the worth direction, (a) the 16 held-out
  outcomes' positions correlate at least **0.7** with their fitted worth u, AND (b) the ingredients'
  effects on the position (a main-effects regression over all 48 outcomes) correlate at least
  **0.7** with their effects on worth (the six coefficients). Both: "assembled on the outcome's own
  words"; otherwise "not assembled there".

Reported, not pre-registered: each ingredient's effect with a 95% bootstrap interval (over prompts),
the largest ingredient, interactions (size × domain, frame × domain), the ingredient maps.

## 5. Limitations known in advance

One model (Qwen3-14B), one answer format (A/B), English, one template. "Small" and "large" are not
equal ratios across domains, so size × domain is expected. Twelve wordings of the same ingredients
would be stronger than two. The worth direction is one direction at one site.

## 6. Budget

One A100 80 GB pod ($1.79/hour), about 35 minutes. The pod prints its results and stops itself
(tested: the log stays readable after the stop); Claude then terminates it.
