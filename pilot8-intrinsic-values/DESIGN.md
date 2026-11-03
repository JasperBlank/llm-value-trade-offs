# Pilot 8: trade-offs between intrinsic values (Qwen3-14B)

Written 3 Nov 2026, before any result. Section 4 is pre-registered; the thresholds are also in
`config.yaml` under `prereg`. This replaces two earlier pilot-8 designs that never ran
(`../pilot8-value-ingredients/`).

## 1. Questions (Jasper)

1. How does the model trade intrinsic values off against each other?
2. Does the same thing light up when it trades different values off against each other?
3. Is the menu consistent?
4. When asked how much it cares about a single value, does the same value representation show up as
   the one that appears before the comparison in the trade-offs?

## 2. Values and prompts (`make_prompts.py`)

24 intrinsic values: Clearer Thinking's 22 categories (Achievement, Beauty, Caring, Defending,
Diversity, Doing, Embodying, Experiencing, Fairness, Freedom, Happiness, Justice, Knowing, Legacy,
Longevity, Loyalty, Nurturing, Pleasure, Preserving, Protection, Respect, Spirituality) plus
Non-suffering and Truth. Each is an outcome for strangers, "{n} people each …", in two phrasings
(`values.json`). A random 12 are the training half and 12 are held out (for Q2 and Q4b).

| set | prompts | what |
|---|---|---|
| trade-offs | 13,248 | every pair of values, 4 phrasing combinations, both orders, 3 numbers of people (2/8, 4/4, 8/2), 2 wordings |
| menus | 600 + 600 | 300 triples: A, B and a third option C, each in both A/B orders, and the matching two-option prompt |
| ratings | 96 | each sentence on its own: "How much does it matter to you that the following happens?", 0-9 |
| rating pairs | 1,104 | rating prompts differing only in the value (for Q4c) |
| swap pairs | 1,440 / 480 | trade-offs differing only in option A's value: all three values from the training half / the held-out half |

## 3. What runs (`pilot8.py`)

- **behave:** P(A) for every trade-off; worth u per value by maximum likelihood,
  P(A) = σ(u_A − u_B + b·log(n_A / n_B) + c), with bootstrap intervals (the price list, Q1); the
  same fit per sentence (phrasing reliability); the three menu checks (Q3); the expected stated
  rating per value (Q4a).
- **source:** the worth direction on option A's own words: the site and layer by the rule below,
  then one direction (distributed alignment search, k = 1) trained on swaps among the training values.
  Each value's position along it (in option A, before B is read) is compared with its worth (Q2a).
- **compare:** the answer position's arrival layer (the first layer of 16, 20, …, 40 where swapping
  it carries at least half of a value swap); the comparison site is the text after B's line, 4 layers
  earlier. One direction there, trained on trade-offs among the training values, is tested on
  trade-offs among the held-out values (Q2b).
- **rating:** the source worth direction read in the rating prompts, at the matching token of the
  outcome line (its last token, or the line break after it) and the same layer (Q4b); and, between
  rating prompts of two values, swapping only that direction against swapping the whole token (Q4c).

## 4. Pre-registered criteria

**Source rule:** among A's last token and the line break after A, and layers 4-24, the site and layer
where swapping the whole site carries the most of a value swap (ties: the earlier layer).

- **Q1** (descriptive): the price list with 95% intervals.
- **Q2a, the same value code across values:** the source direction, trained on 12 values, orders the
  12 held-out values with r ≥ **0.7** against their worth.
- **Q2b, the same comparison across trade-offs:** the comparison direction carries at least **0.5**
  of the site's effect on trade-offs among the held-out values.
- **Q3, a consistent menu** (all three): (i) one scale: worth fitted on one wording predicts the other
  wording's answers with r ≥ **0.8** (log-odds); (ii) transitivity: at most **5%** of the 2,024 triads
  are cyclic (majority preference at equal numbers, over phrasings, orders and wordings); (iii) menu
  invariance: adding a third option changes the A:B odds by a median factor of at most **1.5**.
- **Q4a:** the stated ratings correlate at least **0.7** with the trade-off worth (24 values).
- **Q4b:** in the rating prompts (no comparison), the source direction orders the 12 held-out values
  with r ≥ **0.7** against their trade-off worth.
- **Q4c:** swapping only that direction between rating prompts carries at least **0.3** of the whole
  token's effect on the stated rating (95% interval above 0); not testable if the whole token carries
  less than **0.1** of the difference between the two ratings.

## 5. Limitations known in advance

One model (Qwen3-14B), one answer format, English, our own sentences for each value (two phrasings
help but do not settle it). The values differ in how much is at stake (a meal against a year of life);
the prices reflect these sentences, not the values in the abstract. The rating question is one framing
of "how much do you care"; the held-out halves are 12 values each.

## 6. Budget

One A100 80 GB pod ($1.79/hour), about 30 minutes in all. The pod prints its results and stops itself.
