# Pilot 5: prediction questions

Answered before the run: Claude first (sealed, hash committed), then Jasper one by one, the
overall verdict last. Numbers: a best guess and an 80% range. Probabilities: 0-100%.
Scoring: probabilities by Brier score; numbers by whether the 80% range contains the result and
by the distance of the best guess. Where a result is not defined (for example the currency
ratios when the push has no effect), the question is dropped for both.

"Slope" = change in the log-odds of answering A per value unit pushed (1 value unit = one
step of u, = the number of people times e^(1/1.46) ≈ 2). If the model uses the code as value:
+1 for A's direction, -1 for B's.

## The pre-registered test (steering at the last position)

1. **A's slope at layer 28.** Best guess and 80% range. (Pilot 4, pushing at A's own line at
   layer 23: 0.14.)
2. **B's slope at layer 28.** Best guess and 80% range.
3. **Controls.** Probability that at layer 28 the random direction and both shuffled-values
   directions each have a slope smaller in size than 0.3 × A's slope ("specific").
4. **Layers.** (a) Which of layers 24, 28, 32, 36 gives the largest A slope? (b) Probability that
   "full size" (A's slope in [0.5, 2] and B's in [-2, -0.5]) holds at at least one of the four.
5. **cos(w_A, w_B) at layer 28.** Best guess and 80% range. (-1: one value-difference code; 0:
   unrelated; positive: a shared component, for example how much is at stake.)
6. **One currency** (only if the push has an effect at layer 28). Probability that A's slope is
   within a factor 1.5 (a) for gain-worded against harm-worded goods; (b) for money goods against
   the others.

## Exploratory: the circuit

7. **Who writes A's value** (read at layer 28, over prompts with a real good in A): the share
   written by MLPs (against attention heads). Best guess and 80% range, and the probability that
   it is above 50% ("carry, then compute").
8. **How concentrated:** how many of the 1,149 pieces (embedding, 28 × 40 heads, 28 MLPs) carry
   half of it. Best guess and 80% range.
9. **Patching the whole last position:** the first layer (4, 8, ..., 40) at which it carries at
   least half of a swapped good's effect.
10. **Patching the top 20 writers** of the good's part: the transfer (0 = nothing, 1 = all). Best
    guess and 80% range.
11. **Letter heads:** probability that at least one of the top 10 answer-writing heads is a letter
    head (its attention difference between the "A" and "B" labels correlates at least 0.5 with the
    log-odds).
12. **Number and good in one piece:** probability that at least one of the top 10 writers of A's
    value carries the number of people and the good's worth in about the right mix
    (number-to-good ratio between 0.73 and 2.92, i.e. within a factor 2 of b = 1.46).
13. **The objective at the last position** (goal module 1, objective on a real good): the share of
    the choice's move towards the target that shows up in the value-difference readout at layer
    28. Best guess and 80% range. (Pilot 4, with the weaker codes: about 7%.)

## The extension: other answer formats (DESIGN.md, section 8; asked before question 14)

F1. **Rating, behaviour:** the correlation between the model's expected rating (1-9) of a single
    outcome and its A/B value, over the 9 real options × 7 numbers of people × 2 wordings. Best
    guess and 80% range.
F2. **Rating, readable:** the option-level correlation of A's arrow (layer 28, fitted on A/B
    prompts) with the 9 goods' values in the rating format (A/B format, pilot 4: 0.92). Best guess
    and 80% range.
F3. **Numbered, causal:** A's push slope at layer 28 in the numbered format as a fraction of the
    A/B slope at layer 28 (1 = works just as well). Best guess and 80% range. (Dropped if the A/B
    slope is not above 0.)
F4. **Yes/no, causal:** probability that pushing A's arrow raises "Yes" (95% interval above 0).
F5. **Rating, causal:** probability that pushing A's arrow raises the rating (95% interval above 0).
F6. **Writers:** of the top 20 writers of A's value in the A/B format, how many are also in the
    numbered format's top 20? Best guess and 80% range.

## Last: the verdict at layer 28

14. Probabilities, adding up to 100%: **used as value, at full size** / **used, but not at full
    size** / **an effect, but not specific to value** / **not shown**.
