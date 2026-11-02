# Pilot 6: prediction questions

Answered before any result is seen: Claude first (sealed, only the hash committed), then Jasper
one question at a time, the overall probabilities last. Numbers: a best guess and an 80% range.
Probabilities: 0-100%. Scoring as in pilot 5: probabilities by Brier score; numbers by whether the
80% range contains the result and by the distance of the best guess.

"Transfer" = how much of an effect an intervention carries: 0 = nothing, 1 = all of it.
All numbers are on the held-out prompts (the three wordings the subspace never saw).

## The rival hypothesis: a generic comparator? (P1)

For reference, pilot 5 on the value task: the whole last-position swap first carries at least half
at layer **28** (24: 0.03, 28: 0.77, 32: 0.95), and the top-20 value writers carry **0.65**.

1. **Quantity, where:** the first layer (4, 8, ..., 40) at which swapping the whole last-position
   state carries at least half of the effect, for "which pile has more apples".
2. **Weight, where:** the same for "which of these two is heavier".
3. **Quantity, which pieces:** the transfer of pilot 5's top-20 value writers on the quantity
   question. Best guess and 80% range.
4. **Weight, which pieces:** the same on the weight question. Best guess and 80% range.

## The direction the model uses (P2)

5. **One direction:** at layer 28, the share of the whole-state swap that the best single learned
   direction carries. Best guess and 80% range. (Pass for P2: at least 0.5.)
6. **How many:** the smallest of k = 1, 2, 4, 8, 16 at layer 28 that carries at least 0.9 of the
   whole-state swap ("none" is allowed).
7. **The probe as a direction:** at layer 28, the share of the whole-state swap carried by swapping
   only the probe's direction. Best guess and 80% range.
8. **Learned against probe:** the cosine (in size, 0 to 1) between the learned direction and the
   probe at layer 28. Best guess and 80% range. (Two random directions: about 0.01.)

## Is it real, and is it a lever? (P4, P3)

9. **Tracks value:** the correlation, over held-out prompts, between the learned direction's
   coordinate and A's value at layer 28. Best guess and 80% range. (Pass for P4 needs at least 0.5,
   and the writers' share.)
10. **A better lever:** the push along the learned direction divided by the push along the probe,
    both one natural value unit, at layer 28. Best guess and 80% range. (Pass for P3: at least 3.)

## Overall (asked last)

11. (a) **P1 verdict:** probabilities for "generic comparator", "value-specific pieces" and "mixed"
    (summing to 100%). (b) Probability that **P2** passes. (c) Probability that **P3** passes.
    (d) Probability that **P4** passes.
