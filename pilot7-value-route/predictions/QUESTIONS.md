# Pilot 7: prediction questions

Answered before any result is seen: Claude first (sealed, only the hash committed), then Jasper one
question at a time, the overall probabilities last, then a consistency check. Numbers: a best guess
and an 80% range. Probabilities: 0-100%. Scored as in pilots 5 and 6. Questions 1-8 are about
Qwen3-14B; questions on Llama are dropped if the pod has no Llama access.

"Share" = how much of an effect a swap carries: 0 = nothing, 1 = all of it.

## The route

1. **When A's information leaves A's line:** the first layer (4, 8, ..., 40) at which swapping
   *everything after A's line* carries at least half of an A-good swap. (For reference: the answer
   position alone first carries half at layer 28.)
2. **A's own line end:** the most of an A-good swap that swapping A's last token *or* the line break
   right after it carries, at its best layer. Best guess and 80% range. (Floor for the A site: 0.1.)
3. **The B site:** which site the rule will pick after B's line: the line break after B, the reply
   line, the template tokens, or everything after B's line.

## The monkey test

4. **One currency, before B:** at the A site, the share of an A-*number* swap carried by the direction
   trained only on A-*good* swaps. Best guess and 80% range. (V2 needs 0.5, and 0.5 for good swaps.)
5. **One currency, after B:** the same at the B site. Best guess and 80% range.
6. **The exchange rate, after B:** at the B site, the coordinate's change per value unit for number
   swaps divided by that for good swaps (1 = exactly the model's own exchange rate). Best guess and
   80% range. (V3: between 0.67 and 1.5.)
7. **A's value or the comparison:** at the B site, ρ = the coordinate's change per unit of *B's*
   value divided by that per unit of *A's* value. −1 = the comparison (A minus B), 0 = A's own value,
   +1 = one value code for both options. Best guess and 80% range.
8. **Value or magnitude:** at the B site, the share of an apples swap carried by the value direction.
   Best guess and 80% range. (Below 0.3: value-specific.)

## Llama 3.1 8B

9. **Usable:** probability that Llama's own fitted values predict its held-out choices with r ≥ 0.5.
10. **Its arrival layer, as a fraction of its depth** (Qwen: 28 of 40 = 0.70). Best guess and 80% range.

## Overall (asked last)

11. (a) Probability that Qwen's **A site** is "one currency" (V2 and V3 pass, above the floor).
    (b) Probability that Qwen's **B site** is "one currency".
    (c) Qwen's B-site **V5**: probabilities for "the comparison", "a shared value code", "A's own
    value", "not testable" (summing to 100%).
    (d) Probability that Llama's B-site headline is the same as Qwen's (given Llama is usable).
