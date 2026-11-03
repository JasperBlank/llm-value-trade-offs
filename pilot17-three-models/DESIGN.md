# Pilot 17: do Llama and a bigger Qwen price outcomes the same way?

## 1. The question

Jasper: check Llama for a "which price is bigger" comparator, and see whether it does the same thing or assigns an actual price to things. Same for the 32B model.

For Qwen3-14B, pilots 5-6 and 15-16 found:
- **A generic comparator.** At the answer position, the decision arrives between layers 24 and 28 of 40. Twenty pieces (heads and MLPs) carry 0.60 of a swap of one outcome. The same pieces carry "which pile has more apples" (0.54) and "which is heavier" (0.39).
- **A partly relative price.** About three quarters of an outcome's price is the same whatever the alternative is, against a bar of 0.9 for "fixed". 32% of clearly separated pairs swap order between alternatives, and there are choice cycles.

This pilot runs the same tests on **Llama 3.1 8B Instruct** and **Qwen3-32B**, and again on Qwen3-14B as the reference. On 14B the comparator is found with a new method, so its first job is to check that the method recovers what pilots 5-6 found.

## 2. Material and measures

Each model is run separately; thinking is off for the Qwen models.

### (a) Fixed or relative price (pilot 16)

- **Menus:** pilot 10's 96 outcomes and pilot 14's 994 single-word deletions, each as option A against 9 alternatives B. The B's are pilot 14's 8 partners plus pilot 15's mid-priced B. That is 9,810 menus at 8:8.
- **Common share:** of the choice's word effects (994 × 9) and of the whole outcomes' logits (96 × 9), the part that is the same for every B.
- **Order flips:** the share of clearly separated outcome pairs (a gap above 2 logits under at least two B's) that are ordered differently by different B's.
- **Cycles:** four-option cycles in the choices that no fixed values allow.
- **Inside:** the price readout's common share at the line break after B (pilot 16's Q1). The line break after A is the built-in check and must give 1.
- **Depths:** states are read at 40%, 50%, 60%, 70% and 80% of the model's depth. Qwen3-14B's layer 24 is 60%.

### (b) The comparator (pilots 5, 6 and 15)

- **Swaps:** each of the 96 outcomes, against pilot 15's mid-priced B, gets a donor outcome of another value. The donor is at least 2 logits apart by this model's own average logits.
- **Where the decision arrives:** the whole state at the answer position is swapped at depths of 40-90%.
- **Finding the comparator pieces:**
  - Each attention head's and MLP's direct contribution to logit(A) − logit(B) at the answer position is computed for both menus of a swap.
  - A piece's score is how much of the swap's change in the logit its own change carries (a slope through zero).
  - The 20 top pieces are chosen on the swaps of half the values. Then they are swapped together, as in pilot 5, on the other half.
- **Is it generic?** The same 20 pieces are swapped on pilot 6's "which pile has more apples" pairs and "which is heavier" pairs, 120 of each. The control is 3 random sets of 20 other pieces, each piece from the same block as a top piece.

## 3. Pre-registered questions (per model; bars fixed now)

| | Question | Measure | Pass |
|---|---|---|---|
| **Q1** | **Does it give each outcome a fixed price?** | Common share of the choice's word effects and of the whole outcomes' logits, 9 alternatives | both ≥ 0.9 |
| **Q2** | **Does a compact comparator decide?** | The 20 top pieces, chosen on half the values: their transfer on the other half | ≥ 0.5, every random set ≤ 0.1 |
| **Q3** | **Is it the generic "which is bigger" machine?** | The same 20 pieces: transfer on apples pairs and on weight pairs | both ≥ 0.3, every random set ≤ 0.1 |

**Method check (not a question):** on Qwen3-14B, Q2 and Q3 should come out like pilots 5-6, with the top 20 carrying about 0.6 on sentences and 0.4-0.5 on apples and weight. If they don't, the piece-finding method is suspect for the other models too.

## 4. Also reported (not pre-registered)

- Order flips and cycles, and the common shares without the two extreme alternatives.
- **The price before B is read** (pilot 15's Q1), with the mid-priced B: a readout at the line break after A, trained on the intact outcomes and tested on the word deletions.
- **Where the decision arrives,** for sentences and for apples.
- Pilot 5's own 20 pieces on Qwen3-14B, as a second reference.

## 5. Running it

- **One GPU pod:** the RTX PRO 6000 (96 GB, $2.49/h), since Qwen3-32B needs about 66 GB. A 100 GB volume is enough, since each model is deleted from disk after its run.
- **Order:** Llama first, then Qwen3-14B, then Qwen3-32B. If one model fails, the run moves on to the next.
- **Time and cost:** about 30 minutes, about $1.25.
- **Llama access:** Llama is gated. The pod checks access first and skips it if the Hugging Face token has none.
- The pod pushes its results to `pod-results/pilot17` and stops itself. The watcher checks every minute.
- **Predictions:** no prediction gate. The bars above are fixed before any result.

## 6. Addendum after the first run (3 Nov 2026, 14:35 Amsterdam time; before the rerun)

**What happened.**
- **Qwen3-32B did not run.** The disk filled up: the cleanup between models missed Hugging Face's chunk cache.
- **The method check failed on Qwen3-14B.** The pieces chosen by their direct contribution to logit(A) − logit(B) sit in the last blocks (34-39 of 40, counted from 0). They are the pieces that write the letter. They carry 0.96 of held-out sentence swaps, and 0.94 and 0.96 of apples and weight. But any A-or-B question passes through them, so "generic" is close to automatic for them. They share 1 piece with pilot 5's 20 comparator pieces (blocks 20-27), which carry 0.58 here.

The first run's results stand as reported, in `results/real/`.

**The rerun** (`results/real2/`, all three models, Qwen3-32B first, with both caches cleared after each model) adds the comparator as pilots 5-6 found it:
1. **The decision layer** L* is the first layer at which swapping the whole state at the answer position carries at least half of a sentence swap.
2. **The decision direction** at L*: the difference of means of the answer position's state between the highest- and lowest-logit thirds of the 96 menus.
3. **Each piece in the blocks before L*** gets its contribution along that direction at the answer position. The pieces are scored as before (the slope of their change on the swap's change in the logit) on half the values. The top 20 are the **decision writers**.

**Pre-registered for the rerun (bars fixed now):**

| | Question | Measure | Pass |
|---|---|---|---|
| **Method check** (Qwen3-14B) | Does the new method find pilot 5's comparator? | Pilot 5's 20 pieces among the 20 decision writers | ≥ 5 |
| **Q2b** | Do the decision writers carry the choice? | Their transfer on held-out sentence swaps | ≥ 0.4, every random set ≤ 0.1 |
| **Q3b** | Are they the generic "which is bigger" machine? | Their transfer on apples and on weight swaps | both ≥ 0.3, every random set ≤ 0.1 |

Q1, Q2 and Q3 are measured again as before. Q2b and Q3b are the comparator questions that matter. Q2 and Q3 are about the pieces that write the letter.
