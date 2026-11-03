# Pilot 16: is the finished price absolute, or relative to the alternative?

## 1. The question

Pilot 15 found where Qwen finishes an option's price: at the line break after the second option, at layers 20-24. There it follows the choice word by word (r 0.85). But pilot 15 used one fixed alternative B, so it could not tell apart two different stories.

- **Absolute, finished late.** Qwen completes a value of A that is the same whatever B is. The comparison then works with that fixed value. This is a utility per outcome, just finished at the end of the options.
- **Relative.** What Qwen completes at that point is A's value as seen next to this particular B. A word in A counts for more or less depending on what A is compared with, as in range normalisation, contrast or similarity effects. Then its values exist only within a menu.

**The test:** set the same outcomes, and the same single-word deletions of them, against several different alternatives B.
- If the value is absolute, deleting a word changes A's finished price by the same amount whatever B is.
- If it is relative, the change depends on B.

## 2. Material and measures

- **Model:** Qwen3-14B (bf16, thinking off).
- **Options A:** pilot 10's 96 sentences and pilot 14's 994 single-word deletions of them.
- **Alternatives B:** pilot 14's 8 partners, pilot 13's new sentences spread over the price range: −7.3, −3.1, −2.6, −2.2, −1.1, +0.2, +1.0, +4.3.
- **Menus:** every A against every B, as in pilot 10 at 8:8, with A always first. That is (96 + 994) × 8 = 8,720 menus.
- **Read from each menu:**
  - the choice logit;
  - the states at the line break after A, the line break after B and the answer position, at layers 16-32 by 4.

**A word's effect.** For a deletion d of sentence A and an alternative B_k:
- in the choice: Δl(d, k) = logit(A without the word, B_k) − logit(A, B_k);
- inside: Δp(d, k), the same difference in a price readout at the line break after B. The readout is a ridge readout trained on the 768 intact menus (96 A × 8 B) to predict the choice logit, the same way as pilot 15.

**Common share.** The 994 × 8 table of effects splits into a part that is the same for every B (each deletion's mean over the 8 Bs) and an interaction. The common share is the first part's sum of squares divided by the sum of both parts (after removing each B's mean).
- If every word counts the same whatever B is, the common share is 1.
- If B rescales or reshapes the effects, it is lower.

**Whole outcomes.** The 96 × 8 table of choice logits splits into an A part, a B part and an A×B interaction. The common share is the A part's sum of squares divided by the A part plus the interaction. If the values are fixed, the logits are close to A's value minus B's value, and the common share is close to 1.

**Built-in check.** The line break after A cannot depend on B (it comes before B), so its common share must be exactly 1.

## 3. Pre-registered questions

| | Question | Measure | Pass ("absolute") |
|---|---|---|---|
| **Q1** | **Inside: is A's finished price the same whatever B is?** | Common share of the readout's word effects Δp at the line break after B, layer 24 | ≥ 0.9 |
| **Q2** | **Choice: does a word count the same whatever B is?** | Common share of the choice's word effects Δl | ≥ 0.9 |
| **Q3** | **Choice: are whole outcomes priced the same whatever B is?** | Common share of the 96 × 8 choice logits (A part against A×B interaction) | ≥ 0.9 |

**Verdict.**
- **The choice decides whether Qwen's values are relative.** If Q2 and Q3 pass, Qwen chooses as if it compares fixed values, whatever is going on inside: absolute, finished late. If either fails, its values depend on the alternative.
- **Q1 then says where.**
  - If Q1 fails, the dependence on B is already in the finished price at the line break after B.
  - If Q1 passes while the choice fails, the dependence enters later, in the comparison.

**A limit of Q1, found while testing the code on a fake model.** A readout can pick up information about A at the line break after B that does not depend on B, even when the code the comparison actually uses does. In the fake model, a planted B-dependent rescaling of A's value still gave a common share of 0.95. So a passing Q1 cannot prove that nothing inside depends on B. It can only show that the price readout does not, which is why the choice decides the verdict.

## 4. Also reported (not pre-registered)

- The common share at every position and layer (16-32), and with the readout trained separately for each B.
- **Without the two extreme Bs** (−7.3 and +4.3), where the choice may be close to certain and the logit may saturate.
- **The shape of any dependence:** for each B, the slope of its word effects against their mean over the Bs, plotted against B's price. Range normalisation would make the slopes depend on B's price.
- **Transfer:** a readout trained with one B, tested with each other B.
- **A separate readout for each B.** It is noisier (96 menus each), but it cannot borrow information that is the same across Bs. In the fake model it caught the planted rescaling (0.61).

## 5. Running it

- One GPU pod (L40S, $1.09/h; the 96 GB RTX PRO 6000 at $2.49/h if none is free). About 8,700 passes and two to three minutes of readouts: about 10 minutes, about $0.20.
- The pod pushes its results to `pod-results/pilot16` and stops itself. The watcher checks every minute.
- The bars above are fixed before any result. One optional prediction for Jasper (question 1), since this is a question where intuition is on the line.
