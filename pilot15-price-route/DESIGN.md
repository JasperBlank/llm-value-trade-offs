# Pilot 15: where the price is finished, and how it is compared

## 1. The question

Jasper: *why does the model pick the options it picks?* Pilot 14 found:
- Single words move the choice a lot, and by their meaning. A rescue word turns the harm it names into a good; "each" multiplies the outcome; "brief" and "mild" shrink it.
- The price signal on the sentence's own tokens at layer 20 is coarse. It barely follows those word-level changes (r 0.28).

So the price the choice acts on is finished somewhere else. Earlier pilots point to where:
- **Pilot 7:** with goods instead of sentences, the options' information met at the line breaks around layer 24 and reached the answer position at layer 28.
- **Pilots 5-6:** at the answer position, 20 pieces carry most of a swapped good (0.65). They are 6 MLPs and 14 heads, mostly in blocks 24-27 (pilot 5 counts blocks from 0). The same pieces carry "which pile has more apples" (a generic comparator).

This pilot asks three things:
1. **Is A's price finished before the model reads B?**
2. **Where does A's sentence pass on its way to the answer?**
3. **Do pilots 5-6's comparator pieces compare these prices too?**

It also redoes pilot 14's Q4 with a working control.

## 2. Material and measures

- **Model:** Qwen3-14B (bf16, thinking off).
- **Menus:** pilot 10's prompt at 8:8, with pilot 10's 96 sentences as option A. Option B is fixed: "each help a teenager get through a difficult school year", the new sentence (pilot 13) whose price (+0.35) is nearest the 96's median, so that choices are not saturated.
- **Positions in the prompt:**
  - A's tokens (mean) and A's last token;
  - the line break after A;
  - B's last token and the line break after B;
  - the answer position (the prompt's last token).
- **Layers:** 0-40 by 4.

### (a) The readout map

- For each position and layer, a ridge readout is trained on the 96 intact sentences to predict their choice logit (standardised states; penalty chosen leaving one value out).
- It is then applied to pilot 14's 994 single-word deletions, each as option A against the same B.
- The measure is **r between the readout's predicted change and the choice's actual change**.
- The readout never sees a deletion, so a high r means that position holds a price that follows the words.

### (b) The route

- Each of the 96 sentences is paired with a donor sentence of another value, priced at least 2 apart (seed fixed).
- At one position and one layer, the menu's state is replaced with the state from the donor's menu. The measure is how much of the donor's change in the choice comes through: the transfer, a slope through zero, as in pilot 5.
- **Positions:** A's last token, the line break after A, B's last token, the line break after B, and the answer position.
- **Layers:** 8, 16, 20, 24, 28 and 32.

### (c) The comparator

At the answer position, pilot 5's top-20 pieces (outputs patched from the donor's run) are compared with 3 random sets of 20 other pieces from the same blocks: for each top piece, another piece of its block.

### (d) Pilot 14's Q4, redone

- **What is silenced:** pilot 14's 10 top MLPs (blocks 10 and 12-20), mean-ablated at A's tokens. As before, the measure is the fraction of the choice's price slope lost, against pilot 14's 8 partners.
- **The control:** 5 random sets of 10 MLPs from blocks 2-9, 11 and 21-30. Block 1, which erases the words, is excluded, and none of the sets overlap the top 10.
- **Already known:** the top 10 lost 32% in pilot 14. The new information is the control.

## 3. Pre-registered questions

| | Question | Measure | Pass |
|---|---|---|---|
| **Q1** | Is A's price finished before B is read? | The best of A's positions (tokens' mean, last token, line break) and layers (0-40 by 4), chosen on the deletions of half the values. Its r on the other half's deletions, from the readout map. | r ≥ 0.6 |
| **Q2** | Does A's sentence pass through one place? | The largest transfer of a single-position patch: A's last token, the line break after A, B's last token or the line break after B, at layers 16-32 | ≥ 0.5 |
| **Q3** | Do pilots 5-6's comparator pieces compare these prices? | Transfer of pilot 5's top-20 pieces at the answer position | ≥ 0.5, and every random set ≤ 0.1 |
| **Q4** | (Pilot 14's Q4, redone) Does the choice use the price writers? | The fraction of the price slope lost with the top 10 MLPs silenced, against the 5 random sets | ≥ 0.3, and more than all 5 random sets |

## 4. Also reported (not pre-registered)

- **The full readout map:** every position and layer. Also an upper bound: a ridge trained on the deletions themselves (held out by value), which shows where the information is present at all.
- **The full route map,** including the whole state at the answer position at each layer. Pilot 5, with goods: 0.03 at 24, 0.77 at 28.
- **The flips:** for the deletions that move the choice more than 10 logits (rescue words), the first layer at which each position's readout gets the sign right.
- **The comparator's parts:** the transfer of its MLPs and of its heads, each on their own.

## 5. Running it

- One GPU pod (L40S at $1.09/h if one is free, otherwise the 96 GB RTX PRO 6000 at $2.49/h). About 10 minutes: roughly 1,100 passes for the readout map, 3,000 for the route and comparator, and 4,600 for Q4.
- The pod pushes its results to `pod-results/pilot15` and stops itself. Claude's watcher checks every minute.
- Predictions: Claude's sealed first, then Jasper's (`predictions/`).
