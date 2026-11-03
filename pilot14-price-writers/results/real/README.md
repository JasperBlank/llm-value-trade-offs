# Pilot 14 results (Qwen3-14B, run 3 Nov 2026)

**The run.**

- **GPU:** no L40S (or any other 48 GB card) was free, so the run used an RTX PRO 6000 Blackwell (96 GB, $2.49/h). It ran from 10:07 to 10:13 UTC, about $0.25.
- **Checks:** the quick test on Qwen3-0.6B passed, and on the real model the parts add up to the readout exactly (r 1.0000).
- **Predictions:** both sets were committed before the run, and the hash was checked on the pod.
- **Results:** the pod pushed everything to `pod-results/pilot14` and stopped itself. These files are copied from that branch unchanged.
- **The watcher:** Claude's watcher had been started with the wrong argument (it checked once an hour, not once a minute), so the results were picked up 20 minutes late. The GPU was not affected.

| File | What |
|---|---|
| `report.md` | The pod's report: Q1-Q4, the writers, the block map, the top words, the running price |
| `verdicts14.json` | Everything as data |
| `attribution14.npz` | Per sentence, each head's and MLP's contribution to the readout; the shares; the word-deletion changes |
| `words14.json` | Per sentence, each word's deletion effect on the choice and on the readout |
| `running14.json` | Per sentence, the readout token by token |
| `predictions_vs_results.md` | Both prediction sets scored |
| `log_pod.txt`, `log_real.txt` | The pod's logs |

Figure: [`fig13_price_writers.png`](../../../figures/fig13_price_writers.png).

**Units.** The choice's mean logit rises 2.41 logits per price unit across the 96 sentences (r 0.96 with the price). The word effects in `report.md` and `words14.json` ("contribution") are changes in that mean logit, **not price units as the report's heading says**: divide by 2.41. The readout values are in price units.

## Pre-registered questions

- **Q1 fails, narrowly.** For the median sentence, the two most important words carry 0.48 of the summed word-deletion effects (bar 0.5).
- **Q2 fails.** Over the 994 deletions, the change in the layer-20 readout correlates only 0.28 with the change in the choice (bar 0.6).
- **Q3 passes, narrowly.** The 10 largest writers of the readout carry 0.52 of its spread across sentences (bar 0.5). All 10 are MLP layers, in blocks 10 and 12-20, each carrying 4-7%.
- **Q4 fails as pre-registered, but cannot be read.**
  - Mean-ablating the 10 top MLPs at the sentence's tokens removes 32% of the choice's price dependence, which clears the 30% bar.
  - The control was broken by the design. The random sets had to be MLPs near the top 10, but only 10 other MLPs were left in blocks 1-20: blocks 1-9 and 11. All five random sets were therefore the same set.
  - That set included block 1's MLP. Silencing it alone removes 106%, because it erases the words themselves.
  - So the random sets lost 108%, and Q4 "failed" for a reason that says nothing about the model. Fixing it needs a rerun with a control that leaves out block 1.

## What else the run shows

**The price readout is written gradually by the MLP layers, a little each.**
- MLPs carry 65% of the readout's spread, attention heads 35%, and the embedding 0%.
- The MLP shares are largest in blocks 10-20, at 4-7% each.
- No single head carries more than about 1% (the largest: head 19 of block 18, 1.1%).

**No block is needed on its own.**
- Silencing all 40 heads of one block, or its MLP, at the sentence's tokens leaves the choice's price dependence intact for every block from 2 to 24 (within ±0.04).
- Only block 1 matters (heads 74%, MLP 106%): it is where the words are first read.
- Silencing the 10 top MLPs together removes 32%. The writing is spread out and redundant.

**Single words move the choice a lot, and in ways that make sense.**
- The median deletion moves the choice 1.8 logits (0.76 price units). 18% of deletions move it more than 5 logits.
- Deleting a word lowers the price in 75% of cases.
- **Rescue words carry the most.** Deleting them turns the outcome into the harm itself:
  - "saved" in "are each saved from being killed in a fire": −35 logits (−14.5 price units);
  - "prevented", "no" (in "no longer prevented from making their own choices"), "shielded", "protected", "cured", "spared".
- **"each" matters often.** "8 people each save a stranger" is eight rescues; without "each" it reads as one.
- **Small-making words lower the price.** Deleting them raises it:
  - "for a week" in "relieved of a painful condition for a week": +8 logits;
  - "brief" in "a brief sense of awe";
  - "mild" before "headache".

**But the layer-20 readout barely follows these changes.**
- Over all deletions, r is 0.28.
- For the ordinary deletions, those moving the choice less than 5 logits (82% of them), r is 0.10. Within sentences the median r is 0.23.
- The readout does rise along a sentence at the words that matter. For example, "each create a lasting work that helps people for generations after they are gone" climbs at "lasting", "helps" and "generations". But it does not carry the word-level detail the choice reacts to.

## Cautions

- **Q4 is uninterpretable** (above). Q1 and Q3 sit within 0.02 of their bars.
- **Deletions can make sentences ungrammatical, or turn a good outcome bad.** The price direction was found on good outcomes only, so part of Q2's failure may be the direction not reaching outcomes unlike those it was found on. But Q2 also fails for the ordinary deletions.
- **The readout is one direction**: pilot 11's difference of means at layer 20, averaged over the sentence's tokens. Pilot 13 showed that the choice reads more than this one direction.
- One model, 96 sentences, one readout layer.

## Reading

The price the choice uses follows the meaning of the words, not just which words are present. A rescue word turns the harm it names into a good ("saved from being killed in a fire"). "each" multiplies the outcome by the number of people. "brief", "mild" and "for a week" shrink it.

On the sentence's own tokens, the layer-20 price signal is built gradually by the MLP layers of blocks 10-20, with no single part needed. But that signal is coarse: it tracks how big an outcome is across sentences, not the word-level detail the choice reacts to. So the price the choice acts on must be finished later, or elsewhere. The likely place is the end of option A's line, at layers 20-28, where pilot 7 found the comparison.
