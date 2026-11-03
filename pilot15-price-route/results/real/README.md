# Pilot 15 results (Qwen3-14B, run 3 Nov 2026)

**The run.** One L40S pod, from 12:04 to 12:10 UTC (about $0.11). The quick test on Qwen3-0.6B passed. The pod pushed everything to `pod-results/pilot15` and stopped itself, and the watcher picked it up within a minute. These files are copied from that branch unchanged. There were no probability predictions this time (Jasper's decision); the bars were fixed in [`DESIGN.md`](../../DESIGN.md) before the run.

| File | What |
|---|---|
| `report.md` | The pod's report: Q1-Q4, the readout map (four tables), the route, the comparator |
| `verdicts15.json` | Everything as data |
| `readout15.npz` | The choice's logits for the 96 sentences and the 994 deletions, the swap pairs and their effects |
| `log_pod.txt`, `log_real.txt` | The pod's logs |

Figure: [`fig14_price_route.png`](../../../figures/fig14_price_route.png).

**Set-up in one line.** Pilot 10's 96 sentences, and pilot 14's 994 single-word deletions of them, each as option A against one mid-priced B ("each help a teenager get through a difficult school year", 8:8). With this B the choice's logit correlates 0.88 with pilot 10's price, and the median deletion moves it 2.5 logits.

## The questions

**Q1. Does Qwen value each option on its own before comparing? No, by the bar, but a rough value is there.**
- Before B is read, the best readout on A's own positions follows the choice's word-level changes at r 0.53 (bar 0.6). That readout is the line break after A at layer 20, chosen on half the values and tested on the other half. On all deletions:
  - the line break after A gives 0.55;
  - A's words, averaged, give 0.31;
  - A's last word gives 0.26.
- That rough value already has the big flips right: for the 87 deletions that move the choice more than 10 logits (mostly rescue words), the line break after A gets the sign right 94% of the time.
- **The finished value appears at the line break after B,** at layers 20-24: r 0.85 at layer 24.
- A readout trained on the deletions themselves, an upper bound, gives the same picture: 0.72 at the line break after A against 0.90 at the line break after B.

**Q2. Does an option's worth funnel through one spot on its way to the answer? No, by the bar, but nearly.**
- Swapping in the state of one position from a menu with a different A sentence carries at most 0.43 [0.38, 0.48] of the whole swap (bar 0.5). That position is the line break after B, at layer 24.
- **Earlier,** A's last word carries 0.26 at layers 8-16 and the line break after A 0.10.
- **The answer position** takes over between layers 24 and 28 (0.02, then 0.75, then 0.91 at 32). With goods, pilot 5 found 0.03 and 0.77 at the same layers.

**Q3. Is the comparison a generic "which is bigger" machine? Yes.**
- At the answer position, pilot 5's 20 pieces carry 0.60 [0.58, 0.62] of a swap of A's sentence. Three random sets of 20 pieces from the same blocks carry 0.07, 0.00 and 0.00.
- The same 20 pieces carried 0.65 for goods (pilot 5), and 0.54 for "which pile has more apples" and 0.39 for "which is heavier" (pilot 6).
- Their 14 heads alone carry 0.47; their 6 MLPs alone 0.31.

**Q4 (pilot 14's Q4, redone). Passes, by a whisker.**
- Silencing pilot 14's top 10 MLPs at the sentence removes 32.3% of the choice's price dependence.
- Four random sets of 10 other MLPs remove 5-6%. The fifth removes 31.6%. That set holds the MLPs of blocks 2, 3, 5, 6, 9, 11 and 21-24.
- So the price writers matter far more than most other sets of MLPs, but one set of mostly early MLPs matters about as much.

## The picture so far

How Qwen picks between two outcomes (pilots 5-7 and 10-15):

1. **Reading each option.** As it reads option A, the MLP layers of blocks 10-20 write a coarse price onto A's words (pilot 14). The line break after A already holds a rough value (r 0.55 with the word-level changes), and gets the direction of big flips right.
2. **Finishing the price.** At the line break after B, at layers 20-24, a price appears that follows the words closely (r 0.85). It carries 43% of a swap of A's sentence on its own.
3. **Comparing.** The same 20 pieces that compare goods, piles of apples and weights move the decision to the answer position between layers 24 and 28. They carry 60% of a swap.

## Cautions

- **One B.** With B fixed, a value of A that is finished late cannot be told apart from a value of A relative to B. Testing that means varying B.
- **One position at a time.** Single-position swaps underestimate a route that is spread over several positions. The 43% is what the line break after B carries alone; together with nearby positions the route may carry more.
- **The deletions** include ungrammatical sentences and outcomes turned bad.
- **The answer position** at layer 24 already reads the price well (r 0.82), yet swapping its state there carries nothing (0.02). The later layers rebuild the decision there from other positions, so a high readout is not the same as being used.
- One model.

## Reading

Qwen does not finish valuing an option on its own. While reading it, it forms a rough value that already knows that "saved from a fire" is good and "being killed in a fire" is bad. The value its choice follows word by word is completed only once both options have been read, at the line break after the second option.

The comparison itself is not specific to values. It is the same machinery, the same 20 pieces, that decides which pile has more apples. So what is particular to Qwen's values lies in how an outcome is priced, not in how two prices are compared.
