# Pilot 16 results (Qwen3-14B, run 3 Nov 2026)

**The run.**
- **GPU:** no L40S was free, so it ran on the RTX PRO 6000 (96 GB, $2.49/h), from 12:30 to 12:33 UTC. That is about $0.15 including model loading; the real run took 1.9 minutes.
- **Quick test:** passed on Qwen3-0.6B.
- **Results:** the pod pushed everything to `pod-results/pilot16` and stopped itself, and the watcher picked it up within a minute. These files are copied from that branch unchanged.
- **Predictions:** the bars were fixed in [`DESIGN.md`](../../DESIGN.md) before the run. Jasper's prediction was optional, and none was given. Claude's sealed predictions are in `predictions/` (the hash matches).

| File | What |
|---|---|
| `report.md` | The pod's report: Q1-Q3, the common share by position and layer, without the extreme alternatives, per alternative |
| `verdicts16.json` | Everything as data |
| `effects16.npz` | The 96 × 8 logits, the 994 × 8 logits of the deletions, the word effects in the choice and in the readout |
| `log_pod.txt`, `log_real.txt` | The pod's logs |

Figure: [`fig15_relative.png`](../../../figures/fig15_relative.png).

## The questions (bar 0.9 for "absolute")

- **Q2. Choice: does a word count the same whatever the alternative is? No.** Of the variation in the choice's word effects, 0.75 is common to all 8 alternatives.
- **Q3. Choice: are whole outcomes priced the same whatever the alternative is? No.** Of the variation in the 96 × 8 logits that belongs to the outcomes, 0.76 is common.
- **Q1. Inside: is A's finished price the same whatever the alternative is? No.**
  - At the line break after B (layer 24), the price readout's word effects are 0.75 common, the same as the choice.
  - The built-in check passes: the line break after A, which comes before the alternative, gives 1.00 (0.999 to 1.000 across layers).
- **Verdict (fixed in advance): relative, and already in the finished price.** About three quarters of an outcome's price is its own; about a quarter depends on what it is compared with.

## Is it relative values, or just the logit saturating?

Many menus are near-certain. Against each alternative, 41-83% of the 96 outcomes have a logit beyond ±10, and near-certain choices could squeeze the logit without any change in value. Four checks, none of them pre-registered:

- **Without the two extreme alternatives** (−7.3 and +4.3), the common shares rise only a little: 0.79 for words and 0.81 for whole outcomes.
- **Fixed values with any squeezing keep the order of the outcomes. Qwen's order changes.**
  - A squeezing that is the same for every alternative cannot reorder outcomes: if A1 is valued above A2, A1's logit is higher against every B.
  - Of the pairs of outcomes that are clearly apart (by more than 2 logits) under at least two alternatives, 32% are ordered differently by different alternatives.
  - Counting only menus that are not near-certain, it is 27-36% (with logits within ±6, ±8 or ±10).
- **Even the choices contradict any fixed values.** "Each get to choose which film to watch tonight" beats "each turn the activity they love into their full-time job" (+6.8), while "each have their life extended by a year" loses to it (−2.8). But against "each have a decade of unequal pay made up to them in full", the film choice loses (−2.2) and the extra year wins (+10.0). That is a four-way cycle (film > job > extra year > back pay > film), which no set of fixed values can produce. There are 36 such cycles with every gap above 2 logits, and 262 with any gap.
- **Ranks are not stable either.** The outcomes' order correlates 0.80 between alternatives on average (Spearman; the lowest pair 0.63).

## What else the run shows

- **The dependence is already in the finished price.** At the line break after B, the common share is 0.72-0.79 across layers 16-32, peaking at layer 20. At the answer position it is 0.80 at layer 24. A separate readout for each alternative gives 0.72 at layer 24.
- **A readout trained with one alternative transfers only partly.** It predicts the choice's word effects with that alternative at r 0.83, and with the others at r 0.77.
- **Each alternative stretches or squeezes the differences between outcomes in its own way.**
  - The slopes of each alternative's logits on the average over all 8 run from 0.49 to 1.55.
  - The extreme alternatives squeeze, as near-certainty would.
  - But so does "each finally learn the truth about what happened to a parent they never knew" (0.49), which is not extreme. The steepest is "are each spared a month of back pain" (1.55).

## Cautions

- **One position:** A was always the first option.
- **Eight alternatives,** all from pilot 13's new sentences. One model.
- **The logit is the graded measure** throughout. In probability, many menus are certain whatever the words are.
- **The 0.9 bar was a judgement call.** But the reversals and cycles do not depend on it.
- **This fits with pilots 8 and 10.**
  - Pilot 8 found transitive choices (3% cyclic triads). Pilot 10 found prices stable across menus (r 0.98), but those prices were averaged over many alternatives: the "three quarters" here.
  - What this pilot adds is how much the remaining quarter matters within a single menu.

## Reading

Qwen does not carry a fixed value for each outcome into the comparison. About three quarters of an outcome's price is its own, the same whatever it is compared with. That is the stable price that pilots 8-10 measured by averaging over many menus.

The remaining quarter depends on the alternative, and it is already there when the price is finished, at the line break after the second option. It is big enough to reorder outcomes, and even to produce choice cycles that no fixed set of values allows.

So "Qwen's values" are partly menu-relative. Each outcome has a stable core value, and the comparison bends it.
