# Pilot 13 results (Qwen3-14B, run 3 Nov 2026)

**The run.**

- One L40S pod, from 09:24 to 09:39 UTC (about $0.27).
- The quick test on Qwen3-0.6B passed. The real run took 13.3 minutes.
- Both prediction sets were committed before the run, and the hash was checked on the pod.
- The pod pushed everything to `pod-results/pilot13` and stopped itself. These files are copied from that branch unchanged.
- **The blind ratings (exploratory):** two fresh Claude instances rated the same 120 sentences on the ten questions. The plan, the analysis script and the ratings were all committed before the results came in ([`EXTERNAL_RATINGS.md`](../../EXTERNAL_RATINGS.md)).

| File | What |
|---|---|
| `report.md` | The pod's report: P1-P4, both recipes, the pushes, the new sentences |
| `verdicts13.json` | Everything as data |
| `features13.json` | Each sentence: price, rating, Qwen's ten feature answers; for the new ones, the predicted prices |
| `external_ratings.md`, `.json` | The exploratory comparison with the blind Claude ratings (`external13.py`) |
| `states13.npz` | The model's states for all 120 sentences: layers 0-40 by 4, menu and rating context (float16) |
| `new_menu13.npz` | The 13,824 new trade-offs (logit P(A) and which sentences, numbers of people) |
| `predictions_vs_results.md` | Both prediction sets scored |
| `log_pod.txt`, `log_real.txt` | The pod's logs |

Figure: [`fig12_recipe_in_words.png`](../../../figures/fig12_recipe_in_words.png).

## Pre-registered questions

- **P1 fails, by a hair.** From Qwen's own answers to the nine descriptive questions (F2-F10), the price is predicted with a held-out r of 0.79, against a bar of 0.8. One question alone, "how big a difference would this make" (F1), gives the same 0.79. Adding F1 to the nine gives 0.80.
- **P2 fails.** With Qwen's own answers, "putting right something morally wrong" weighs no more in the stated rating than in the choice: −0.04 [−0.22, +0.15].
- **P3 passes.** Pilot 12's disagreement push beats all 20 random directions. Its excess is +0.0064 rating points per residual unit, against −0.0062 to +0.0050 for the random directions (rank p = 1/21). The margin over the best random direction is small: the 95% interval over sentences is [+0.0049, +0.0079].
- **P4 passes, by a hair.** The recipe fitted on the 96 predicts the 24 new sentences' prices with r = 0.71 (bar 0.7).

## What else the run shows

**Qwen's own descriptions are entangled (a halo).**
- Among Qwen's answers, the nine features correlate 0.46 with each other on average. Among the blind ratings they correlate 0.02.
- The clearest case is moral. Qwen's "how much does this put right a moral wrong" correlates 0.59 with the price; the blind rating correlates 0.00. When Qwen values an outcome, it also says the outcome rights a wrong.
- So the recipe weights from Qwen's own answers are not reliable feature by feature. For example, the large duration difference (−0.47) does not survive with the blind ratings (+0.09).

**Rated blind, the features predict Qwen's choices at least as well (exploratory).**
- Price from the nine blind-rated features: held-out r 0.84 (Qwen's own answers: 0.79).
- The new sentences: 0.68 (Qwen's own: 0.71).
- So the recipe is not just Qwen describing its own vibe.
- Qwen's own answers do predict its **stated rating** much better: 0.86, against 0.62 for the blind ratings. Its stated answers share a halo with each other. Its choices follow the outside descriptions more closely.

**The two recipes with the blind ratings (exploratory).**
- **The choice** weighs meaning and purpose most (+0.47). Then pleasure and fun (+0.37; its plain correlation with price is −0.21, so it counts only once the other features are taken into account), health and safety (+0.24), how serious the need (+0.23) and how long it lasts (+0.22). Concrete, certain outcomes are priced lower (−0.24): in this set they are the small everyday ones.
- **The stated rating** weighs moral (+0.15 [+0.01, +0.29]) and relationships and care (+0.18 [+0.03, +0.33]) more than the choice does. This is P2's hypothesis, seen with the blind ratings, and it fits pilot 12 (Defending, Respect and Justice rated above their price). It was not the pre-registered test, which failed.

**Every predictor stops at about 0.7 on new outcomes.**

| predictor | 96 old sentences (r) | 24 new sentences (r) |
|---|---|---|
| nine features, Qwen's own answers | 0.79 | 0.71 |
| nine features, blind ratings | 0.84 | 0.68 |
| one question: how big a difference (F1) | 0.79 | 0.55 |
| Qwen's states at layer 20 (ridge, held out) | 0.84 | 0.74 |
| Qwen's stated "how good" rating (plain correlation) | 0.83 | 0.74 |

- About half the variance of a new sentence's price is not predicted by anything here. This fits pilots 8 and 10: the price belongs to the exact sentence.
- **The largest misses are priced far below their description.**
  - "spend a week travelling somewhere they have never been": −7.3 (predicted −1.0). That is below "get to decide where to go for lunch" (−3.1).
  - "find out how a magic trick works": −8.1 (predicted −4.1).
- **And one far above:** "see the company that poisoned their town held fully responsible": +4.3 (predicted +1.5).

**The pushes: random directions are not zero.**
- The price push replicates pilot 12 exactly: +0.771 logits and +0.112 rating points per push.
- The 20 random directions move the choice by −0.33 to +0.33 logits per push, so the single random direction of pilots 11-12 understated how much an arbitrary push of this size does. The price direction (0.77) is still far above all of them.
- The ten features' directions all move the choice by +0.45 to +0.63, even those nearly at right angles to the price direction. Concreteness has a cosine of 0.18 with the price direction and still moves the choice +0.45.
- So the choice reads more than the one price direction. This fits pilot 11's transfer of about 40%.

## Cautions

- **Thin margins.** P1 and P4 sit within 0.01 of their bars, and P3's margin over the best random direction is small. Read them as "about 0.8", "about 0.7" and "slightly above random".
- **The blind ratings are exploratory.** There were two raters, one per half of the features, and they had a different instruction from Qwen (answer literally, ignore how good the outcome is).
- **"Meaning and purpose" is itself close to an evaluation**, like F1. Without it, the blind-rated recipe drops from 0.84 to 0.79.
- One model, 120 sentences, one push layer.

## Reading

In words, Qwen prefers the outcome that adds more meaning or purpose to people's lives, deals with a more serious need, or protects health, and that is more pleasant once those are taken into account. That description predicts its choices about as well as asking it "how big a difference would this make", or reading its own states.

An outside rater's description works at least as well as Qwen's own. Qwen's descriptions of outcomes are coloured by how much it likes them. Its stated "how good" ratings give moral acts and caring for others extra weight that its choices do not.

What no description captures is the rest of the price. About half the variance in new outcomes' prices belongs to the exact wording, sometimes far off the description: a week of travel is priced near the very bottom.
