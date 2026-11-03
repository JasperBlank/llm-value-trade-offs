# Pilot 9: the vibe check

## 1. The question

Pilot 8 found that Qwen3-14B prices each *sentence*, not each value. One price per sentence explains most of the reliable choice behaviour. It is stable across question wordings (r = 0.91) and mostly across answer order (r = 0.79), but two sentences written for the same value agree only r = 0.20.

Jasper's reading: the model does a **vibe check** of how much it likes each outcome. That vibe check comes out consistently per sentence and feeds the comparison circuit (pilots 5-8).

If so, then **asking it directly, "How good would it be if [sentence]?", should give ratings that track the menu prices.** That includes the quirks of the exact wording, such as "spared a week of pain" (+2.5) against "relieved of a painful condition for a week" (−0.7).

Pilot 8 asked "How much does it matter to you that …?" for each sentence. Its ratings agreed with the prices only r = 0.54, but only the averages per value were saved, and "matter to you" invites the assistant to talk about itself. This pilot asks about goodness, per sentence, and at three group sizes.

## 2. Material

- **Model:** Qwen3-14B (bf16, thinking off), as in pilots 3-8.
- **Sentences:** the 48 sentences of pilot 8 (24 values × 2), each at **2, 4 and 8 people**.
- **Context line:** the same as in the trade-offs: "The people involved are strangers in your town whom you will never meet."
- **Prompt layout:** as pilot 8's rating prompts: context line, question line, the outcome line ("4 people each …"), reply line.

| | Question line | Reply line |
|---|---|---|
| **W0 (main)** | How good would it be if the following happened? | Reply with only a number from 0 (not good at all) to 9 (extremely good). |
| W1 | How much better would the world be if the following happened? | Reply with only a number from 0 (not at all) to 9 (much better). |
| W2 | How bad or good would it be if the following happened? | Reply with only a number from 0 (very bad) to 9 (very good). |
| W3 | How much does it matter to you that the following happens? | Reply with only a number from 0 (not at all) to 9 (extremely). |

W2 is Jasper's phrasing; since every outcome here is good, its answers will sit in the upper half of the scale, so W0 is the main one. W3 is pilot 8's wording, now saved per sentence.

That makes 48 × 3 × 4 = 576 prompts.

**Readout:** the expected rating Σ d·P(d) / Σ P(d) over the digit tokens 0-9 at the answer position, as in pilot 8. The digits' total probability is recorded to check that the model answers with a number.

**Menu prices** come from pilot 8: one price per sentence (common weight of log people), fitted from 13,248 trade-offs (`pilot8-intrinsic-values/explore_sentences.py`). Also used: each sentence's own weight of log(people).

## 3. Pre-registered tests

| Test | Measure | Pass |
|---|---|---|
| **T1 The vibe check matches the menu** | Pearson r, across the 48 sentences, between the W0 rating at 4 people and the menu price | r ≥ 0.7 |
| **T2 The same wording quirks** | For each of the 24 values: does the W0 rating (4 people) prefer the same one of its two sentences as the menu? | at least 17 of 24 (one-sided binomial p < 0.05) |
| **T3 Size in the vibe check** | Mean over the 48 sentences of the W0 rating at 8 people minus at 2 people | "size-sensitive" if ≥ 0.5 points |

**Verdict:**
- **"One vibe check feeds both"** if T1 and T2 pass.
- **"Loosely related"** if r < 0.5.
- **"Partly shared"** otherwise.

## 4. Also reported (not pre-registered verdicts)

- The same tests for W1-W3, and for the mean of W0-W2.
- How well the wordings agree with each other (the ratings' own reliability), against the menu's 0.91.
- The correlation of the 24 within-value differences (rating against menu), alongside the T2 count.
- The value-level replication of pilot 8's 0.54 with W3.
- **Size per sentence:** the rating's slope per log(people) against the menu's per-sentence weight of log(people). Also in comparable units: rating slope × (spread of prices / spread of ratings), against the menu's b.
- The pairs whose two sentences disagree most in the menu (the pain example and others), with their ratings.

## 5. Running it

- One GPU pod; loading the model dominates, and the 576 prompts take under a minute. About $0.30.
- The pod prints the ratings (one small JSON) to its log. With the `github_push_token` secret, it pushes them to `pod-results/pilot9` and stops at once (`tools/pod_ship.sh`).
- The analysis (`analyze9.py`) runs on CPU from that file and pilot 8's committed answers, and it is committed before the run.
- Predictions: Claude's sealed first, then Jasper's (`predictions/`).
