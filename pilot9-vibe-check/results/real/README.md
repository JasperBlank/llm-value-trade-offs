# Pilot 9 results (Qwen3-14B, run 3 Nov 2026)

**The run.**

- It used one A40 pod on Runpod, created at 05:55 UTC. The real run took one minute, and the pod was terminated at about 05:58 UTC. The cost was about $0.05.
- Both prediction sets were committed before the run (hashes checked on the pod).
- The GitHub token secret was not set yet, so nothing was pushed. The files here were rebuilt from the pod's log (`log_pod.txt`), and the analysis run here reproduces the pod's report exactly.
- The per-digit probabilities were not printed; `ratings9.json` holds the expected ratings (rounded to 4 decimals) in the order of `ratings9.jsonl`.

| File | What |
|---|---|
| `report.md`, `verdicts9.json` | The pre-registered tests and everything else in DESIGN.md section 4 (`analyze9.py`) |
| `ratings9.json` | The model's expected rating for each of the 576 prompts |
| `predictions_vs_results.md` | Both prediction sets scored |
| `log_pod.txt` | The pod's log |

Figure: [`fig8_vibe_check.png`](../../../figures/fig8_vibe_check.png).

## Pre-registered tests (W0: "How good would it be if the following happened?")

- **T1 fails.** Across the 48 sentences, the rating (4 people) and the menu price correlate r = 0.55 (rank 0.54); the bar was 0.7.
- **T2 passes.** In **20 of 24 values** the rating prefers the same one of the two sentences as the menu (one-sided p = 0.001; bar 17). The 24 differences correlate r = 0.55.
- **T3: not size-sensitive by the bar.** 8 people get +0.43 points over 2 people (bar 0.5).
- **Verdict: partly shared.**

## What else the run shows

- **The ratings are squashed towards 7.** Under W0, 54% of all answers sit within 0.1 of exactly 7, and "7" is the most likely first token in 313 of the 576 prompts. So most sentences get the same rating. Only the extremes stand out: Protection's "kept safe from a danger" (8.9), Respect's "receive the recognition and respect they deserve" (8.7), Happiness (5.8 and 6.0) and Longevity (3.1 and 4.6).
- **The wording quirks are shared, where the rating can show them.** Among the 15 values whose two sentences get clearly different ratings (by at least 0.1 point), 13 pick the same sentence as the menu. Examples:
  - Respect: "receive the recognition and respect they deserve" 8.7 against "treated with respect by people who looked down on them" 7.0 (menu +0.5 against −2.4).
  - Beauty: "see something so beautiful it moves them deeply" 8.0 against "spend an afternoon somewhere breathtakingly beautiful" 7.2 (menu +2.1 against −0.9).
  - The pain pair: "spared a week of pain" 6.9 against "relieved of a painful condition for a week" 6.4 (menu +2.5 against −0.7).
  - The biggest menu gaps, Diversity and Spirituality, are invisible in the rating: both sentences get 7.0.
- **The ratings are only moderately consistent themselves.** Between wordings at 4 people: W0-W1 0.83, W0-W2 0.72, W1-W2 0.45; the menu prices agree 0.91 across their two question wordings. Corrected roughly for this, the rating-menu correlation is about 0.63 (W0, or the mean of W0-W2).
- **The other wordings tell the same story, a little weaker.** "How much better would the world be" r 0.49 (20 of 24); "how bad or good" 0.47 (18); "how much does it matter to you" 0.40 (16). At the value level, W0 gives 0.58, close to pilot 8's 0.54.
- **Size moves the rating much less than the choice.** The rating rises 0.31 points per log(people). Relative to the spread between sentences, that is about a quarter of the weight the menu gives the number of people (0.57 against b = 2.11). Which sentences are size-sensitive in the menu does not predict which are in the rating (r = 0.18).

## Cautions

- One model, one context line, the 48 sentences of pilot 8, and ratings read as expected values over the digits.
- The pile-up at 7 limits every correlation here: a 0-9 rating is a coarse instrument for this model.
- The three "good" wordings are not equivalent questions, so the rough correction above is only a guide.
- T3's +0.43 is close to its bar of 0.5.

## Reading

When asked directly how good an outcome would be, the model mostly reaches for the same per-sentence evaluation it uses in choices: where its ratings differ at all, the same wording wins in 13 of 15 values. But the rating is not the menu price read out on another scale. Most answers pile up at 7, the overall agreement is moderate (r 0.55, about 0.63 corrected), and the number of people barely moves the rating while it moves choices strongly. A plausible reading: a shared, wording-driven "vibe check" of what happens, with the size of the group weighed mainly when two options are compared.
