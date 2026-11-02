# Pilot 6 results (Qwen3-14B, run 2 Nov 2026)

The run used an A100 SXM 80 GB pod on Runpod (US-MD-1) from 11:28 to 12:55 UTC, about $2.60. The
full run took 48 minutes, and the steer stage was rerun once more after a fix. Both prediction sets
were committed before any result was shown: the pod printed nothing but progress until they were
in the repo. These files were rebuilt from the pod's log, and `verdicts6.json` was recomputed
here; it matches the pod's copy exactly.

| File | What |
|---|---|
| `report.md` | The report (after the steer fix) |
| `report_first_run.md` | The first run's report, with the sign error in the learned push |
| `control.json` | P1: patching on the value task and the two non-value comparisons |
| `das.json` | P2: every learned subspace, plus probe and random baselines, on held-out pairs |
| `directions6.json` | P4: cosines, correlations with A's value and with V_A − V_B, and the writers' share |
| `steer6.json` | P3 (fixed), plus the exploratory decision check |
| `steer6_first_run_sign_error.json` | The first run's steering, with the learned push reversed |
| `verdicts6.json` / `verdicts6_first_run.json` | The verdicts after and before the fix (only P3 differs) |
| `k1_directions.npz` | The learned single directions (layers 28 and 32, both targets), float16 |
| `predictions_vs_results.md` | Both prediction sets scored |

## Verdicts

| | Pre-registered test | Result |
|---|---|---|
| **P1** | Value circuit or generic comparator? | **Generic comparator.** All three tasks reach the answer position at the same layer: the whole-state swap first carries half at layer 28 for value (0.77), quantity (0.79) and weight (0.73), and close to nothing at 24. Pilot 5's top-20 value pieces carry 0.64 of a swapped good, 0.54 of a swapped number of apples and 0.39 of a swapped object's weight. Twenty random pieces carry 0.00 on all three. |
| **P2** | Does one direction carry it? | **Pass, by far.** At layer 28 a single learned direction carries 0.99 [0.98, 0.99] of the whole-state swap on the new wordings, and the same at layer 32. Adding directions (k = 2-16) adds nothing. Swapping the probe's direction carries 0.01, and a random direction 0.00. |
| **P3** | Is it a better lever than the probe? | **Pass (after the sign fix).** Per natural value unit, the learned push moves the logit difference by +0.52 [0.51, 0.52], the probe's by +0.06 and a random push of the same length by +0.00: 8.6 × the probe. |
| **P4** | Real, or an illusion? | **Pass, narrowly.** On held-out prompts its coordinate correlates 0.52 with A's value (the threshold was 0.5). It lies 25.8 × chance inside the space the top-20 pieces write into; the probe lies at 2.7 ×. |

## What the direction is: the decision, not A's value

This analysis was not pre-registered; it was added together with the steer fix.

- Over held-out prompts, the learned coordinate correlates **0.99** with the model's own logit
  difference between "A" and "B". It correlates 0.67 with V_A − V_B and 0.52 with V_A.
  - 0.52 is about what a pure decision variable would show here. In the held-out design, V_A
    correlates 0.71 with V_A − V_B.
  - On pilot 4's training prompts the correlation with V_A is only 0.08. There B's value varies far
    more than A's (sd 4.5 against 1.4), so V_A alone barely predicts the choice (correlation 0.10
    with V_A − V_B).
  - So **P4 passes for a reason that does not make this a value code**. The direction tracks
    "how much the model leans to A", and A's value only in so far as that predicts the leaning.
- It is not yet the output token. Its cosine with the unembedding difference "A" − "B" is 0.09.
  At layer 28 it is an internal decision variable that the later layers turn into the letter.
- The probe for A's value still reads A's value well: 0.85 on held-out prompts. But swapping it
  carries 1% of the effect, and pushing it gives 0.06. **A's value can be read at the last
  position, but the model does not use that readable value there.** What it uses is the outcome of
  the comparison.
- The learned directions at layers 28 and 32 overlap only partly: |cos| = 0.42.

## Cautions

- With the donor's answer as the target, one direction at layer 28 carries **1.24 × the
  whole-state swap**. No natural state change produces that, so this target can find
  off-distribution levers (the Makelov et al. illusion). The main target (the whole-state swap)
  does not overshoot (0.99).
- P1's verdict uses a ratio (0.5 × the value task's top-20 transfer). Weight passes it at 0.61 ×,
  and quantity at 0.85 ×.
- The non-value questions are easy. The model's logit difference moves by 46-56 when the answer
  flips, against 6.4 for value. Transfer is a ratio, so this matters less than it would for raw
  effects.
- **Sign fix:** the first run's steer stage reversed the learned push. The raw calibration was
  −1.06; DESIGN.md section 8 has the details. The stage was rerun on the same pod with the same
  inputs, and the first run's files are kept.
