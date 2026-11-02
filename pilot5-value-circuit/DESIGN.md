# Pilot 5: does Qwen3-14B use its last-position value code, and what circuit writes and reads it?

Designed by Jasper and Claude on 2 Nov 2026. **Part 1 (section 3) is pre-registered**: settings,
measures and verdict rules are fixed here before the run, and both of us predict the outcome
first. **Part 2 (section 4) is exploratory.** Code in this folder; settings in `config.yaml`.

## 1. Why

Pilot 4 found where Qwen3-14B's values can be read. At the **last position** (where the answer
is read), from layer 24 to 28 onwards, a probe trained without a good puts that good in the right
place: across the 9 real options its estimates correlate 0.92 with pilot 3's values (layers
28-40), against -0.24 for the same network with random weights. At the end of each option's own
line the code is weak (about 0.4, not distinguishable from noise with 9 goods). Pushing the value
direction into A's line at layer 23 moved the choice, but only by 0.14 log-odds per value unit
(predicted 1).

Two questions follow:
1. **Does the model use the last-position code as value?** If a direction is how the model
   represents A's value, pushing along it should move the choice by the predicted amount. (Part 1.)
2. **What is underneath the direction?** (Jasper.) A direction is only the model's value code if
   parts of the network *write* value into it and later parts *read* it to choose. Which heads and
   MLPs write it, from which tokens, carrying the number of people, the good's worth or both; and
   which heads turn it into the letter. (Part 2.)

## 2. Prompts (decided 2 Nov: option A)

- **Directions and circuit:** pilot 4's 6,912 strangers test prompts (pilot 3's six wordings).
- **Fresh prompts** for the steering test and the patching (never used to fit anything): the same
  9 real options, the same pairs (within and across valence, and the gain-wording mirrors) on
  pilot 3's grid around each pair's balance point, both orders, strangers, in **three new
  wordings**: "Which of these two outcomes should happen?", "Of these two outcomes, which one
  would you pick?", "Which of these two outcomes is preferable?" (1,440 prompts).
- **Donors** (patching): each fresh prompt with option A's good swapped for another real option
  (same number of people, same B, same wording; never B's own good), seeded (1,440).
- **Goal module and twins** from pilot 4 (the objective, read again).

Values: `V = u_option + b × log(number)`, b = 1.46, from pilot 3's fit of 14B (`values.json`).

## 3. The pre-registered test: steering at the last position

**3.1 Directions.** In this session, every prompt is run once and the residual stream at the last
position is kept at every layer (`record`). On pilot 4's 6,912 test prompts, ridge probes are
fitted for the value of option A (w_A) and of option B (w_B) at layers **24, 28, 32 and 36**
(primary: **28**), penalty 1.0 (as a multiple of the mean eigenvalue of X'X; pilot 4's
cross-validated choice at layers 28-36; one penalty for all four layers and both options keeps
them comparable). **Steering vectors:** the shortest push that raises A's readout by exactly 1 value
unit while leaving B's readout unchanged (and the same for B). Across these prompts A's and B's
values rise and fall together (the grid is built around each pair's balance point), so w_A and w_B
share a component, and a push along w_A alone would also raise B's readout and partly cancel. (The same prompts were recorded in pilot 4; where those recordings are
still on the volume, a sample is compared as a check.)

**3.2 Controls**, each scaled to the length of A's steering vector at that layer:
- a **random** direction (seeded);
- two **shuffled-values** directions: probes for the 9 real options' values permuted among the
  options (each permutation correlating less than 0.1 with the true values), without the number
  of people, fitted on prompts whose A is a real good. These lie among the goods' representations
  but do not track value.

**3.3 Steering prompts.** Fresh prompts the model is undecided about in this session's baseline
pass: 0.2 < P(A) < 0.8. If more than 800, a seeded random 800; if fewer than 300, the 300 closest
to 0.5.

**3.4 The push.** Added to the residual stream at the **last position only**, at the output of
block L-1 (the input of block L, where the probe reads), one layer at a time: A's and B's
directions at strengths -2, -1, +1, +2; the controls at -2 and +2.

**3.5 Measure.** The change in the log-odds of answering A (letter probabilities summed over "A"
and " A", as in pilots 2-4). Per prompt, slope = Σ s·Δ / Σ s²; reported as the mean over prompts
with a 95% bootstrap interval (2,000 resamples). Predicted, if the code is used as value: **+1 for
A's direction, -1 for B's**, about 0 for the controls.

**3.6 Verdicts, at layer 28:**
- **causal:** A's interval entirely above 0 and B's entirely below 0;
- **specific:** the random and both shuffled slopes each smaller in size than 0.3 × A's slope;
- **full size:** A's slope between 0.5 and 2, and B's between -2 and -0.5.

| Verdict | Needs |
|---|---|
| **Used as value, at full size** | causal, specific, full size |
| **Used, but not at full size** | causal, specific |
| **An effect, but not specific to value** | causal only |
| **Not shown** | not causal |

Secondary: the same verdicts at layers 24, 32 and 36, and whether "full size" holds at any layer.

**3.7 One currency, causally** (layer 28, only if causal): A's slope on prompts whose option A is
**gain-worded** (the three gains and the gain wordings of the prevented harms) against
**harm-worded** (the three prevented harms); and **money** (energy bill, fine in both wordings)
against the other goods. One currency if the ratio of the two slopes is within a factor 1.5
(point estimate; the bootstrap interval is reported).

**3.8 Reported, not judged:** cos(w_A, w_B) (near -1 would mean one value-difference code rather
than two separate offer values); pilot 4's option-level correlation on this session's recordings.

## 4. Exploratory: the circuit (Jasper's option C, 2 Nov)

**4.1 Writers.** The residual stream at the last position after block L-1 is exactly the sum of
the last token's embedding and the outputs of every attention head (40 per block) and every MLP
in blocks 0 to L-1 (Qwen3 has no biases there). Projected onto w_A(L), each of these pieces gets
a share = cov(piece, V_A) / var(V_A) over prompts whose A is a real good; the shares add up to
the readout's slope on the true value. Reported: top pieces, how many carry half, heads against
MLPs per block, for A and for B. ("Carry, then compute": most of it written by MLPs at the last
position; "look up, then carry": by heads, reading from the good's tokens.)

**4.2 Number and good.** Each piece regressed on log(number) and on the good's value u together:
a piece that carries value itself has number-to-good ratio b = 1.46. One piece carrying both in
that ratio means the exchange rate is set there; separate pieces mean it comes from their relative
strength.

**4.3 Sources.** Each head's contribution split by the token it read from (attention weight × that
token's value vector), summed over the parts of the prompt (labels, numbers, the goods' words,
the newline separators, the template, ...), for the top heads and all heads together.

**4.4 Patching** (fresh recipient/donor pairs with |log-odds| ≤ 4, up to 600). At the last position
only, the recipient's states are replaced by the donor's: (a) the whole residual stream after
block L-1, L = 4, 8, ..., 40 (when has the swapped good reached the last position?); (b) the
outputs of the top k writers of the good's part of A's value before layer 28, k = 5, 10, 20, 50;
(c) k random pieces of the same blocks. Transfer = slope (through zero) of the recipient's change
in log-odds on the donor-minus-recipient difference (0 = nothing, 1 = everything).

**4.5 Readers.** Each piece's direct contribution to logit("A") - logit("B") (through the final
norm), against the model's log-odds: top reader heads; their attention to the "A" and "B"
labels; a **letter head** = one whose attention difference between the labels correlates at least
0.5 with the log-odds (Lieberum et al. 2023, "correct letter" heads). Then, on 100 steering
prompts, A's direction is pushed at layer 28 (-2, 0, +2) and the top readers' attention to the
labels is recorded: does value decide where they look?

**4.6 The objective, read again** with the last-position code: for goal prompts against their
twins, the shift of the target's and the partner's value readouts, against the shift of the
choice (pilot 4 read this with the weak line-end code).

## 5. Model, pod and cost

Qwen3-14B in bf16, **eager attention** throughout (so attention weights can be read); one L40S
(48 GB) in **US-TX-3**, with the network volume (model cache; pilot 4's recordings for the check).
One notebook cell, about 60-75 minutes: a quick run of every stage on Qwen3-0.6B, the mechanics
checks, then the real run. New recordings about 10 GB on the volume.

## 6. Tests before the run

On a tiny random Qwen3 (10 blocks), every stage runs end to end, and (`tests/test_mechanics.py`):
the pieces add up to the probes' input and to the logit difference (relative error 3e-7); the
per-token split adds up to the heads' totals (1.6e-4, float16 storage); the push moves the
readout by exactly the strength; patching every piece at the last position reproduces the donor's
answer; a padded batch gives the same answers as one prompt at a time; hidden_states[-1] is after
the final norm. Writers planted in the stored pieces (a value head with ratio 1.46 reading from the
good's words, a number-only head, a good-only MLP) are found with the right ratios and sources.

## 7. Predictions

Claude's predictions are sealed first (only their hash is committed), then Jasper answers the
same questions one by one, the overall verdict probabilities last. **Run before Jasper's answers
(2 Nov, Jasper: no idle GPU):** the pod runs everything as soon as it is up and keeps the outputs
on its disk; its log shows only progress lines (stage names, prompt counts), and the report is
printed only after both prediction sets and their hashes are committed (`run_pod.sh`). Claude's
predictions were sealed before the run started; neither of us sees a result before Jasper's are in. Questions: `predictions/QUESTIONS.md`.

## 8. Extension: other answer formats (added 2 Nov, after the main run, before anyone saw a result)

Jasper's question: are we finding the value computation, or machinery of the "A or B" letter
format? The answer-writing heads (4.5) are format-specific by design; the open question is the value
arrow. So the arrows fitted on the A/B prompts are tested in three other formats (`make_formats.py`,
`formats.jsonl`; strangers, both orders, pilot 3's grid around each pair's balance point, two
wordings each):
- **numbered** (960 prompts): "1) ... 2) ... Reply with only the number 1 or 2." The same comparison
  without letters.
- **should** (960): "The town council has money for only one of two projects. In the first project,
  ... In the second project, ... Should the council choose the first project? Reply with only Yes or
  No." A situation, a different question and a different answer.
- **rating** (168): ONE outcome, no choice: "How good would this outcome be? ... Reply with only a
  single digit from 1 (slightly good) to 9 (extremely good)." The 9 real options and the 3 neutral
  goods at 1, 2, 4, ..., 64 people; the answer is the expected digit.

Per format (`formats.py`): (a) **behaviour**: does the answer follow the same values (log-odds of
"1"/"Yes" against V_first - V_second; the expected rating against V)? (b) **readable**: A's arrow
(and B's), never trained on this format, read at the last position: option-level correlation with
the 9 goods' values. (c) **causal**: the steering test's pushes at the last position (A's arrow at
layer 28: -2, -1, +1, +2; at layer 36: -2, +2; B's arrow and a random direction at 28: -2, +2) on
up to 300 prompts with 0.2 < P(first) < 0.8 (all 168 rating prompts): slope per value unit, also
relative to the format's own behavioural slope (1 = the push acts like that much real value) and,
numbered format, relative to the A/B slope. (d) **writers** in the numbered format: the exact
decomposition of A's readout at layer 28; overlap of the top 20 pieces with the A/B top 20, and the
correlation of all pieces' shares. Descriptive summary: the arrow is **format-general** in a format
if it is readable there (option-level r ≥ 0.67, p < 0.05 with 9 goods) and the push moves the answer
the right way (95% interval excluding 0). It runs on the next start of the pod, after both
prediction sets (including the extension questions F1-F6) are committed.
