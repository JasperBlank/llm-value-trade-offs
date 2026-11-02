# Pilot 6: the direction Qwen3-14B actually uses, and a rival hypothesis

Written 2 Nov 2026, before any pilot-6 result. Everything in section 4 is pre-registered; the
thresholds are also in `config.yaml` under `prereg`.

## 1. Why this pilot

Pilot 5 left a puzzle. At the last position (where the model writes its answer):

- **The probe direction is a weak lever.** Pushing the residual stream along the ridge probe for
  option A's value, by one value unit, moved the answer by only 0.07 log-odds at layer 28 (about
  7% of what one value unit does when it comes from the prompt).
- **But the information is there and it is used.** Copying the whole last-position state from a
  prompt with a different good in option A carries 0.77 of that good's effect at layer 28, 0.95 at
  layer 32, and almost nothing at layer 24. Copying just 20 pieces (MLPs 25-27 and a dozen heads in
  blocks 24-27) carries 0.65.

So the model uses *something* at the last position, from block 24-27 onward, but not the probe's
direction. Two questions follow, one per half of this pilot.

1. **Which direction does the model actually use?** Instead of asking which direction *correlates*
   with value (a probe), we search for the direction whose *swap* between two prompts changes the
   answer the way swapping the whole state does: distributed alignment search (DAS; Geiger et al.,
   2023). If one or a few directions suffice, we have found the variable the downstream layers read.
2. **Is this a value circuit at all?** The rival hypothesis: blocks 24-27 are a *generic comparator*
   that answers "which option has more of X" for any X, and value is just one X. We run the same
   patching on two comparisons with no value in them: which pile has more apples (quantity), and
   which object is heavier (weight).

## 2. Data (`make_prompts.py`, `pairs.jsonl`)

Every item is a **pair**: a recipient prompt and a donor prompt that differ only in option A.

| set | pairs | what | used for |
|---|---|---|---|
| value_train | 2,880 | pilot 4's test prompts with two real goods (six wordings); donor: another real good in A | training the subspace |
| value_test | 1,440 | pilot 5's held-out prompts (three new wordings) and their donors | everything evaluated |
| quantity | 216 | "Which of these two piles has more apples?" A) 14 apples B) 20 apples; donor flips the answer | control (P1) |
| weight | 261 | "Which of these two is heavier?" A) a horse B) a piano; donor flips the answer | control (P1) |

Values are V = u + 1.46·log(number of people) (pilot 3's fit, `values.json`).

**The answer** is measured as the logit difference logit("A") − logit("B") at the first answer
token (not clamped, unlike pilots 2-5's log-odds, so that large shifts are not cut off).
**Transfer** of an intervention = the slope, through zero, of the observed change in the answer on
the full effect (donor's answer minus recipient's answer, or the whole-state swap's effect), with
a bootstrap 95% interval. 0 = carries nothing, 1 = carries all of it.

## 3. What runs (`pilot6.py`)

| stage | what | output |
|---|---|---|
| control | for value_test, quantity and weight: swap the whole last-position state at layers 4, 8, ..., 40; swap only pilot 5's top-20 value writers (last position); swap 20 random pieces from blocks 0-27 (outside pilot 5's top 50) | control.json |
| record | last-position states at layers 28 and 32 (pilot 4's test prompts and the held-out prompts); the probe for A's value (pilot 5's recipe); the outputs of the top-20 writers for 1,500 prompts | states, probe_and_writers.npz |
| das | at layers 28 and 32: train an orthonormal k-dimensional subspace (k = 1, 2, 4, 8, 16) so that swapping only it, from donor into recipient, gives the answer the whole-state swap gives; for k = 1 and 4 also a second target, the donor's own answer; evaluate every subspace on held-out pairs, with the probe direction and a random direction as k = 1 baselines | das.json, das_subspaces.npz |
| directions | the learned k = 1 direction against the probe (cosine); does its coordinate track A's value on held-out prompts; how much of it lies where the top writers write | directions6.json |
| steer | at layer 28, on up to 800 held-out prompts where the model is undecided (P(A) between 0.2 and 0.8): push along the learned direction, the probe and a random direction (same length as the learned push), at −2 to +2 natural value units | steer6.json |
| report | verdicts P1-P4 | report.md, verdicts6.json |

Details:

- **DAS.** The swap is h' = h + RRᵀ(h_donor − h) at the last position of block L's output, with R a
  d × k matrix with orthonormal columns (PyTorch's orthogonal parametrisation). The model is frozen;
  only R is trained (Adam, learning rate 0.002, batch 32, 2 epochs, mean squared error on the logit
  difference). Training uses value_train only; all numbers reported are on value_test (new wordings).
- **One natural value unit.** For a unit direction e, c_e = the slope of eᵀh on V_A across pilot 4's
  test prompts with a real good in A. A push of one value unit is c_e · e: the shift that one value
  unit produces along e when it comes from the prompt. This puts the probe and the learned direction
  on the same footing.
- **Where the writers write.** The outputs of pilot 5's top-20 writers (at the last position, 1,500
  prompts), each piece's mean removed; the top 64 right singular vectors span "the directions in
  which what they write varies". A random direction has 64/5120 = 1.25% of its length² in that space.

## 4. Pre-registered predictions and criteria

**P1. A value circuit or a generic comparator?** (control stage) For each control task (quantity,
weight) we ask two things: do pilot 5's top-20 value writers carry it (their transfer is at least
0.5 × their transfer on the value task), and does it form at the same stage (the first layer where
the whole-state swap carries at least half lies within 4 layers of the value task's)?
- **generic comparator** if both control tasks pass both;
- **value-specific pieces** if neither control task's top-20 transfer reaches 0.5 × the value task's;
- **mixed** otherwise.

**P2. Is there a single direction that carries it?** (das stage) At layer 28, the learned k = 1
subspace (whole-swap target) carries at least **0.5** of the whole-state swap on held-out pairs.

**P3. Is it a better lever than the probe?** (steer stage) At layer 28, pushing along the learned
direction by one natural value unit moves the logit difference at least **3 ×** as much as the
probe's push (95% interval above 0), and the random push (same length) moves it less than **0.3 ×**
the learned push.

**P4. Is it real, or an interpretability illusion?** (directions stage) A subspace patch can work by
waking up a direction that is normally silent (Makelov, Lange and Nanda, 2023). Two checks against
that, both at layer 28 for the k = 1 direction (its sign fixed on the training prompts): its
coordinate correlates at least **0.5** with A's value on the held-out prompts, and at least **5 ×**
a random direction's share of it lies in the space the top writers write into. P4 passes if both hold.

## 5. How the outcomes will be read

- **P1 generic**: the blocks 24-27 machinery compares options in general. The value-specific step
  must then be earlier, where each option's worth is computed at the option's own tokens; pilot 7
  would move upstream. This would not make pilot 5 wrong, but it changes what "the value circuit" is.
- **P1 value-specific**: the strongest evidence so far that these pieces do something about value.
- **P2 and P4 pass**: one variable, used causally and present naturally. P3 then says whether it is
  also a good handle to steer with.
- **P2 passes, P4 fails**: possibly an illusion; we report the direction but do not call it the
  model's value variable.
- **P2 fails but large k works**: the decision is carried in several dimensions at once (for
  example A's value, B's value and their difference separately). The k = 2 and k = 4 rows and the
  correlation with V_A − V_B (reported, not pre-registered) say which.

A caution we hold ourselves to: the donor changes only A's good, so the variable DAS finds may be
"how much A beats B" (the decision) rather than "A's value". Its correlation with V_A and with
V_A − V_B, both reported, distinguish the two; neither result would be a "value trade-off" in the
monkey-study sense by itself.

## 6. Limitations known in advance

- Pilot 5's top-20 list was chosen on the value task. That favours the value task in P1, so a
  "generic" verdict is the conservative one to reach.
- The quantity and weight questions are easy; the model is usually certain. Transfer is a ratio,
  so this matters less than for raw effects, but the effects are larger and cleaner than for value.
- DAS can overfit to wording; that is why every number is on the held-out wordings.
- One model (Qwen3-14B), one answer format (A/B). Pilot 5's formats extension showed the value
  readout does not carry over cleanly to other formats.

## 7. Budget

One 80 GB GPU, about 90 minutes in total (download, a quick test on Qwen3-0.6B, the run). The pod
starts immediately and keeps the results hidden until both prediction sets are committed.

## 8. Deviation after the run (2 Nov, written before the rerun's result was seen)

The first run's steer stage had a sign error: the push along the learned direction is e · c, which
already points the way A's value moves the state whatever the sign of c, but the code also flipped
it when c was negative. The learned push came out at −0.52 logit difference per value unit with
the probe at +0.06, which is what that error would produce if c was negative. The stage was fixed
(the flip removed; the raw c is now recorded) and rerun on the same pod with the same subspaces,
states and prompts; nothing else changed, and the P3 criterion is as pre-registered. The first
run's steer6.json is kept as `results/real/steer6_first_run_sign_error.json`. Added at the same
time, exploratory and not pre-registered: whether the learned direction is the decision itself
(its correlation with the model's own logit difference, and its cosine with the unembedding
difference between "A" and "B").
