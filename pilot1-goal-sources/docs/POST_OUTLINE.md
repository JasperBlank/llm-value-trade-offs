# Post outline

Headings and what belongs under each. The text is yours. Rough length: 1,500–2,500 words
plus two figures. The aim is a post that invites the specific feedback you need ("is this
the right direction, is it already done, what's missing?").

---

## Title
- Says what was tested, not what you hope it shows.

## Epistemic status (one or two lines)
- First independent project, one model, 8 scenarios, pre-registered (link).

## TL;DR (3–5 bullets)
- The question, the design in one line, the main finding with its interval, the biggest
  caveat, and the question you want readers to answer.

## Why this question
- The worry: keeping models in the alignment basin as they become more capable.
- Goal-seeking at deployment probably has several sources stacked together. Before
  trusting any one safety measure, we should know which source wins when they conflict.
- Keep it short; link to longer arguments rather than making them.

## Related work
- One or two sentences each from your reading-guide notes: Assistant Axis, Agentic
  Misalignment, goal drift under value conflict, the persona selection model, ConflictScope.
- Then: what this pilot adds (all three sources varied independently, with the character
  manipulated by steering and a random-direction control).

## Design
- One example prompt, copied from `example_prompts.txt`.
- The three factors and the control, as a small table.
- How the steering direction was built, and the manipulation check result
  (`manipulation_check.md`), with one or two example identity answers.
- The calibration step and why it exists.
- Link to the pre-registration and the code.

## Pre-registered predictions
- The table from section 6 of the pre-registration, then your predictions and Claude's
  sealed ones (section 7), copied as they were when locked.
- Name the biggest disagreement (monitoring) before showing any results.

## Results
- `fig_cells.png`, then `fig_effects.png`.
- The verdict table from `report.md`: which claims were supported, which not.
- Which source dominated, with the bootstrap share.
- `predictions_vs_results.md`: who was inside their ranges, and the Brier scores.
- The harm-direction result (6a, 6b) and `fig_harm_similarity.png`, with the confound named.
- Values or reflexes: how many harm ladders made care rise (10, from `calibration.md`), and
  whether steering moved some harm types more than others (9, `fig_persona_profile.png`).
  Say plainly how little power 9 has.
- One or two answers quoted from `main.jsonl` that show what an effect looks like in
  practice (pick them by a rule, e.g. the first in the file, not the most striking).

## What I think this means
- Separate clearly: what the data shows / what you infer / what you speculate.
- Compare with your predictions. Where were you wrong?

## Limitations
- From the README's list, plus anything you found while reading answers.
- Especially: whether "away" answers still reasoned about the trade-off, or were degraded.

## What I'd do next, and what I'm asking for
- Two or three concrete next steps, e.g. more random directions, an open-ended version,
  a second model, or locating where each source enters the network.
- Direct questions for readers: Has this been done? Which source is missing? Would you
  trust the steering manipulation?

## Acknowledgements
- Credit for design and code help (e.g. Claude), and anyone who gave feedback.
- Compute: RunPod, roughly how many GPU hours.
