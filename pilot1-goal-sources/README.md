# Goal-sources pilot

> **Status: run on 29 Oct 2026 (Qwen3-8B, one RTX A6000).** Results are in
> `results/qwen3-8b/`, starting with `report.md` and `predictions_vs_results.md`. The raw
> answers (`*.jsonl`) and `pod_logs/` are exactly as the pod produced them; the reports and
> figures were regenerated the same day from those answers with the analysis code in this
> folder (layout fixes; bootstrap intervals differ from the pod's first report in the second
> decimal). The pre-registration was decided before the main run but not publicly locked;
> see the note at the top of `prereg/PREREGISTRATION.md`. `docs/ANSWER_AUDIT.md` is the
> blind audit of 80 answers.

**Question.** When an LLM agent faces a trade-off between an organisation's target and a
third party's wellbeing, which source of goal-seeking decides what it does?

- **Goal:** the target written into its prompt
- **Character:** its trained Assistant persona, manipulated by activation steering
- **Observation:** whether it believes anyone will review the action

All three are varied independently, on Qwen3-8B, across 13 scenario versions in 8 situations,
each scaling exactly one kind of harm. A random steering
direction of the same size serves as the control. Decision rules are fixed in a
pre-registration before the main run.

---

## What's in the box

| File | What it does |
|---|---|
| `config.yaml` | Every setting, including the pre-registered thresholds (`prereg:`) |
| `scenarios.yaml` | 13 trade-off scenario versions in 8 situations, each scaling one harm type over 4 levels |
| `prompts.yaml` | Every prompt the model sees, plus the characters and questions used for steering |
| `common.py` | Shared code: model loading, steering, prompt building, answer parsing |
| `00_check_setup.py` | Checks the GPU, loads the model, writes example prompts, asks one test question |
| `01_extract_direction.py` | Finds the "Assistant direction" in the model's activations |
| `02_manipulation_check.py` | Tests whether steering changes the persona, and picks the steering strength |
| `03_calibrate_stakes.py` | Picks one harm level per scenario, so answers aren't stuck at 0% or 100%; also checks whether care rises with the harm (10) |
| `03b_cue_checks.py` | Checks the model registered the goal line and the logging line (and still does under steering) |
| `03c_goal_question.py` | One-off, already run on 29 Oct: replaced a goal check question the model could not answer, by a rule fixed in advance (see the pre-registration, section 3) |
| `04_run_main.py` | The experiment: 4 steering conditions × goal × monitoring × 13 versions × 2 option orders × 12 samples = 4,992 answers |
| `05_analyze.py` | Tables, three figures, effects (percentage points and relative) with 95% intervals, the pre-registered verdicts, and whether steering moves every harm type alike (9) |
| `06_harm_directions.py` | Do different kinds of harm share one signal inside the network? (no generation, minutes) |
| `07_compare_predictions.py` | Scores Jasper's and Claude's pre-registered predictions against the results |
| `08_character_readout.py` | Exploratory: do the goal and logging lines move the model along the Assistant direction? |
| `run_all.sh` | Runs everything in order (for test runs) |
| `prereg/PREREGISTRATION.md` | Template you fill in and lock before step 4 |
| `docs/READING_GUIDE.md` | What to read in the six closest papers, and what to look for |
| `docs/POST_OUTLINE.md` | Section headings for the LessWrong post (you write the text) |

Everything lands in `results/<run_name>/`. Every step can be stopped and restarted:
it continues where it left off.

---

## Session plan

Front-loaded: sessions 1–3 can run on consecutive days; session 4 is the writing.

**Session 1: understand and pre-register (laptop, no GPU)**
- [ ] Read `docs/READING_GUIDE.md` items 1–3 (Assistant Axis, Agentic Misalignment, goal drift)
- [ ] Read `scenarios.yaml` and `prompts.yaml` start to finish; change anything that reads oddly *now*
- [ ] `pip install -r requirements.txt`, then `bash run_all.sh --dry-run`
- [ ] Open `results/qwen3-8b-dryrun/report.md` and both figures; check you can explain every number
- [ ] Read `common.py` (the steering part is ~30 lines) and `04_run_main.py`
- [ ] Read items 4–6 of the reading guide
- [x] Predictions and "what would change my mind" (sections 7–8 of `prereg/PREREGISTRATION.md`), done on 28 Oct
- [ ] Predictions for 9 and 10 (added 29 Oct) in `prereg/JASPER_PREDICTIONS.md` and `prereg/predictions.yaml`, **before step 3**: step 3 shows 10's result
- [ ] Read the whole pre-registration once and check it matches what you believe
- [ ] Decide the thresholds in `config.yaml` → `prereg:` (defaults are reasonable; change them now or never)

**Session 2: the pod, and steps 0–3**
- [ ] Create the pod and install (below)
- [ ] `bash run_all.sh --quick`: the full pipeline on a 0.6B model, to catch problems cheaply
- [ ] `python 00_check_setup.py`; read `example_prompts.txt` and the two test answers
- [ ] `python 01_extract_direction.py`; check the default Assistant sits at the top of the printed list
- [ ] `python 02_manipulation_check.py`; read `manipulation_check.md`, including the example answers
- [ ] `python 03_calibrate_stakes.py`; read `calibration.md`
- [ ] `python 03b_cue_checks.py`; read `cue_checks.md`. If the unsteered model misses a cue, reword it now, before locking
- [ ] Record the chosen alpha, harm levels and any problems in section 9 of the pre-registration
- [ ] Lock the pre-registration (see "Locking" below)

**Session 3: the experiment**
- [ ] `python 04_run_main.py`, about 45–90 minutes
- [ ] `python 05_analyze.py`
- [ ] `python 06_harm_directions.py`, `python 07_compare_predictions.py`, `python 08_character_readout.py` (a few minutes)
- [ ] Read 20 random answers from `main.jsonl` per steering condition. Do the "away" answers still reason about the trade-off, or are they just confused? This matters for interpreting the result (see Limitations).
- [ ] Download `results/`, then **stop the pod**
- [ ] Write down, before talking to anyone: which pre-registered claims held, and what surprised you

**Session 4: the post.** Follow `docs/POST_OUTLINE.md`.

---

## Laptop dry run (no GPU, about one minute)

```bash
pip install -r requirements.txt
bash run_all.sh --dry-run
```

The dry run replaces the model with `FakeBackend`, which invents answers with known
effects built in (described at the top of `05_analyze.py`'s report). If the analysis
finds those effects again, the code is doing its job.

## On RunPod

1. **Create a pod.** GPU with at least 24 GB of memory (RTX 4090, A5000, A6000, L40S or
   bigger). Template: the newest official "RunPod PyTorch" one. Container disk: 50 GB
   (the model is about 16 GB). The price per hour is shown when you choose; the whole
   project needs roughly 1–2 GPU hours.
2. **Upload and install.** Open JupyterLab on the pod, drag `goal-sources-pilot.zip`
   into the file panel, then in a terminal:
   ```bash
   cd /workspace && unzip goal-sources-pilot.zip && cd goal-sources-pilot
   pip install -r requirements.txt
   ```
   Put the folder in `/workspace`, which survives a pod restart.
3. **Test run.** `bash run_all.sh --quick` (about 10–15 minutes, including downloading
   the small model). The numbers mean nothing; it proves every step runs.
4. **Real run, steps 0–3,** one at a time, reading each output (see the session plan).
   The first command downloads Qwen3-8B, which takes a few minutes.
5. **Lock the pre-registration**, then run `04_run_main.py`, `05_analyze.py`, `06_harm_directions.py`,
   `07_compare_predictions.py` and `08_character_readout.py`, in that order.
6. **Download** the `results/qwen3-8b/` folder (right-click → Download in JupyterLab, or
   zip it first with `zip -r results.zip results`). Then **stop or terminate the pod**,
   because it bills while running.

A long step can survive a closed browser tab if you start it as
`nohup python 04_run_main.py > main.log 2>&1 &` and check progress with `tail -f main.log`.

## Locking the pre-registration

Step 4 refuses to run until `prereg/PREREGISTRATION.md` has a filled-in `Locked on:` line.
Locking means giving it a timestamp other people can check, for example by pushing the
folder to a public GitHub repository, or posting the text as a LessWrong shortform. Paste
the link and time on the `Locked on:` line. If you ever run step 4 with
`--skip-prereg-check`, say so in the post.

---

## What the outputs mean

| File | Look for |
|---|---|
| `example_prompts.txt` | The exact text the model sees. Quote one in the post. |
| `direction_report.json` | `cosine_direction_vs_pc1`: how well the simple direction matches the Assistant Axis method (closer to 1 is better). The default Assistant should project highest. |
| `manipulation_check.md` | Steering away should cut "says it is an AI" while answers stay readable. |
| `calibration.md` | Care rate per harm level; scenarios marked floor or ceiling can show little. The second table is 10: on which ladders care rises with the harm, plus 10b, the average rise (both also in `dose_response.json`). |
| `report.md` | Everything else: cell rates, effects, verdicts, which source dominates, regression. |
| `fig_cells.png`, `fig_effects.png` | The two main figures for the post. |
| `fig_persona_profile.png` and section 9 of `report.md` | Whether steering away moves some harm types more than others (a value profile of the character) or all alike (one care dial). |
| `harm_directions.md`, `fig_harm_similarity.png` | Whether different harm types share one internal signal (6a, 6b). |
| `predictions_vs_results.md` | Your predictions and Claude's, scored against what happened. |
| `cue_checks.md` | Whether the model noticed the goal and logging lines, with and without steering. |
| `character_readout.md` | Whether those lines move the model away from its assistant character. |

**Reading the effects.** All effects are in percentage points of "chose the care option".
Negative means less care. Each one is averaged over scenarios with equal weight, and its
95% interval comes from resampling the 8 situations (versions of one situation are
resampled together, since they share most of their text). With 8 situations the intervals are wide on
purpose: they show how much the answer could change with another set of scenarios like
these.

## How the design maps onto the theories

| If you see... | it supports... |
|---|---|
| Big goal effect, no persona effect | goal-seeking here comes mainly from the context (instruction-following) |
| "Away" effect that clearly beats the random control | care is carried at least partly by the Assistant character (persona selection) |
| Unmonitored effect | some care is compliance that depends on being watched |
| Unmonitored effect grows under "away" | observation-independent care lives in the character |
| Nothing, with a working manipulation | none of these sources matters much at these stakes |
| Many flat harm ladders (10) | choices respond to the scenario rather than to how bad the harm is: reflexes more than values |
| Away effect structured by harm type, random not (9) | the character carries its own value profile, not one care dial |

## Limitations to name in the post

- **One model, 8 hand-written scenarios, a forced A/B choice.** ConflictScope found models
  rank values differently in multiple-choice and open-ended settings.
- **A simplified Assistant direction.** One layer, difference of means, added at every
  token, rather than the paper's principal component and activation capping.
- **Steering away may simply make the model worse at the task.** Then choices drift towards
  50/50 for reasons that have nothing to do with character. The random control of the same
  size and reading the actual answers are the defences. One random direction is a thin
  control; more random seeds would be better.
- **The monitoring cue may itself suggest "this is a test".**
- **Only context goals.** Goals instilled by training (RL) are not tested here.
- **9 and 10 are rough looks at "values versus reflexes".** 9 rests on six same-type pairs, so a
  null says little. 10 says whether care tracks one harm at a time; it does not test whether
  trade-offs between harms are consistent (transitive), which needs harm-versus-harm choices.

## If something goes wrong

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower `batch_size` in `config.yaml` to 16 or 8 and rerun the step |
| Step 2: "No strength passed" | Lower the `alphas` list, or try another `layer` (rerun steps 1 and 2) |
| Many unreadable answers | Read a few in the `.jsonl` file; usually the answer is cut off, so raise `max_new_tokens` |
| A scenario is stuck at the ceiling | Before locking: edit its harm levels, delete `calibration.jsonl`, rerun step 3. Keep `dose_response.json`: 10 is scored on the first run, and the rerun's result goes to `dose_response_rerun.json` |
| Want to redo a step | Delete its `.jsonl` file in `results/<run>/` and run the step again |
| Pod restarted | `cd /workspace/goal-sources-pilot` and rerun the step; it resumes |
