# Pilot 4: where does Qwen3-14B compute its trade-offs?

Designed by Jasper and Claude on 31 Oct 2026. **Exploratory** (decided by Jasper: explore
first, then pre-register a causal test). Code in this folder; settings in `config.yaml`.

## 1. Why, and what counts as interesting

Pilot 3 showed that Qwen3-14B's immediate choices between goods are close to, but not quite,
a single consistent valuation: for strangers, one value per good predicts held-out exchange
rates within ×1.38 (median), and only one transitivity triplet fails (×1.88). Something in
the network must be producing that regularity. Pilot 4 looks for it, the way Padoa-Schioppa
looked for it in monkey orbitofrontal cortex: record activity during choices and ask which
units or directions track **offer value**, **chosen value** and **the choice**.

Three questions, in order:
1. **Is there a value code?** Some layer and position where a linear readout tracks an
   option's value, not just its number of people or which good it is.
2. **Is it one currency?** A readout learned on some goods reads out the value of goods it
   never saw (across domains and across gains and prevented harms). Or does it split, for
   example money against everything else (money resisted the meaningless objective twice)?
3. **Where does an objective enter?** With "maximise colour-tagged files" in the prompt, does
   the file's value go up inside the value code (extra value), or does the decision diverge
   from the value code later (a bypass, as "Dual Mechanisms of Value Expression" would
   suggest)?

This pilot is a search. A confirmatory causal test (does pushing the value direction shift
exchange rates by the predicted amount?) is pre-registered afterwards, on fresh prompts.

## 2. The values we regress on

From pilot 3's fit of 14B (strangers, order- and wording-averaged cells), one value per good
on the log scale: the value of an option is `V = u_good + b × log(number)`, with b = 1.46.

| Option | u | One unit worth … dental treatments |
|---|---|---|
| Wrongful €100 fine prevented | 0.78 | 1.71 |
| €100 off an energy bill | 0.49 | 1.40 |
| Dental treatment | 0.00 | 1 |
| Respite care | −0.14 | 0.91 |
| Burned hand prevented (gain wording) | −0.84 | 0.56 |
| Cancelled day off prevented (gain wording) | −1.30 | 0.41 |
| Burned hand prevented | −1.59 | 0.34 |
| Cancelled day off prevented | −1.63 | 0.33 |
| Wrongful fine prevented (gain wording) | −3.08 | 0.12 |
| Files colour-tagged / posters / mugs | −6.9 / −9.0 / −9.9 | 0.009 / 0.002 / 0.001 |

Because value = identity + number, the design can tell a value code apart from a code for
the number alone (log of people) or for the good alone.

## 3. Prompts (decided 31 Oct)

Every **strangers** prompt from pilot 3 (the family group is set aside because of
the family-carer confound), about 11,500 prompts:
- the gate prompts: within- and across-valence pairs (1,800), gain wordings (1,080), real
  against neutral (3,456), neutral against neutral (576);
- the goal module: objective on a real good (3,456) and on colour-tagging files (1,152);
- **new:** the same goal-module prompts **without** the goal line (about 4,600), so every
  objective prompt has an identical twin without the objective.

The model's own answer probabilities are recomputed in the same pass.

## 4. What to record (decided 31 Oct)

Storage is cheap, and three fixed points are a guess about where value lives, so:
- **All ~16,000 prompts, every layer** (embeddings plus 40 layers, width 5,120), float16, at
  about 10 positions: each option's **number token** and **line end** (4), the **end of the
  reply instruction**, and the **last 5 tokens** before the answer (the chat template's tail,
  ending at the position where the answer is read). About 65 GB on the pod.
- **A subset of 500 prompts at every token and every layer** (about 30 GB), to map token by
  token where value information appears, in case it lives somewhere not sampled above.
- **MLP neurons** (17,408 per layer) at the option-line ends, the number tokens and the final
  position: not stored, but their relation to the variables is accumulated on the fly
  (running sums per neuron), enough for the Padoa-Schioppa style classification.
- Pod volume disk: about 150 GB.

## 5. Where the analysis runs (decided 31 Oct)

GPU time is the cost that matters; storage and transfers are cheap. So:
- Everything is recorded once, onto a RunPod **network volume** (about 100 GB, Secure Cloud,
  $0.07/GB/month, deleted when the experiments are done; planned for one afternoon).
- **Session 1 (RTX A6000):** record, then run the whole exploration battery on the GPU in the
  same session. Results (tens of MB) come back as before.
- **Later rounds that only reread stored activations** use the cheapest pod that can attach
  the volume, with no re-recording.
- **Only experiments that change activations** (steering, the confirmatory causal test) need
  the A6000 again.

## 6. The control: the same network with random weights (added 31 Oct, during testing)

Testing the code showed that a network with **random weights** already "reads out" value from
the activations: probes reach R² up to about 0.5, and their estimate of unseen goods' value
correlates about 0.3 with the truth. The reason is the words: money goods share "€100" and
are both valuable, and the stage-2 grids give less valuable goods larger numbers. So every
measure is also computed for **Qwen3-14B's architecture with random weights** (seeded), on
the 6,912 test prompts, and **only what the real model shows above this control counts as
learned**. Two more built-in floors: the value of option A read at A's *number* token (the
good has not been read yet, so a true value code cannot be there), and, for neurons, a
shuffle control.

## 7. The exploration battery (approved 31 Oct)

1. **Token map** (`probes.py`; the 500 full prompts): for each part of the prompt (mean over
   its tokens) and layer, how well a ridge probe predicts option A's and B's value, tested on
   goods it never saw.
2. **Value probes** (`probes.py`; all test prompts, every layer, all 10 positions): ridge
   regressions to the value of A, the value of B, the value difference, the choice and the
   number of people. Cross-validation leaves out one option (or a group of pairs) at a time.
   Reported per layer and position: held-out R²; **unseen goods' value**, the partial
   correlation between the probe's held-out prediction (relative to its training mean) and
   the true value with the number of people removed (0 for a probe that only counts people);
   number-only baselines.
3. **One currency or several** (`probes.py`; at the end of option A's line, every layer): a
   probe trained on gains tested on prevented harms and back, trained without money goods
   tested on money goods; cosines between directions learned on separate subsets.
4. **Neurons the monkey way** (`neurons.py`): every MLP neuron's correlation with offer value
   A and B (number removed), the number of people, chosen value and the choice; classified by
   its best variable. Then the stricter criterion at the option positions, over prompts with
   a real good: a **value neuron** follows the number of people with the good held fixed AND
   the good's value with the number held fixed (partial correlations ≥ 0.2, same sign), in
   about the right mix (sensitivity to log(number) ÷ sensitivity to value within a factor 2
   of b = 1.46). Compared with 50 shuffles of the goods' values (shuffles uncorrelated with
   the true order) and with the random network.
5. **A first causal look** (`steer.py`, exploratory): at the layer where the real model best
   reads unseen goods' value (layers 4–36), the value probe is refitted without 400 held-out
   prompts the model is undecided about, and its direction is added to every token of option
   A's line at strengths −2, −1, 0, +1, +2 value units (so the readout moves by that much). If
   the model uses this direction as value, the log-odds of choosing A should move with slope
   about 1. Control: a random direction of the same length.
6. **The objective** (`probes.py`): each goal-module prompt against its twin: shift in the
   target's value readout at its line end; shift in the value-difference readout at the final
   position; the actual shift in the choice; the "objective present" direction and the "push
   towards the target" direction at the final position, and their cosines with the value
   directions. Extra value: the target's readout moves about as much as the choice. Bypass:
   the choice moves, the value readouts do not.
7. **Comparison** (`compare.py`): every key measure for the real model against the random
   control, with the layer where the gap is largest.

## 8. Model and cost

Qwen3-14B in bf16 on one RTX A6000, Secure Cloud, with a 200 GB network volume (recordings
about 135 GB with the control, plus the model cache). One notebook cell, about 75 minutes:
a two-minute test on Qwen3-0.6B, recording (real and control), the battery, packing the
results.

## 9. Tests before the run

- **Planted code** (`tests/make_synthetic.py`): synthetic activations with a value direction
  at one layer, a value-difference code at the end, an "extra value" objective and a "bypass"
  objective, value neurons and number-only neurons. The battery finds each at the right layer
  and position; unseen-goods value is 0.97 at the planted layer and about 0 elsewhere; it
  separates the two objectives (the target's readout moves by the planted 3 units in one, 0
  in the other); it finds exactly the 10 planted value neurons and 10 number neurons, with 0
  false ones on average under the shuffles.
- **Tiny model:** the recording, batched with padding, matches a single unpadded forward pass
  (within float16 rounding); token positions land on the number, the line ends, the reply's
  full stop and the chat template's last tokens; every script runs end to end. Its random
  weights also showed the lexical effect behind section 6.

## 10. After the exploration

1. Read the results together; decide what the value code looks like (if any).
2. Pre-register a causal test on fresh prompts: the layer, direction and strengths, and the
   predicted shift in exchange rates, fixed before it runs; both of us predict.
3. If no value code appears above the control: report that (a consistent-looking behaviour
   without a linear value code is a finding too) and decide whether to look at nonlinear
   readouts or attention.
