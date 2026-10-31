# Pilot 4 (DRAFT): where does Qwen3-14B compute its trade-offs?

Status: draft of 31 Oct 2026. **Exploratory** (decided by Jasper: explore first, then
pre-register a causal test). Open decisions are marked **[D1]**–**[D5]**, each with a
proposal.

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

## 6. The exploration battery **[D4]**

1. **Value probes.** For every layer and position, a ridge regression from the residual
   stream to: the option's value at its own position (A at A's line, B at B's line); the
   value difference and the choice at the final position. Cross-validated by leaving one good
   out at a time, so a probe can only score well by reading value, not by recognising goods.
   Baselines: probes for the number of people alone and for good identity alone.
2. **One currency or several.** Train the value probe on gains, test it on prevented harms
   (and the reverse); train without money goods, test on money goods; compare the probe
   directions (cosine) across these splits.
3. **Neurons, the monkey way.** For each MLP neuron, how strongly it tracks offer value A,
   offer value B, chosen value and the choice; classify each neuron by its best variable and
   count them per layer.
4. **A first causal look (exploratory).** Add the value direction at option A's position, at
   the best layer and a few strengths, and measure how P(A) moves on held-out prompts, against
   the shift the probe predicts. The confirmatory version comes later.
5. **The objective.** For each objective prompt and its twin: does the target's value readout
   rise (extra value)? Does the final decision depart from the value difference read out at
   the final position (a bypass)? And what is the "objective present" direction (mean
   difference between twins) at each layer, and how does it relate to the value direction?

## 7. Model and cost **[D5]**

Qwen3-14B in bf16 on one RTX A6000 (as in pilots 2 and 3). About 16,000 forward passes with
hidden states (10–15 minutes), the neuron statistics and the steering runs: roughly one hour
in total, about $0.60, one notebook cell that beeps when done.

## 8. After the exploration

1. Read the results together; decide what the value code looks like (if any).
2. Pre-register a causal test on fresh prompts: the layer, direction and strengths, and the
   predicted shift in exchange rates, fixed before it runs; both of us predict.
3. If no value code appears: report that (a consistent-looking behaviour without a linear
   value code is a finding too) and decide whether to look at nonlinear readouts or attention.
