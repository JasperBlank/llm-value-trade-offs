# Pilot 12 results (Qwen3-14B, run 3 Nov 2026)

**The run.**

- The first pod was an A40. Three times Runpod placed it on the same machine, which never started the container, so each attempt was terminated after 1-10 minutes: about 12 minutes of billing, roughly $0.12.
- The run itself used an L40S, from 08:49 to 08:52 UTC (about $0.05).
- Both prediction sets were committed before the run, and the hashes were checked on the pod.
- This is the first run that **pushed its results itself**. The pod committed everything, the model's states included, to the branch `pod-results/pilot12` and stopped itself straight away. These files are copied from that branch unchanged.

| File | What |
|---|---|
| `report.md` | The pod's report: Q1, Q2, the four pushes, Q1 by layer, the PLS readouts, values |
| `verdicts12.json` | Everything as data |
| `disagreement12.json` | Each sentence's price, rating (mean of W0-W2, 8 people), disagreement, and its held-out reading |
| `states12.npz` | The model's states for all 96 sentences: layers 0-40 by 4, menu and rating context, mean over the sentence (float16) |
| `predictions_vs_results.md` | Both prediction sets scored |
| `log_pod.txt` | The pod's log |

Figure: [`fig11_recipes.png`](../../../figures/fig11_recipes.png).

## Pre-registered questions

- **Q1 passes: where the rating disagrees with the price can be read from the sentence.**
  - The disagreement is the rating minus a cubic in the price.
  - A ridge readout of the sentence's state as option A, before the model has seen any rating question, predicts it on values it never saw: r = 0.35 at layer 28. A permutation test gives p = 0.010; 95% of runs with a shuffled target stay below 0.19.
  - In the rating prompt it is a little higher (0.41 at layer 24).
- **Q2 passes: pushing along the disagreement moves the rating beyond what its effect on the choice would carry.**
  - All pushes are at layer 20, by the same distance.
  - The price direction moves the choice +0.77 logits and the rating +0.11 points: 0.145 rating points per logit of choice.
  - The disagreement direction moves the choice only +0.10 logits but the rating +0.055 points, about 3.7 times what that ratio would give. The excess is +0.0064 rating points per unit of push, 95% interval [+0.0049, +0.0080].
- **Verdict: different recipes.**

## What else the run shows

- **The price push replicates pilot 11.** +0.77 logits for 2 price units is a transfer of 0.39 (pilot 11: 0.38).
- **The rating direction sits close to the price direction**, slightly above the line (excess +0.0035). This fits pilot 11's cosine of 0.87: both mostly carry "how big the outcome is".
- **Which values the disagreement readout gets right (held out).**
  - It puts Defending (+0.31; actual +0.72), Respect (+0.19; +0.58) and Justice (+0.12; +0.21) above their price, and Freedom (−0.08; −0.42) below.
  - It misses some: Doing, Truth, Beauty and Experiencing get the wrong sign.
- **A few directions take up more of the overlap, but not all of it.** With partial least squares, the partial correlation of price and rating given both held-out readouts falls from 0.85 (one component) to 0.63-0.65 (two to five) and then levels off. With 96 sentences the readouts are too noisy to go further, as the design warned.

## Cautions

- **The random direction is also off the line,** below it: the choice +0.13 logits, the rating −0.01 points, excess −0.0042 [−0.0053, −0.0032]. So a push along an arbitrary direction does not follow the shared-vibe line exactly either, and the narrow interval for Q2 covers only the sampling of sentences, not of directions. The disagreement direction is off the line in the predicted direction and by more (+0.0064 against −0.0042). Pushing a set of random directions would give a proper benchmark.
- **The disagreement is readable, but modestly** (r 0.35). Much of where rating and price part ways is not predicted.
- One model, 96 sentences, one push layer, one kind of push direction (difference of means).

## Reading

Price and stated rating overlap strongly because both are built mostly from the same ingredient: how big the outcome sounds. They are not one variable read twice. Where they part ways is partly systematic and already written into the sentence's representation before any rating question is asked: moral and protective acts (Defending, Respect, Justice) count for more in the stated rating than in the choice. Pushing the model along that difference moves its stated rating far more than its choice. Shared ingredients, different recipes.
