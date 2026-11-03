# Pilot 18: Utility Engineering on Llama 3.1 8B, and is the utility used?

## 1. The question

Jasper wanted a published study to anchor to, run slightly altered. The study is **Utility Engineering** (Mazeika et al.
2025, [arXiv 2502.08640](https://arxiv.org/abs/2502.08640); code: [centerforaisafety/emergent-values](https://github.com/centerforaisafety/emergent-values),
commit `5e5966d`). It asks a model to choose between pairs of 510 outcomes, fits a utility to each outcome
(a Thurstonian model), and reports two things:
- how consistent the choices are: how well the utilities predict held-out choices, and how often three outcomes form a
  cycle;
- whether the utilities can be read from the model's hidden states with a linear probe.

Jasper's three steps:
1. Reproduce their consistency numbers for Llama 3.1 8B with their code and outcomes.
2. Reproduce their readout of the utilities.
3. **New:** push the model's internal state along their readout direction and see whether the choice moves. Then check
   whether the choice runs through the parts that compare piles of apples (pilot 17).

## 2. Material and measures

- **Model:** Llama-3.1-8B-Instruct at the revision in their `models.yaml` (`0e9e39f`), bf16. Their model key is
  `llama-31-8b-instruct`.
- **Outcomes and prompt:** their 510 outcomes in 30 categories (`shared_options/options_hierarchical.json`). Their
  forced-choice prompt ("Which implied state of the world would you prefer? … Please respond with only "A" or "B"."),
  with the system message "You are a helpful assistant.". Every pair is asked in both orders.

### Step 1: their code, unmodified

The only change is the model's path in `models.yaml`. Their code runs in its own environment with their pinned versions
(vLLM 0.6.6.post1, torch 2.5.1, transformers 4.46.2).

- **(a) Utilities, their default setting** (`thurstonian_active_learning`):
  - 10 samples per prompt at temperature 1, both orders;
  - active learning up to 9,174 pairs (2·n·log2 n);
  - 1,000 held-out pairs, drawn with their seed 42, so they are the same pairs as theirs.
- **(b) The same in the repo's logprobs mode:** one forward pass per prompt, P(A) from the A and B tokens' probabilities.
  This serves the pipeline check below, and gives a second estimate.
- **(c) Transitivity:**
  - their script with their settings: 1,000 triads, seed 42, 10 samples, both orders;
  - the cycle rate with their notebook's function: a triad counts if its three majority preferences form a cycle, with
    exact 50:50 ties counted as "indifference cycles".
- **(d) Their released results** for `llama-31-8b-instruct`: the utilities and the held-out pairs, from Hugging Face
  `mmazeika/emergent-values-data`, `options_hierarchical.zip`.

**The targets** (exact where their files have them):
- **Held-out accuracy:** their results file, scored with their notebook's function. Their Figure 4 shows 90.4%, read
  from the plot embedded in their notebook: Llama 3.1 8B Instruct is the point at MMLU 68.1.
- **Cycle rate:** their Figure 7 shows log10 = −1.50, about 3.2% of triads, read the same way. Their triad files are not
  in the repo; if the Hugging Face repo has them, the exact value is used.
- **Utilities:** their released means.

### Step 2: the readout

Their probe code is not released. The paper says linear probes on the hidden states predict each outcome's Thurstonian
mean and variance, are "trained using preference data from the LLM", take "individual outcomes as inputs", and are
scored by test accuracy at the best layer (Figure 8; Figure 23 by layer). Where the paper is silent, the choices here are
ours.
- **Input:** each outcome on its own: the BOS token and the outcome's text, with no chat template. The state after every
  layer (0-32) at the last token. The mean over the outcome's tokens is a variant.
- **Probe:** a linear probit on standardised states, P(A over B) = Φ(w·(z_A − z_B)/√2).
  - It is trained on the preference data: each pair's observed probability is a soft label, with an L2 penalty chosen on
    an inner split.
  - The variance part is left out: their accuracy measure depends only on the means.
- **Test:** 5 folds by outcome, stratified by category. The probe is trained on the pairs within the training outcomes and
  tested on the pairs within the held-out outcomes. Accuracy is their measure: the predicted majority against the observed
  one. The best layer is the one with the highest test accuracy, as in their Figure 8.
- **Check:** a ridge regression of the fitted means on the states (penalty by leave-one-out), on the same folds.
- **Target:** their Figure 8 bar for Llama-3.1-8B. The pod renders the figure from the paper, and the value is written
  into this file before our probe results are read.

### Step 3 (new): is the readout used?

**(a) Pushing along the readout.**
- **Direction:** the probe's weight vector at layer ℓs, trained on all 510 outcomes and mapped back to the model's own
  coordinates.
  - ℓs is the layer from 8 to 16 with the best probe test accuracy.
  - Why this window: the decision reaches the answer position at layer 19 (pilot 17), so the value has to be read before
    that. In pilot 11, pushes at a quarter of the depth or less barely worked.
- **Menus:** 150 pairs of outcomes whose fitted means are within 1 SD of each other, in both orders: 300 prompts.
- **Push:** add c·s·ŵ to the state after layer ℓs at every token of option A, or of option B.
  - c is −4, −2, +2 or +4.
  - s is the SD, across the menus' options, of the readout (averaged over the option's tokens).
- **Effect:** the change in logit P(A), per unit of c. It is signed so that pushing an option towards a higher utility
  should raise its own choice: positive for A, negative for B.
- **Benchmark γ:** in the clean menus, how much the logit differs per SD of difference in the readout between A and B (an
  ordinary least-squares slope). If the choice were driven by this direction alone, a push of 1 SD would move it by γ.
- **Control:** 20 random directions, pushed by the same amount (the same norm).
- **Also:** the probe's own direction at layers 4, 8, 12, 16, 20 and 24 (c = ±4), to see where pushing works.

**(b) The apple comparator.**
- **Swaps:**
  - 150 menus of the form "X vs Z" each get a donor menu "Y vs Z". Y is at least 1 SD from X and from another category.
  - A pair is kept if the donor moves the logit by at least 2.
- **Patching:**
  - Pilot 17's 20 decision writers for Llama (heads and MLPs in blocks 14-18, found on our own menus) are patched at the
    answer position.
  - The controls are 3 random sets of 20 parts from the same blocks.
  - Transfer is as in pilot 15: the slope through zero of the change on the donor's effect.
- **Also:** the whole state at the answer position, swapped after layers 8, 12, 16, 19, 22 and 26.
- **Reference:** on our menus the same 20 parts carry 0.86 of the choice, 0.90 of "which pile has more apples" and 0.95 of
  "which is heavier".

## 3. Pre-registered bars (fixed now)

| | Question | Measure | Pass |
|---|---|---|---|
| **B1a** | Same held-out accuracy? | Our (a) against their results file | within 3 points |
| **B1b** | Same utilities? | Pearson r, our (a) means against theirs | ≥ 0.90 |
| **B1c** | Same consistency? | log10 cycle rate, our (c) against their Figure 7 (−1.50) | within 0.30 (a factor of 2) |
| **B2** | Same readout? | Best-layer probe test accuracy against their Figure 8 (Llama-3.1-8B) | within 5 points |
| **B3a** | Is the readout used for the choice? | The push's effect per SD | above all 20 random directions, and ≥ 0.25 γ |
| **B3b** | Does the choice run through the apple comparator? | Pilot 17's decision writers' transfer on these menus | ≥ 0.4, every random set ≤ 0.1 |

**Pipeline check (ours, not about the paper).**
- Measure: our forward pass's P(A) against vLLM's, per prompt, on the 2,000 held-out prompts of run (b).
- Pass: r ≥ 0.99 between the logits.
- Two tokenisations are tried: with one BOS token and with two. vLLM 0.6.6 adds the tokenizer's BOS to a chat text that
  already starts with one. The closer one is used for steps 2 and 3.

**How B3a is read.**
- Above all random directions:
  - ≥ 0.5 γ: **mostly used.** This was pilot 11's bar for the same kind of test on Qwen3-14B, which reached 0.38.
  - 0.25-0.5 γ: **partly used.**
  - below 0.25 γ: **weakly used.**
- Not above them: **not used**, at least not at this layer and these tokens.
- If γ ≤ 0, the readout does not track the choice inside menus, and B3a cannot be read.

## 4. Also reported (not pre-registered)

- **The cycle rate with exact probabilities.** Our forward pass on the same 1,000 triads, order-averaged. This shows how
  much of the sampled cycle rate is sampling noise.
- **Order flips.** The share of pairs in run (b) where the two orders pick different outcomes.
- **Pair by pair.** Our (a) and (b) against their probabilities on the 1,000 shared held-out pairs.
- **Probe variants:**
  - the mean over the outcome's tokens;
  - the ridge check;
  - the probe trained on all outcomes and tested on the 1,000 held-out pairs (their Figure 4's pairs).
- **Whether the readout reads the utility inside the menus:** r between the in-menu readout and the fitted means.
- The push's effect at each dose, and the layer sweep.

## 5. Running it

- **One H100 80 GB.** vLLM 0.6.6 with torch 2.5.1 does not run on the Blackwell RTX PRO 6000. The hourly price is stated
  before the pod is created.
- **Order on the pod:**
  1. Download the model.
  2. Fetch the paper's figures and their data, and push them to `pod-results/pilot18-paper`. That branch is deleted once
     the figure is read.
  3. A quick test of our code on synthetic utilities, so it fails fast.
  4. Their code: (a), (b), (c).
  5. Our steps 1-3.
  6. Push to `pod-results/pilot18` and stop.
- **Time and cost:** about an hour, about $3-4.
- **Predictions:** no prediction gate. The bars above are fixed before any result.

## 6. Addendum before the rerun (3 Nov 2026, about 19:00 UTC)

**The targets, read before any of our results existed.**
- **Their released results** for `llama-31-8b-instruct`: held-out accuracy exactly **90.4%** (log loss 0.505), as read
  from their Figure 4. Their outcomes are the repo's 510, in the same order.
- **Figure 8, the Llama-3.1-8B bar: 78.0%.** Measured on the page rendered at 200 dpi:
  - the zero line at y = 1135.5, 2.533 px per percentage point, checked against the gridlines at 60 and 80;
  - the other bars: Llama-3.2-1B 44.0%, Llama-3.3-70B 89.0%.
- **Figure 23 agrees:** Llama-3.1 8B's cells are pale yellow, a test error of about 0.22, from early layers on.
- **So B2 passes if our best-layer accuracy is 73-83%.**

**What happened in the first run.**
- **The quick test failed, as designed,** before their code or our real run started.
  - At layer 0 every outcome's last token is ".", so all the states are identical and the ridge's penalty search had
    nothing to work with.
  - The ridge now falls back to the mean in that case.
- **Their released file has 38,925 pairs, not 9,174.** After the 9,174 asked pairs and the 1,000 held out, their code
  adds every unasked pair that the fitted model is at least 95% sure of, as a 0/1 "pseudolabel". This is described after
  their Algorithm 1.
  - Those are the fitted model's own labels, not the model's choices. So **the probe is trained and tested only on asked
    pairs.**
  - The utilities are still fitted their way, with pseudolabels.
- **The pod's container took 13 minutes to start,** because it was creating a 100 GB network volume. The rerun uses a
  large container disk instead.
