# Pilot 8: why are the relative values what they are? Worth by domain and size

Written 3 Nov 2026, before any pilot-8 result. Section 4 is pre-registered; the thresholds are also
in `config.yaml` under `prereg`.

## 0. Revision (3 Nov, before any result)

The first version crossed five ingredients (domain, gain vs prevented loss, size, children vs
adults, poverty). Jasper narrowed it before the run: only **domain and size** matter for his
question. This version traces worth as a curve over size within each domain. The first version's
sealed predictions and Jasper's answers to it are kept in `predictions/` (superseded, never run).

## 1. Question

Why are the relative values what they are? Here: how does an outcome's worth depend on *what* it is
(money, free time, health) and *how much* of it there is? Concretely: (1) does worth follow a simple
curve per domain, worth = a_domain + s_domain · log(amount)? (2) Is the sensitivity to size the same
across domains? (3) What is the model's price list (an hour of free time, a day of good health, in
euros)? (4) Inside the model, is the worth already assembled on the outcome's own words?

## 2. Outcomes and prompts (`make_prompts.py`)

24 outcomes: 3 domains × 8 sizes, all "{n} people each gain …":

| domain | sizes |
|---|---|
| money | €2, €5, €20, €50, €200, €1,000, €5,000, €20,000 |
| free time | 10 minutes, an hour, an afternoon, a day, a week, a month, half a year, a year (of free time) |
| health | the same eight durations, "of good health" |

Amounts in a common unit per domain (euros; hours, with an afternoon = 4 hours and a day = 24).

**Behaviour.** Every unordered pair (276), both orders, three numbers of people (A 2 / B 8, 4 / 4,
8 / 2) and four question wordings (pilot 3's wordings 0-3, strangers frame): 6,624 prompts. The
answer is P(A) from the letter probabilities.

**Fit.** u per outcome (the first fixed at 0), b (the weight of log people) and a position bias by
maximum likelihood: P(A) = σ(u_A − u_B + b·log(n_A / n_B) + c). Then the curves u = a_d + s_d·log(amount)
by least squares; bootstrap over prompts for intervals (200).

## 3. Inside the model (Qwen3-14B)

- **Maps** (descriptive): minimal pairs in option A (size one step up within a domain; domain changed
  at the same size rank), swapped at one site and layer (A's last token, the line break after A, the
  line break after B, the text after B, the answer position; layers 4, 8, ..., 40).
- **The worth direction:** at the A site chosen by the rule, one direction trained on swaps of option
  A's outcome among 16 training outcomes; the 8 held-out outcomes are interpolations (never the
  smallest or largest size). Each outcome's position along it is read off its own state (option A,
  before option B is read).

## 4. Pre-registered criteria

**Site rule:** among A's last token and the line break after A, and layers 4-24, the site and layer
where swapping the whole site carries the most of an outcome swap (ties: the earlier layer).

- **W1, worth = domain + log(size):** the per-domain curves, fitted on 23 outcomes, predict the 24th;
  over all 24, held-out R² ≥ **0.8**: "worth = domain + log(size)"; 0.5-0.8: "partly"; below 0.5:
  "not explained this way".
- **W3, one sensitivity to size:** all three slopes positive and the largest at most **1.5 ×** the
  smallest: "one sensitivity to size across domains"; otherwise "different sensitivities".
- **W2, assembled on the outcome's own words:** along the worth direction, the 8 held-out outcomes'
  positions correlate at least **0.7** with their worth, AND the position grows with log(amount) in
  every domain (over all 24 outcomes).

Reported, not pre-registered: the price list with 95% intervals, the common-slope model, how many
people doubling an amount is worth (2^(s_d / b)), the maps.

## 5. Limitations known in advance

One model, one answer format, one template ("people each gain"). Units are nominal (a day = 24
hours of free time is not 24 waking hours); the slopes per ×e of the amount do not depend on the unit,
but the price list does through these conventions. Very small amounts ("10 minutes of good health",
"€2") may read oddly. Qwen splits numbers into digits, so for money A's last token is a digit.

## 6. Budget

One A100 80 GB pod ($1.79/hour), about 20-25 minutes. The pod prints its results and stops itself
(tested: the log stays readable after the stop); Claude then terminates it.
