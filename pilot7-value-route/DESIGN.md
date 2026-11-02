# Pilot 7: where does the model form the values it puts through the decision stage?

Written 2 Nov 2026, before any pilot-7 result. Everything in section 4 is pre-registered; the
thresholds are also in `config.yaml` under `prereg`.

## 1. Why this pilot

Pilot 6 found that the answer position receives *the outcome of a comparison*, in one direction
that tracks the model's own A-vs-B lean (r 0.99), and that this happens at layer 28 in machinery
shared with non-value comparisons. A's value can be read there (probe r 0.85) but is not what is
used. So the values are formed **upstream**, before layer 28, somewhere in the text after option A.

Earlier hints: a probe on A's own last token reads A's value only weakly (pilot 4), and the heads
that carry A's value to the answer position read mostly from the newlines (18%) and the chat-template
tokens (19%), not from A's words (6%) (pilot 5).

Two stories:

1. **One currency (monkey-like).** Before the comparison, option A is turned into one number that
   combines *which* good it is and *how many* people get it, weighted the way the model's choices
   weigh them. This is the analogue of the "offer value" cells found in monkey orbitofrontal cortex
   (Padoa-Schioppa and Assad, 2006).
2. **No separate value.** The good and the number travel separately and are only weighed against each
   other in the comparison itself.

## 2. Models and data

- **Qwen3-14B** (as in pilots 3-6) and **Llama 3.1 8B Instruct** (a second family; gated, so it runs
  only if the account has access; otherwise Qwen alone).
- **Each model's own values.** Its logit difference on pilot 4's prompts is fitted as
  u(A's good) − u(B's good) + b·log(n_A / n_B) + a bias towards A (least squares, a tiny ridge on the
  goods). A model enters the verdicts only if this fit predicts its choices on the held-out wordings with
  r ≥ 0.5 and b > 0 ("usable").
- **Pairs** (`make_prompts.py`, `pairs.jsonl`), a recipient and a donor that differ only in option A's line:

| set | pairs | what changes in option A |
|---|---|---|
| good_train / good_test | 2,880 / 1,440 | the good (same number of people); pilot 4's six wordings / pilot 5's three new ones |
| number_train / number_test | 2,880 / 1,440 | the number of people, by at least a factor 2 (same good, same recipients) |
| bnumber_test | 1,440 | B's number of people (same number of digits, never 1), so the whole prompt keeps its length |
| quantity | 216 | the number of apples (pilot 6's control; the answer flips) |
| weight | 261 | the object (pilot 6's control; the answer flips) |

Everything after option A's line is the same text in a recipient and its donor, so positions there
line up exactly (counted from the end); pairs whose tokens after A do not match are dropped. For B
swaps the whole prompt has the same length and differs only inside B's line.

## 3. What runs (`pilot7.py`), per model

**Sites** (parts of the prompt): `A_end` (the last token of A's line), `A_nl` (the line break right
after A's line), `newlines` (all line breaks after A), `B_line`, `B_end`, `B_nl` (the line break after
B's line), `reply` (the "Reply with only the letter A or B." line), `template` (the chat-template
tokens before the answer position), `after_B` (everything after B's line except the answer position),
`between` (everything after A's line except the answer position), `after_A`, and `last` (the answer
position). `A_end` and `A_nl` come **before** B's line, so they cannot depend on option B at all.

| stage | what | output |
|---|---|---|
| fit | the model's own values, as above | fit.json |
| map | for 128 held-out pairs of each kind (A's good, A's number, B's number, apples, weight): replace one site's states at one layer (every 4th) by the donor's, and measure the transfer of the donor's effect on the answer | map.json |
| das | at the two sites chosen by the rule below: one direction trained on A's GOOD swaps (1,440 pairs, training wordings), and one on A's NUMBER swaps for comparison; both evaluated on held-out A-good, A-number and B-number swaps, apples and weight | das.json, directions.npz |
| report | verdicts | report.md, verdicts7.json |

The swap along a direction r at a site is h' = h + r rᵀ(h_donor − h) at every position of the site
(one direction shared by the site's positions), trained as in pilot 6 (frozen model, Adam, 2 epochs,
the target being the answer when the whole site is swapped). The "coordinate change" of a pair is the
mean over the site's positions of rᵀ(h_donor − h).

## 4. Pre-registered rule and criteria

**Arrival layer:** the first map layer where swapping the answer position alone carries at least 0.5
of an A-good swap (pilot 6, Qwen: 28).

**Which sites are tested (the rule):**
- the **A site**: among `A_end, A_nl` and the map layers up to the arrival layer, the site and layer
  where swapping the whole site carries the most of an A-good swap (ties: the earlier layer). Being
  before B's line, it can only hold something about A: the closest analogue of an "offer value";
- the **B site**: among `B_nl, reply, template, after_B` and the map layers *before* the arrival
  layer, the same: where A's information sits after the model has also read B, before the answer
  position takes it.

**For the direction trained on A-good swaps at each site** (shares are of the site's own whole-swap
effect, on held-out pairs):

- **V2, one currency:** it carries at least **0.5** of the site's effect for A-good swaps AND at least
  **0.5** for A-number swaps, which it never saw in training.
- **V3, the exchange rate:** the coordinate change per unit of the model's own value (number swaps:
  b·Δlog n; good swaps: Δu) is the same for number and good swaps within a factor **1.5**.
- **V4, value or magnitude:** if the site carries at least **0.2** of an apples swap: "value-specific"
  when the direction carries less than **0.3** of the site's apples effect, "generic magnitude"
  otherwise; if the site carries less than 0.2: "not testable".
- **V5, A's value or already the comparison** (B site only): ρ = the coordinate change per unit of
  B's value (B-number swaps) divided by that per unit of A's value (A-good swaps). ρ ≤ **−0.5**: "the
  comparison (A minus B)"; ρ ≥ **0.5**: "a value code shared by both options"; in between: "A's own
  value". Not testable if the site carries less than **0.2** of a B swap.
- **Floor** (A site only): if its whole swap carries less than **0.1** of an A-good swap, A's
  information is not concentrated at A's line end; its V2-V4 are reported but not counted.

**Headline, per usable model and site:** "one currency" if V2 and V3 both pass (with V5's reading at
the B site); otherwise "no single value here".

## 5. How the outcomes will be read

- **One currency at the A site:** the model turns option A into one number, at A's own line end,
  before it has seen B, combining good and number at its own exchange rate: the closest thing to the
  monkeys' offer-value cells, and the place where "it picks the value". V4 says whether that number is
  about value or about "how much" in general.
- **No single value at the A site, one currency at the B site, V5 "the comparison":** the model never
  forms A's value on its own; good and number are weighed together with B, after B has been read.
- **V5 "a value code shared by both options":** one code for "how good is this option" used for A and
  B alike: a common currency in the strongest sense.
- **V2 passes but V3 fails:** one channel carries both, but not at the model's own exchange rate; the
  weighing happens later.
- **The map** says where A's information goes and when (descriptive).

## 6. Limitations known in advance

- Sites are coarse: a whole line or span, or one token. A single shared direction over a span may
  miss a code that differs by position; the whole-site swap is the reference, so the shares stay fair.
- The rule picks the site with the largest good-swap effect, which favours big spans (`after_B`).
- A's own line cannot be swapped as a whole (the donor's line has a different length), so A's own
  words are represented by their last token only.
- The values are each model's own fit on the A/B format; another format may weigh things differently
  (pilot 5's formats extension).
- One answer format, two models, 9 goods.

## 7. Budget

One A100 80 GB pod (Runpod, $1.79/hour), about 80 minutes: downloads, a quick test on Qwen3-0.6B,
Qwen3-14B (about 45 minutes), Llama 3.1 8B (about 25 minutes). The run starts at once and keeps its
results hidden until both prediction sets are committed.
