# Pilot 10: the full range of goodness

## 1. The question

In pilot 9, asked "How good would it be if [sentence]?", Qwen3-14B gave about half of all answers as 7. Pilot 8's 48 sentences are all moderately good outcomes, and the expected rating could barely move. The rating matched the menu prices only r = 0.55, although the same wording of a value won in 20 of 24 values.

Jasper's point: give the model outcomes that span the whole range, from trivial to life-changing, and see whether the rating and the menu then agree.

Each rating prompt shows one sentence on its own, so the new sentences do not change how pilot 8's sentences are rated. The fine question, whether rating and menu agree among outcomes of similar size, is answered by a finer readout: the log-scale score of pilot 9's follow-up. This run covers both.

## 2. Material

- **Model:** Qwen3-14B (bf16, thinking off).
- **Sentences:** 96 = 24 values × 4 (`build_prompts10.py`, `sentences10.json`):
  - a **small** version (e.g. "are each spared a mild headache"),
  - **pilot 8's two sentences** (medium),
  - a **large** version (e.g. "are each cured of a severe pain they have suffered from for years").
- **Menu:** every ordered pair of different sentences at 2:32, 4:16, 8:8, 16:4 and 32:2 people. That is 9,120 × 5 = 45,600 trade-offs, with pilot 8's context line, the question "Which of these two outcomes would you rather happen?" and the answer letters A or B.
- **Ratings:** each sentence at 2, 4, 8 and 32 people, asked pilot 9's four ways, giving 1,536 prompts. The 576 prompts at 2, 4 and 8 people for pilot 8's sentences are identical to pilot 9's.

**Readouts:**
- **Menu:** P(A), the letters' probabilities renormalised over A and B.
- **Prices:** P(A) = σ(u_A − u_B + b·log(n_A/n_B) + c), fitted by maximum likelihood with a small ridge (0.01 Σu², so that a sentence that wins every comparison keeps a finite price).
- **Ratings:** the digits' log-probabilities (digit logits in float32 from the final hidden state), giving:
  - the **expected rating** Σ d·P(d);
  - the **log-scale score**, the mean over the cut-points k = 1..9 of log P(answer ≥ k) − log P(answer < k).

## 3. Pre-registered tests

| Test | Measure | Pass |
|---|---|---|
| **F1 The full range** | r, across the 96 sentences, between the W0 ("How good would it be …") expected rating at 8 people and the menu price | r ≥ 0.8 |
| **F2 Hidden in the 7s** | r, across pilot 9's 48 sentences, between the W0 log-scale score at 4 people and pilot 8's per-sentence prices (pilot 9's follow-up question) | r ≥ 0.7 |
| **F3 Size** | mean over the 96 sentences of the W0 expected rating at 32 people minus at 2 people | "size-sensitive" if ≥ 1 point |

F1 is expected to rise partly for an uninteresting reason: anyone ranks "saved from a fire" above "a polite nod". F2 and the within-level measures below carry the finer question.

## 4. Also reported (not pre-registered verdicts)

- The menu's fit, b, the bias towards A, and the split-half reliability of the 96 prices.
- **The designed order** (small < both medium < large) per value, in the menu and in the rating.
- **Stakes against domain:** the share of the spread in prices (and in ratings) explained by the stakes level, by the value, and by both.
- **Context:** pilot 8's 48 sentences priced in this new menu against their pilot 8 prices, and whether the same sentence of each pair wins.
- **For each readout** (expected rating, log-scale score, the "lean around 7": log P(≥ 8) − log P(≤ 6)) **and each wording:**
  - r over all 96;
  - r within stakes level;
  - r over pilot 8's 48 sentences against the new menu and against pilot 8's;
  - same-sentence-wins over the 24 pairs.
- **Size:** the rating's gain from 2 to 32 people by level; the slope per log(people) in the menu's units against b.

## 5. Running it

- One A40 pod. The 45,600 trade-offs take about 20-30 minutes; about $0.30.
- With the `github_push_token` secret, the results are pushed to `pod-results/pilot10` and the pod stops at once.
- Without it, the pod prints the report and the raw results in compact lines (`print_raw10.py`, rebuilt by `rebuild_from_log10.py`), then waits to be terminated.
- Predictions: Claude's sealed first, then Jasper's (`predictions/`).
