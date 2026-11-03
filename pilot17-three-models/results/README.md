# Pilot 17 results (3 Nov 2026)

Llama 3.1 8B Instruct, Qwen3-14B and Qwen3-32B, run through the same tests as pilots 15-16 (prices) and 5-6 (comparator). Design and bars: [`DESIGN.md`](../DESIGN.md), including the addendum written between the two runs.

## The runs

| Run | Models | GPU and cost | Folder |
|---|---|---|---|
| First | Llama 8B, Qwen3-14B. Qwen3-32B failed: the disk filled up | RTX PRO 6000, 13:19-13:26 UTC, about $0.30 | `real/` |
| Rerun, with the decision writers added | Qwen3-32B, Qwen3-14B. Llama failed on disk again | H100 (the 96 GB card was sold out), 13:31-13:43 UTC, about $0.80 | `real2/` |
| Llama alone, decision writers | Llama 8B | pending: no GPU was free at 13:50 UTC; a retry is scheduled | `real2_llama/` |

**Why the disk filled twice.** The pods use a network volume, and deleting a model's files does not free its quota straight away. So three models (111 GB) do not fit in 100 GB one after the other, even with the cleanup.

Figure: [`fig16_three_models.png`](../../figures/fig16_three_models.png).

## The questions

| | Llama 3.1 8B | Qwen3-14B | Qwen3-32B |
|---|---|---|---|
| **Q1 Does it give each outcome a fixed price?** Common share, words / outcomes (bar 0.9) | **no**: 0.78 / 0.70 | **no**: 0.75 / 0.77 | **no**: 0.71 / 0.82 |
| **Method check:** pilot 5's 20 comparator pieces among the 20 decision writers (bar 5) | – | **14** | – |
| **Q2b Do the decision writers carry the choice?** Held-out value swaps (bar 0.4) | pending | **yes**: 0.71 | **yes**: 0.46 |
| **Q3b Are they the generic "which is bigger" machine?** Apples / weight (bar 0.3) | pending | **yes**: 0.81 / 0.76 | **yes**: 0.47 / 0.38 |
| Random sets of 20 pieces, same blocks | – | ≤ 0.01 | ≤ 0.01 |
| *Q2, Q3: the pieces that write the letter (see below)* | *0.82; 0.86 / 0.88* | *0.96; 0.94 / 0.96* | *0.75; 0.83 / 0.81* |

## What the runs show

**No model gives outcomes a fixed price.**
- In all three, 70-82% of an outcome's price is the same whatever it is compared with, below the 0.9 bar.
- **Order flips** (clearly separated pairs of outcomes ordered differently by different alternatives):
  - Llama: 22%;
  - Qwen3-14B: 34%;
  - Qwen3-32B: 18%.

  Counting only menus that are not near-certain, the figures are 17%, 38% and 15%.
- **Choice cycles** that no fixed values allow:
  - Llama: none;
  - Qwen3-14B: 96;
  - Qwen3-32B: 12.
- **The bigger Qwen** is less often near-certain (11% of menus against 55%), and has fewer order flips and cycles. But its common share is no higher. Scale does not give it a fixed price.

**In every model, a rough price exists before the alternative is read, and is finished after it.**
- At the line break after A, a readout trained on the intact outcomes follows the word deletions at r 0.45 (Llama), 0.53 (Qwen3-14B) and 0.41 (Qwen3-32B).
- At the line break after B it reaches 0.81, 0.85 and 0.83.
- Llama finishes earlier: at 40% of its depth, against 60% for Qwen3-14B and 70% for Qwen3-32B.

**In every model, the decision reaches the answer position at the same depth for value menus as for "which pile has more apples".**

| | Value menus | Apples | Share of depth |
|---|---|---|---|
| Llama | 0.47 at layer 16, 0.90 at 19 | 0.58, 0.90 | 50-60% |
| Qwen3-14B | 0.20 at layer 26, 0.73 at 28 | 0.04, 0.79 | 65-70% |
| Qwen3-32B | 0.03 at layer 48, 0.51 at 51 | 0.02, 0.57 | 75-80% |

The deeper the model, the later in it the decision forms.

**The comparator is the generic one, in both Qwens.**
- The new method checks out on Qwen3-14B. The 20 pieces that write the decision's direction at the answer position, in the blocks before it arrives, include 14 of pilot 5's 20.
- On Qwen3-14B they carry 0.71 of held-out value swaps, 0.81 of apples swaps and 0.76 of weight swaps. Pilot 5's own 20 carry 0.57, 0.55 and 0.39 here.
- On Qwen3-32B the decision writers sit in blocks 47-50 (counted from 0). They carry less, but the same for all three tasks: 0.46, 0.47 and 0.38.
- Random sets carry at most 0.01.

**The pieces that write the letter (Q2 and Q3 of the first design) carry almost everything, in every model and every task.**
- The first run's method picked the pieces with the largest direct effect on logit(A) − logit(B). These are the last blocks, which write the letter.
- They carry 0.75-0.96 of value swaps and 0.81-0.96 of apples and weight swaps.
- That they are shared says little, since every A-or-B question ends there. The method check (1 of pilot 5's pieces) showed they are not the comparator.

## Cautions

- **The decision writers for Llama are still to come.** For Llama there is so far only the shared timing, and the letter writers.
- One prompt format, with A always first. The alternatives are from pilot 13.
- **On Qwen3-32B, Q3b's weight result (0.38) is close to its bar (0.3).** The 20 pieces carry less there, so its comparator may be spread over more pieces.

## Reading

The picture from Qwen3-14B holds in a bigger Qwen and, as far as it has been tested, in Llama.
- None of them gives outcomes a fixed price. A rough price forms while reading each option, and the finished price depends on the alternative.
- The choice is then made by the same machinery that decides which pile has more apples. It reaches the answer at the same depth, and in both Qwens it runs through the same pieces.
