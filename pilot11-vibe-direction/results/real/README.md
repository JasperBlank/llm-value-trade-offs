# Pilot 11 results (Qwen3-14B, run 3 Nov 2026)

**The run.**

- It used one A40 pod on Runpod, from 07:47 to about 07:55 UTC; the real run took 6 minutes and the cost was about $0.08.
- Both prediction sets were committed before the run, and the hashes were checked on the pod.
- The GitHub token secret was not set, so nothing was pushed. `report.md` and `predictions_vs_results.md` are the pod's own, taken from its log (`log_pod.txt`).
- `readings11.json` (each sentence's held-out reading in both prompts) was rebuilt from a line in the log and matches the report.
- `verdicts11.json` holds only the key numbers. The read states, the full JSON and the directions were not pushed.

| File | What |
|---|---|
| `report.md` | The pod's report: the three questions, within stakes level, every layer, every push |
| `readings11.json` | Each sentence's held-out reading at layer 20: read as option A, and the same direction read in the rating prompt |
| `items11.json` | The sentences with their price, rating and the 4 menu partners used for pushing |
| `verdicts11.json` | The key numbers |
| `predictions_vs_results.md` | Both prediction sets scored |

Figure: [`fig10_vibe_direction.png`](../../../figures/fig10_vibe_direction.png).

## Pre-registered questions

- **Q1 passes.** A sentence's menu price can be read from its own words as option A, before the model has seen B.
  - One direction (difference of means between the highest- and lowest-priced thirds), learned on 23 values and read on the 24th, gives held-out r = 0.78 at layer 20.
  - A ridge readout reaches 0.84 (layers 20-24).
- **Q2 fails.** Read in the rating prompt, the same direction predicts the model's rating r = 0.64. But price and rating still correlate 0.65 once it is accounted for (they correlate 0.81 without it; the bar was 0.3). So this direction does not carry the overlap.
- **Q3 fails.** Pushing along the direction at layer 20 by ±2 price units moves the choice by 0.38 of the predicted amount and the rating by 0.17. A random direction moves neither (−0.02 and −0.02); the bar was 0.5 for both.
- **Verdict: readable but not shared.**

## What else the run shows

- **The price is written into the sentence in both prompts.** The same kind of direction reads the menu price about equally well from the rating prompt (held-out r 0.77). The direction learned in the menu prompt, read in the rating prompt, tracks the menu price (0.79) better than the model's own rating (0.64).
- **It is mostly a "big or small outcome" axis.** Within each stakes level (small, medium, large), the direction reads the price only r 0.37, and in the rating prompt it says nothing about the rating (0.04).
  - The price direction and a direction learned from the ratings point almost the same way (cosine 0.87).
  - A readout of where rating and price disagree is weakly possible (held-out r 0.34 at layer 20).
- **Where pushing works.**
  - The choice responds at layers 12-20 (0.27, 0.41, 0.38), barely at 4-8 (0.10).
  - At 28 and later there is no effect at all: by then the sentence's information has moved on to later positions, as pilots 7 and 8 found.
  - The rating responds at most 0.17 (layer 20); the log-scale score shows the same, 0.15 of its slope across sentences.

## Cautions

- 96 sentences is few for a readout of a 5,120-dimensional state (Gilg et al. used 5,000 tasks). The direction mostly picks up the strongest axis, the size of the outcome.
- A push at one layer only reaches what later layers read. Information already copied to later positions before that layer is not changed, which caps the transfer from any single layer.
- One kind of direction (difference of means). A ridge or learned (DAS) direction might carry more.
- The menu push used each sentence's 4 nearest-priced partners at 8 vs 8 people. Many base choices were still far from 50:50 (median |logit P(A)| 4.5).

## Reading

The model writes something like a price onto each outcome as it reads it, before it has seen the alternative, and it does so whatever it was asked: it is there in the rating prompt too. The choice partly uses it: pushing along that one direction at layers 16-20 moves the choice by about 40% of the predicted amount, and a random direction does nothing. The stated rating hardly uses it (about 15%). So the overlap between what the model chooses and what it says is not carried by this single direction. The choice and the stated rating both draw on how big the outcome sounds, but they do not read it out the same way.
