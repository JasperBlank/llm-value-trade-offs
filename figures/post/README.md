# Figures for the post

Drawn by `make_post_figures.py` straight from the committed result files (`python figures/post/make_post_figures.py`,
or name one figure to redraw only that one). Each PNG is rendered at 2x from an HTML page in `html/`, which holds the
same text: to change a title or a caption, edit the script and rerun. Colours: the dataviz reference palette (blue,
orange, aqua; a blue ramp for small / medium / large; blue against red around a grey midpoint for the word effects).
Fonts: Inter and Noto Color Emoji.

| File | Size | Shows | Data |
|---|---|---|---|
| `hero_x.png` | 1600 × 900 (3200 × 1800) | **The main figure, for X and the top of the post.** The route from the words to the answer: read the options, a vibe check of how big a deal each is, then the generic comparator that also handles apples and weight. Three findings underneath | pilots 10, 13, 14, 16, 17 |
| `f1_same_value.png` | 1200 wide | Same value, wildly different prices: six values' four outcomes on the price scale, with the cheapest and dearest of each, above all 96 | pilot 10 |
| `f2_vibe_check.png` | 1200 wide | It's a vibe check: Qwen's answer to "How good would it be?" against the price (96 + 24 new outcomes), and how well each predictor does | pilot 13 |
| `f3_one_word.png` | 1200 wide | One word can flip the choice: five outcomes word by word, coloured by what each word adds to the log-odds of choosing it | pilot 14 |
| `f4_cycle.png` | 1200 wide | Qwen's preferences go in circles: a four-way cycle in real menus, and per model the number of such cycles and the share of pairs that swap order | pilots 16, 17 |
| `f5_apples.png` | 1200 wide | Outcomes are compared like piles of apples: when the decision reaches the answer, and what the 20 decision writers carry, in all three models | pilot 17 |
| `f6_price_timing.png` | 1200 wide | A rough price while reading, finished after the alternative: the price readout at the end of option A and of option B, in all three models | pilot 17 |

**Definitions that differ slightly from the pilot READMEs.**
- `f6`: for both positions, the layer is chosen on one half of the values and r is reported on the other half, from
  the rerun. Pilot 17's README gives 0.81 for Llama at the end of option B, from the first run; the rerun gives 0.80.
- `f2`: the predictors' r on the 96 are held out by value, except Qwen's "how good" answer (a plain correlation), as in
  pilot 13's README. The scatter's r are plain correlations.
- `f4`: the probabilities are P(answer) from the logits at 8 people each; the cycle and order-flip counts are pilot 17's
  (96 outcomes × 9 alternatives), the example menus are pilot 16's.
