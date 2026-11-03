# Figures for the post

Drawn by `make_post_figures.py` straight from the committed result files (`python figures/post/make_post_figures.py`,
or name one figure to redraw only that one). Each PNG is rendered at 2x from an HTML page in `html/`, which holds the
same text: to change a title or a caption, edit the script and rerun. Colours: the dataviz reference palette (blue for
option A, orange for option B; a blue ramp for small / medium / large; blue against red around a grey midpoint for the
word effects). Fonts: Inter and Noto Color Emoji.

| File | Size | Shows | Data |
|---|---|---|---|
| `hero_x.png` | 1600 × 900 (3200 × 1800) | **The main figure, for X and the top of the post.** "A vibe check, then 'which is bigger?'": one real menu, twice. With "a **brief** sense of awe while looking at the stars", Qwen3-14B picks the dream job (85%); delete "brief" and it picks the stars (99.97%). Step 1, the vibe check; step 2, the same "which is bigger?" check as for piles of apples | pilots 16, 17 |
| `f1_same_value.png` | 1200 wide | Same value, wildly different prices: six values' four outcomes on the price scale, with the cheapest and dearest of each, above all 96 | pilot 10 |
| `f2_vibe_check.png` | 1200 wide | Step 1: the price tracks a vibe check. Qwen's "how good would it be?" rating against the price (96 + 24 new outcomes), and how well each predictor does | pilot 13 |
| `f3_one_word.png` | 1200 wide | One word can flip the choice: five outcomes word by word, coloured by what each word adds to the log-odds of choosing it | pilot 14 |
| `f4_cycle.png` | 1200 wide | Qwen's preferences can go in circles: a four-way cycle in real menus, and per model the number of such cycles (of the sets that could cycle) and the share of pairs that swap order | pilots 16, 17 |
| `f5_apples.png` | 1200 wide | Step 2: outcomes are compared by the same machinery as piles of apples. When the decision reaches the answer, and what the 20 parts that write it carry, in all three models | pilot 17 |
| `f6_price_timing.png` | 1200 wide | A rough price while reading, finished after the alternative: the price readout at the end of option A and of option B, in all three models | pilot 17 |

**The hero's example.** The menu is pilot 16's (Qwen3-14B, 8 people each): "each feel a brief sense of awe while
looking at the stars" against "each turn the activity they love into their full-time job". P(A) is 0.148 with "brief"
and 0.9997 without it. Pilot 17's rerun of Qwen3-14B gives 0.22 and 0.9996. Qwen3-32B flips too (0.15 → 0.95), while
Llama 3.1 8B picks the stars either way (0.9996). The bars are a sketch: they show which way the comparison goes, not
measured prices.

**How the hero was designed.** It follows common advice for graphical abstracts and teaser figures:
- One message, stated in the title.
- A clear entry point, read left to right.
- Plain words, with no jargon.
- One concrete, relatable example.
- A minimal pair (the same menu with one word deleted), so that what decides the choice is visible.
- Colour only where it carries meaning: A, B, and the deleted word.
- Large type that survives the X timeline (about 600 px wide).

Sources:
- [SIGGRAPH: what are figures for](https://research.siggraph.org/blog/guides/what-are-figures-for-and-how-do-i-compose-them)
- [Ten simple rules for designing graphical abstracts](https://pmc.ncbi.nlm.nih.gov/articles/PMC10833524/)
- [MIT CommLab: figure design](https://mitcommlab.mit.edu/eecs/commkit/figure-design/)
- [Qiao and Hullman 2018, translating scientific graphics for public audiences](https://c4pgv.dbvis.de/Qiao_Hullman_2018.pdf)
- [FlowingData: visualizing differences](https://flowingdata.com/2018/04/17/visualizing-differences/)
- [Springer Nature: effective graphical abstracts](https://authorservices.springernature.com/featured-articles/how-to-create-an-effective-graphical-abstract-for-your-research-paper/)

**Definitions that differ slightly from the pilot READMEs.**
- `f6`: for both positions, the layer is chosen on one half of the values and r is reported on the other half, from
  the rerun. Pilot 17's README gives 0.81 for Llama at the end of option B, from the first run; the rerun gives 0.80.
- `f2`: the predictors' r on the 96 are held out by value, except Qwen's "how good" rating (a plain correlation), as in
  pilot 13's README. The scatter's r are plain correlations.
- `f4`: the probabilities are P(answer) from the logits at 8 people each. The cycle and order-flip counts are pilot
  17's (96 outcomes × 9 alternatives); the example menus are pilot 16's. A set that could cycle is two outcomes and two
  alternatives where each alternative clearly beats one outcome and loses to the other.
