# Figures for the post

Drawn by `make_post_figures.py` straight from the committed result files (`python figures/post/make_post_figures.py`,
or name one figure to redraw only that one). Each PNG is rendered at 2x from an HTML page in `html/`. The full run also
writes [`CAPTIONS.md`](CAPTIONS.md): the method details and qualifiers kept out of the images, ready to use or adapt as
figure captions.

| File | Size | Shows | Data |
|---|---|---|---|
| `hero_x.png` | 1600 × 900 (3200 × 1800) | **The main figure, for X and the top of the post.** "A vibe check, then 'which is bigger?'". One real menu, twice: with "a **brief** sense of awe while looking at the stars", Qwen3-14B picks the job (85%); delete "brief" and it picks the stars (99.97%) | pilots 16, 17 |
| `f1_same_value.png` | 1200 wide | Same value, wildly different prices: four values' outcomes on the price scale, with the cheapest and dearest of each, above all 96 | pilot 10 |
| `f2_vibe_check.png` | 1200 wide | The price tracks a vibe check: Qwen's own "how good would it be?" rating against the price, and how well three predictors do | pilot 13 |
| `f3_one_word.png` | 1200 wide | One word can flip the choice: four outcomes word by word, shaded by what each word adds to the chance of being chosen | pilot 14 |
| `f4_cycle.png` | 1200 wide | Qwen's preferences can go in circles: a four-way cycle in real menus | pilot 16 (rates: pilot 17) |
| `f5_apples.png` | 1200 wide | Outcomes are compared like piles of apples: when the decision reaches the answer, and what the same 20 internal parts carry, in all three models | pilot 17 |
| `f6_price_timing.png` | 1200 wide | The price is finished only after both options: the price readout after option A and after option B, in all three models | pilot 17 |

## Design rules

These rules are deliberate. They avoid the usual tells of generated ("vibe-coded") graphics:
- emoji used as icons;
- tiny tracked capital labels above headlines;
- cards with soft shadows, and pill badges;
- big-number stat blocks;
- Inter on an off-white page;
- every word kept and shrunk to fit.

The redesign cut the text on the images by about three quarters (1,747 → 424 words).
- **One message per figure.** It goes in a serif title of about eight words, with at most one short line under it. No
  kicker labels and no notes.
- **Type.** Source Serif 4 for titles, Libre Franklin for everything else. Both are in `fonts/`, under the SIL Open Font
  License.
- **Surface.** A white page. Space and hairline rules do the grouping: no cards, shadows, pills or emoji.
- **Colour only where it means something.** Blue for option A or outcomes, orange for option B or apples, aqua for
  weight, red for the deleted word. Text stays in ink colours.
- **Details go in the caption,** not the image (`CAPTIONS.md`).

The hero follows common advice for teaser figures and graphical abstracts:
- One message, stated in the title.
- Read left to right.
- Plain words.
- One concrete example, shown as a minimal pair, so that what decides the choice is visible.

A cold reader with no background summed up the earlier version correctly. Their confusions led to the current wording.

Sources:
- [SIGGRAPH: what are figures for](https://research.siggraph.org/blog/guides/what-are-figures-for-and-how-do-i-compose-them)
- [Ten simple rules for designing graphical abstracts](https://pmc.ncbi.nlm.nih.gov/articles/PMC10833524/)
- [MIT CommLab: figure design](https://mitcommlab.mit.edu/eecs/commkit/figure-design/)
- [Qiao and Hullman 2018, translating scientific graphics for public audiences](https://c4pgv.dbvis.de/Qiao_Hullman_2018.pdf)
- [FlowingData: visualizing differences](https://flowingdata.com/2018/04/17/visualizing-differences/)
- [Fountain Institute: 7 signs a UI has been vibe coded](https://www.thefountaininstitute.com/blog/signs-vibe-coded-ui)
- [The visible tells of AI design](https://924b8765.impeccable-2rv.pages.dev/slop)
- [How to avoid that vibe-coded look](https://robotsonpayroll.substack.com/p/how-to-avoid-that-vibe-coded-look)
- [How to spot AI-made graphic design](https://mikareyes.com/ai/how-to-spot-ai-made-graphic-design-and-posters.md)

## Notes on the data

- **Hero.** The menu is pilot 16's (Qwen3-14B, 8 people each). P(A) is 0.148 with "brief" and 0.9997 without it.
  Pilot 17's rerun of Qwen3-14B gives 0.22 and 0.9996. Qwen3-32B switches too (0.15 → 0.95); Llama 3.1 8B picks the
  stars either way (0.9996). The bars are illustrative.
- **`f6`.** For both positions, the layer is chosen on one half of the values and r is reported on the other half, from
  the rerun. Pilot 17's README gives 0.81 for Llama at the end of option B, from the first run; the rerun gives 0.80.
- **`f2`.** The predictors' r on the 96 are held out by value, except Qwen's "how good" rating, which is a plain
  correlation (as in pilot 13's README). "Nine named features" are the blind Claude ratings.
- **`f4` counts (in `CAPTIONS.md`).**
  - A set that could cycle is two outcomes and two alternatives where each alternative clearly beats one outcome and
    loses to the other, with margins above 2 log-odds.
  - Llama 3.1 8B has 0 cycles of 229 such sets because its choices are much less extreme. With the margin scaled to
    its spread (0.48 log-odds), it has 3 of 1,292 (0.23%). Qwen3-32B at its scaled margin has 0.32%, and Qwen3-14B at
    2 log-odds has 0.77%.
  - Order flips (pairs of outcomes that swap order between alternatives) are common in all three models.
