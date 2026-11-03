# Figures

Drawn by `make_figures.py` straight from the committed result files of pilots 6 to 12
(`python figures/make_figures.py`). Colours: the reference palette of the dataviz guidance
(categorical slots 1-3 validated for colour-blind separation; a single-hue blue ramp for the maps).

| File | Shows |
|---|---|
| `fig1_readable_vs_used.png` | Pilot 6: the value probe reads A's value, but the decision runs along another direction |
| `fig2_generic_stage.png` | Pilot 6: value, apples and weight reach the answer position at layer 28, through the same pieces |
| `fig3_route_qwen.png` | Pilot 7: where the options' information sits, layer by layer (Qwen3-14B) |
| `fig3b_route_llama.png` | Pilot 7: the same for Llama 3.1 8B |
| `fig4_monkey_test.png` | Pilot 7: what one value direction carries before and after option B (Qwen3-14B) |
| `fig5_price_list.png` | Pilot 8: Qwen3-14B's price for each of 24 intrinsic values in trade-offs, next to its stated rating |
| `fig6_sentences_not_values.png` | Pilot 8 (exploratory): prices are stable per sentence (r 0.91) but not per value (r 0.20) |
| `fig7_pilot8_tests.png` | Pilot 8: the pre-registered tests against their bars |
| `fig8_vibe_check.png` | Pilot 9: the direct 'how good would it be' rating of each sentence against its menu price, and within each value |
| `fig9_full_range.png` | Pilot 10: the 'how good' rating against the menu price over the full range of goodness (96 sentences), and pilot 9's sentences on a log scale |
| `fig10_vibe_direction.png` | Pilot 11: where the price direction can be read from a sentence (held out), and what pushing along it does to the choice and the rating |
| `fig11_recipes.png` | Pilot 12: where the rating disagrees with the price can be read from the sentence, and pushing along it moves the rating beyond what a shared variable would |
| `fig12_recipe_in_words.png` | Pilot 13: the choice's and the stated rating's recipes in named features (rated blind by Claude), how well each predictor prices Qwen's outcomes, and the pushes at layer 20 against 20 random directions |
| `fig13_price_writers.png` | Pilot 14: which heads and MLP layers write the layer-20 price readout, what silencing one block at the sentence does to the choice, and single-word deletions: the choice's price change against the readout's |
| `fig14_price_route.png` | Pilot 15: where a price that follows the words is readable (positions × layers), which single position carries a swap of A's sentence to the answer, and pilot 5's comparator pieces at the answer |
| `fig15_relative.png` | Pilot 16: the share of an outcome's price that is the same whatever the alternative is (choice and inside), the 96 outcomes against two alternatives (order flips), and how each alternative stretches or squeezes the differences |
| `fig16_three_models.png` | Pilot 17: Llama 3.1 8B, Qwen3-14B and Qwen3-32B: how much of a price is fixed, when the decision reaches the answer (value menus and apples), and the decision writers' transfer on value, apples and weight swaps |
