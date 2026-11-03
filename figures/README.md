# Figures

Drawn by `make_figures.py` straight from the committed result files of pilots 6 to 9
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
