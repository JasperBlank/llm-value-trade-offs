# Pilot 7 results (Qwen3-14B and Llama 3.1 8B, run 2 Nov 2026)

**The run.**

- It used one A100 SXM 80 GB pod on Runpod, from 15:14 to 17:32 UTC.
- Qwen ran from 15:16 to 15:50. Llama ran from 16:46 to 17:07, after the Hugging Face token was fixed and the pod restarted.
- The pod sat idle for about 80 minutes in total, because two scheduled check-ins fired 25–43 minutes late. The total cost was about $4.10, of which about $2.40 was idle time.
- Both prediction sets were committed before any result was printed.
- The files here were rebuilt from the pod's log. The second log was cut off by the log reader's size limit, but only after every new file had been printed in full. Qwen's learned directions come from the first log.

| File | What |
|---|---|
| `report.md` | The report for both models, from the pod |
| `report_qwen_only.md`, `verdicts7_qwen_only.json` | The same, from the first start (Qwen only) |
| `<model>/fit.json` | Each model's own values, fitted from its choices |
| `<model>/map.json` | Site × layer patching for each kind of swap |
| `<model>/das.json`, `<model>/directions.npz` | The monkey test: the learned directions (float16) and their evaluations |
| `verdicts7.json` | The pre-registered verdicts |
| `predictions_vs_results.md` | Both prediction sets scored |
| `log_pod_qwen.txt`, `log_pod_llama.txt` | The pod logs (Llama's without the long base64 lines) |

## Verdicts (pre-registered, DESIGN.md section 4)

| | Qwen3-14B | Llama 3.1 8B |
|---|---|---|
| Own values: held-out r; weight of the number of people (b) | 0.72; b = 5.1 (goods correlate 0.99 with pilot 3) | 0.68; **b = 0.29**, with a bias of 5.1 logits towards answering A |
| Arrival at the answer position | layer 28 of 40 (0.70) | layer 16 of 32 (0.50) |
| **A site** (before B's line) | A's last token, layer 12: carries 0.34 of a good swap but **0.00 of a number swap**. No single value. | The line break after A, layer 8: carries 0.29 of a good swap and 0.23 of a number swap, **in different directions** (cosine 0.13). No single value. |
| **B site** (after B's line) | Everything after B, layer 24: one direction carries the good swaps (1.02) and the number swaps (0.76): **V2 passes**. The exchange rate fails (V3, ratio 2.06). V4: value-specific (apples 0.11, weight 0.10). V5: **the comparison** (ρ = −2.5). Headline: no single value. | Everything after B, layer 12: good swaps 0.84, number swaps 0.44: V2 fails. V3 fails. V4: value-specific (apples 0.10). V5 by the rule: "shared code", but this is not reliable (see the cautions). Headline: no single value. |

## The route (the maps)

**In both models the pattern is the same:**

1. A's information leaves A's line and gathers at the **line breaks** after A and after B, and in the text after B.
2. In Qwen this happens at layers 16–24; in Llama, at layers 8–12.
3. It then jumps to the answer position in one step: at layer 28 in Qwen, at layer 16 in Llama.

**In Qwen the good and the number take different paths.**

- The good sits at A's last token at layers 4–16 (about 0.3 of its effect).
- The number never does (0.00). Instead, B's line has picked up A's number by layer 16 (0.40).

**The non-value questions take a different route.**

- In Qwen, "more apples" information sits mostly in B's line (0.44 at layer 24), not in the line breaks (0.08).
- "Heavier" sits at A's last token until layer 16, which here is the object itself.

So the line-break route is used much more for values than for apples.

## Cautions

- **The coordinate tests are weak for good swaps.** V3 (the exchange rate) and V5 (comparison or not) use how much the learned direction's coordinate changes per unit of the model's own value.
  - For good swaps, that change barely correlates with the value change: Qwen −0.11, Llama −0.04. This holds even though swapping along the direction carries most of the good swaps' effect.
  - So the good-swap slope is noisy. Qwen's ratio of 2.06 is uncertain.
  - Llama's V5 ("shared code", ρ = 36) divides by a slope of almost zero, so it is not meaningful.
- **The number swaps are clear in both models.** At the B site, A's number and B's number move the coordinate in opposite directions: Qwen −0.59 and +0.68, Llama +0.71 and −0.67. That is a comparison signal.
- **Llama barely weighs the number of people.** Doubling the people moves its answer by about 0.2 logits, while the goods move it by 2–4. It also leans strongly towards answering A. It passed the pre-registered usability bar, but "integrating good and number" means little for a model that hardly uses the number.
- **The rule chose Qwen's A site.** It picked A's last token, which carries the good but not the number. The line break after A carries some of both (0.29 good, 0.14 number) but was not tested with a learned direction.
- These results come from one answer format, 9 goods, and patching one span at a time.

## Reading

Neither model forms a single value for option A on its own, before it has read option B. In Qwen, once B has been read, there is a value-specific channel (layer 24, at the line breaks after the options). It carries changes in both the good and the number, and moves in opposite directions for A and B. In other words, it is a comparison of the two options, not a price for each. That comparison is then handed to the generic decision stage at layer 28 (pilot 6).

This is not the monkeys' "offer value" picture, where each option gets its own value before the choice. It is closer to "compare first": attribute by attribute, with a value-specific comparison signal formed before the shared decision machinery. The exchange-rate test could not confirm that this comparison weighs good and number at the model's own rate.
