# LLM value trade-offs

Do language models trade values off against each other consistently, like the monkeys in
Padoa-Schioppa's juice-choice experiments? And if they do, where in the network does the
trade-off happen? Research by Jasper Blank; code written with Claude.

| Folder | Question | Status and main finding |
|---|---|---|
| [`pilot1-goal-sources/`](pilot1-goal-sources/) | When an LLM agent faces a trade-off between an organisation's target and a third party's wellbeing, what decides its choice: the goal in its prompt, its trained persona, or whether it is observed? | Run 29 Oct 2026, Qwen3-8B |
| [`pilot2-value-gate/`](pilot2-value-gate/) | Does the model hold consistent values (quantity sensitivity, one common scale, transitivity)? Does an assigned objective enter as extra value or override like a rule? | Run 30 Oct, Qwen3-4B/8B/14B. All fail the pre-registered gate; 14B closest. Objectives override like a rule |
| [`pilot3-value-gate/`](pilot3-value-gate/) | The redesigned gate (fresh goods, order and wording averaged, held-out exchange rates) | Run 31 Oct, 14B and 8B. 14B: held-out exchange rates within ×1.38 (median), one transitivity triplet off (×1.88); framing and decoy effects; one file beats 128 people under an objective |
| [`pilot4-value-code/`](pilot4-value-code/) | Exploratory: where can 14B's values be read? | Run 1-2 Nov. At the last position (layers 28-40), a probe orders held-out goods by value (r 0.92 over 9 options; random-weights control -0.24); weak at the options' own tokens |
| [`pilot5-value-circuit/`](pilot5-value-circuit/) | Pre-registered: is that value arrow used? Exploratory: what writes and reads it; other answer formats | Run 2 Nov. Pushing the arrow moves the choice only 7% of the predicted amount ("an effect, not specific"). But the decision forms in blocks 24-27 at the last position (patching 0.03 at layer 24, 0.77 at 28), and 20 pieces (MLPs 25-27, heads in blocks 24-27) carry 65% of a swapped good. Separate codes for A and B; the same circuit for numbered answers; not used to rate a single outcome |
| [`pilot6-value-subspace/`](pilot6-value-subspace/) | Pre-registered: which direction does the model actually use at the last position, and is pilot 5's circuit about value at all? | Run 2 Nov. **Generic comparator:** "which pile has more apples" and "which is heavier" arrive at the answer position at the same layer (28) and use the same 20 pieces (0.54 and 0.39 transfer, value 0.64). One learned direction carries 99% of the whole-state swap (the probe: 1%), but it is the decision itself (correlation 0.99 with the model's logit difference), not A's value. Next: look upstream, at the options' own tokens |
| [`pilot7-value-route/`](pilot7-value-route/) | Pre-registered: where is option A's value formed before the decision stage? One value for A before the model reads B (the monkeys' "offer value")? Qwen3-14B and Llama 3.1 8B | Run 2 Nov. **No single value for A before B** in either model: at A's line end Qwen holds the good but not the number of people; Llama holds both, in different directions. After B, Qwen has a value-specific channel at the line breaks (layer 24) that carries good and number changes and moves oppositely for A and B: a comparison, not a price per option. Exchange-rate test failed (noisy). Llama barely weighs the number of people |

## How pre-registration works here

For pilots 2, 3, 5, 6 and 7, the design, the analysis code, the thresholds and both sets of predictions
(Jasper's and Claude's, Claude's sealed first) are committed before anyone sees a result. The format chosen in the tune phase is committed before the test phase. Results come
last. The commit history is the record.
