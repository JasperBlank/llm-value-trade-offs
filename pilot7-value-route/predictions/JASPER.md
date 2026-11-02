# Jasper's predictions for pilot 7

Given on 2 Nov 2026, 16:23-16:28 Amsterdam time, after Claude's predictions were sealed (hash in
`SEALED.md`) and while the run's outputs were hidden on the pod. Jasper asked for fewer questions,
so he answered a short set of five (the most decisive ones); the final values are in `predictions.yaml`.

| Short-set question | In QUESTIONS.md | First answer | Final |
|---|---|---|---|
| 1. One value before B (A's line end): probability of "one currency" | 11a | 40% ("I think somewhere in there it does hold it") | same |
| 2. One value after B: probability of "one currency" | 11b | 40% | same |
| 3. After B, the share of an apples swap the value direction carries | 8 | 0.5 (0.1-0.9) | same |
| 4. What sits after B: comparison / A's own value / shared code / not testable | 11c | 50 / 10 / 10 / 30% | 30 / 30 / 30 / 10% |
| 5. Llama's after-B headline the same as Qwen's | 11d | 70% | same |

**Consistency check.** Claude pointed out that a 50% chance of "the comparison" fits poorly with only
40% for "one currency" after B (a comparison would usually pass that test), and that "not testable"
needs the text after B to carry almost nothing of a B swap, which is unlikely. Jasper changed
question 4 and kept the rest.
