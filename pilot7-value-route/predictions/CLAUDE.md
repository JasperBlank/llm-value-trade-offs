# Claude's predictions for pilot 7, with reasons

Written 2 Nov 2026 before Jasper's predictions and before any pilot-7 result; numbers in
`claude_predictions.yaml`.

**The route.** Pilot 5's answer-writing heads read mostly from the newlines and template tokens,
not from A's words, and the answer position gets nothing before layer 24. So A's information has to
sit in the text after A's line well before 28: I expect swapping everything after A's line to carry
half of a good swap by about layer 16 (8-24). A's own line end alone (its last token or the line
break after it) should carry a modest part, about 0.2 (0.03-0.6): later positions also read A's
other tokens directly. The rule should pick "everything after B's line" as the B site (80%): it
contains the other candidates.

**The monkey test.** Before B (the A site) I am unsure whether the model already integrates the
good and the number of people: they are separate words in the line, and the line-end summary may
keep them apart. A direction trained on good swaps carrying number swaps there: about 0.35 (wide).
After B, by layer 20-24, I expect the template positions to hold something close to pilot 6's
decision variable, which by construction mixes both: number share 0.7, and an exchange ratio near 1,
because the model's own values are fitted on its own logit differences. If it is the decision, B's
value should push it the other way: ρ about −0.6 (between "comparison" and "A's own value", since
the span also includes B's line break). Pilot 6's comparator was generic, so the value direction
after B should carry a fair part of an apples swap too: 0.5 (wide).

**Llama.** An 8B model is less consistent than Qwen3-14B (pilot 2), so 60% that its own values
predict its held-out choices with r ≥ 0.5. Its answer should arrive at a similar depth: 0.65 of the
layers (0.45-0.85).

**Overall.** One currency before B: 15% (the floor, then V2, then V3). After B: 50%. V5 after B:
comparison 50%, A's own value 35%, a shared code 7%, not testable 8%. Llama's B-site headline the same
as Qwen's: 60%.
