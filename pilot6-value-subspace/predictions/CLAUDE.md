# Claude's predictions for pilot 6, with reasons

Written 2 Nov 2026 before Jasper's predictions and before any pilot-6 result. The numbers are in
`claude_predictions.yaml`.

**P1 (generic comparator?).** I expect the non-value questions to move into the last position at
about the same stage as value (layer 28, range 24-32): putting the chosen option's content at the
answer position looks like the standard multiple-choice machinery at about two thirds of the depth,
not something special to value. Whether pilot 5's top-20 pieces carry them is less clear. The
heads among them look like movers that copy option content to the last position, which would be
generic; MLPs 25-27, which carry the most, could be doing value-specific work. Hence transfer about
0.4 on each control (wide range), and generic 40%, mixed 35%, value-specific 25%. The top-20 list
was chosen on the value task, which pushes towards "value-specific" by selection alone.

**P2 (one direction?).** The answer is binary, and by layer 28 the model is already leaning one
way, so one direction that holds "how much A beats B" should carry most of the whole-state swap:
0.65 (range 0.3-0.9), 70% that it reaches 0.5. Held-out wordings cost some. Smallest k for 0.9:
best guess 4, with "none" plausible (20%) because the whole-state swap at 28 itself carries only
0.77 of the donor's answer and part of it may be spread thin.

**The probe as a direction.** Pilot 5's push along the probe gave about 7% of the expected effect,
so its swap should carry little: 0.08 (0.01-0.3). The learned direction and the probe should be
nearly unrelated in 5,120 dimensions, but not orthogonal: |cos| 0.15 (0.03-0.5).

**P4 (real or illusion?).** If the learned direction is the decision ("A beats B") rather than A's
value, its correlation with A's value alone is limited by B's value varying too: about 0.45
(0.15-0.8), and I expect a higher correlation with V_A − V_B. Lying in the writers' space at least
5 × chance seems likely (about 75%) if it is real, since MLPs 25-27 write most of what moves the
answer. Together: 40%.

**P3 (a better lever?).** Calibrated to natural variation, a push along the variable the model
reads should do roughly what one value unit from the prompt does, against the probe's 7%. That
argues for a large ratio, but the natural-unit calibration is noisy and DAS directions can be
partly dormant (a small natural slope means a small push). Ratio 5 (1-20); 50% that P3 passes,
counting the random-direction condition.
