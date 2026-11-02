# Claude's predictions for pilot 5 (sealed)

Written before Jasper's answers and before any pilot-5 result; only the SHA-256 of this file and
of claude_predictions.yaml are committed until Jasper has answered.

1. A's slope at layer 28: **0.55** (80%: 0.15 to 1.3). The last-position code is strong (option-level
   r 0.92) and the push now holds B's readout fixed, but later blocks keep reading the options' own
   tokens, and a ridge direction is only partly the model's own; pilot 4's 0.14 at the option line
   is the floor I anchor above.
2. B's slope at layer 28: **-0.45** (80%: -1.2 to -0.1); B's code was weaker in pilot 4 (0.6).
3. Controls specific at layer 28: **65%**. The random push should do nothing; the shuffled-values
   directions lie among the goods' representations and may share some of the decision direction.
4. (a) Largest A slope at layer **32** (24: 15%, 28: 25%, 32: 32%, 36: 28%). (b) Full size at some
   layer: **45%**.
5. cos(w_A, w_B) at 28: **+0.3** (80%: -0.4 to +0.8). The grid makes A's and B's values move
   together, so a shared "how much is at stake" component should outweigh the opposite-signed
   decision component.
6. One currency: (a) gain- vs harm-worded **55%**; (b) money vs other goods **50%**.
7. MLP share of A's value at 28: **0.5** (80%: 0.2 to 0.8); above half: **50%**.
8. Pieces for half: **12** (80%: 4 to 60).
9. First layer at which the whole last position carries half of a swapped good: **28**.
10. Top-20 writers' transfer: **0.25** (80%: 0.05 to 0.6). Later blocks re-read the recipient's own
    option tokens, and the writing is probably spread out.
11. A letter head among the top 10 readers: **60%**.
12. A top-10 writer carrying number and good in the right mix: **60%**.
13. Goal module 1, share of the choice's move in the value-difference readout at 28: **0.12**
    (80%: 0.02 to 0.45). Mostly a bypass, as in pilot 4.
14. Verdict at 28: full size **32%**, used but not full size **38%**, not specific **18%**, not
    shown **12%**.
