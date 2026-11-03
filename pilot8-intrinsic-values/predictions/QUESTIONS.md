# Pilot 8: prediction questions

Answered before any result: Claude first (sealed, only the hash committed), then Jasper.

1. **The top value:** which of the 24 values does the model price highest (worth per person)?
2. **A consistent menu (Q3):** probability that all three checks pass (one scale, at most 5% cyclic
   triads, a third option changes the A:B odds by a median factor of at most 1.5).
3. **The same thing lights up (Q2):** probability that both Q2a (the worth direction learned on 12
   values orders the other 12) and Q2b (the comparison direction learned on 12 values carries
   trade-offs among the other 12) pass.
4. **Stated ratings match the trade-offs (Q4a):** probability that r ≥ 0.7.
5. **The same value representation in the rating prompt (Q4b):** probability that the trade-off worth
   direction orders the held-out values in the rating prompts with r ≥ 0.7.
