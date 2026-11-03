# Pilot 8: prediction questions (revised design: domain and size)

Answered before any result: Claude first (sealed, only the hash committed), then Jasper. Numbers: a
best guess and an 80% range. Probabilities: 0-100%. Scored as in pilots 5-7.

1. **Worth = domain + log(size) (W1):** probability that one simple curve per domain (worth rises by
   a fixed amount each time the amount is multiplied by the same factor) predicts held-out outcomes
   with R² ≥ 0.8.
2. **The price of health:** how many euros "a day of good health" is worth to the model (per person,
   read off the fitted curves). Best guess and 80% range.
3. **One sensitivity to size (W3):** probability that worth grows with size equally fast (within a
   factor 1.5) in all three domains, on a log scale.
4. **Assembled on the outcome's own words (W2):** probability that the worth, with its growth in
   size, is already put together on the outcome's own words, before the model reads the other option.

The questions of the first design (five ingredients) are in the git history; Jasper's answers to them
are in `JASPER_v1.md`.
