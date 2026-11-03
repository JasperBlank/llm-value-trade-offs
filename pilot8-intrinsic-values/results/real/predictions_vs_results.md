# Pilot 8: predictions against results

| question | result | Claude | Jasper |
|---|---|---|---|
| q1_top_value | Freedom | Longevity: ranked 23 of 24 | Protection: ranked 2 of 24 |
| q2_menu_consistent | False | 15%; Brier 0.022 | 25%; Brier 0.062 |
| q3_same_thing_lights_up | False | 30%; Brier 0.090 | 60%; Brier 0.360 |
| q4_ratings_match | False | 50%; Brier 0.250 | 50%; Brier 0.250 |
| q5_same_value_in_rating | False | 30%; Brier 0.090 | 40%; Brier 0.160 |

Mean Brier over the four probability questions (lower is better; always saying 50% scores 0.250): Claude 0.113, Jasper 0.208

Claude's sealed file also gave probabilities for the top value: Longevity 0.35, Protection 0.20, Non-suffering 0.15, Justice 0.07, Fairness 0.07, any other value 0.16. The top value, Freedom, falls under 'any other value'.
