# Pilot 10 results (Qwen/Qwen3-14B)

Menu: 45600 trade-offs; P(A) = sigmoid(u_A - u_B + b log(nA/nB) + c), b = 1.45, bias towards A +1.05; the fit orders the answers with r = 0.84; split-half reliability of the 96 prices r = 1.00. Letters take at least 1.000 of the answer probability.

## Pre-registered tests

- **F1** the full range: 'How good would it be …' (8 people) against the menu price, 96 sentences: r = 0.81: **pass** (bar 0.8)
- **F2** the log-scale score on pilot 9's 48 sentences (4 people) against pilot 8's prices: r = 0.50: **fail** (bar 0.7; the expected rating gave 0.55)
- **F3** 32 people against 2: the rating rises by +0.74 points on average: **not size-sensitive** (bar 1.0)

## The menu across the range

- Designed order (small < both medium < large) holds for 17 of 24 values in the menu, 17 in the rating.
- Share of the price spread explained by the stakes level alone: 0.63; by the value alone: 0.20; both: 0.83. For the rating: 0.57, 0.26, 0.83.
- Pilot 8's 48 sentences in this new menu against their pilot 8 prices: r = 0.98; the same sentence of each pair wins in 22 of 24.
- Mean W0 rating at 8 people: small 5.67 (SD 1.04), medium 7.21 (SD 0.80), large 8.37 (SD 0.61); 31% of W0 answers within 0.1 of 7.

## By readout and wording

| readout | wording | all 96 (8 people) | within stakes level | pilot 8's 48, new menu | pilot 8's 48 (4 people), pilot 8 prices | pairs: same as new menu (of 24) |
|---|---|---|---|---|---|---|
| rating | W0 how good | 0.81 | 0.54 | 0.51 | 0.55 | 18 |
| rating | W1 how much better the world | 0.75 | 0.50 | 0.52 | 0.48 | 21 |
| rating | W2 how bad or good | 0.74 | 0.50 | 0.42 | 0.47 | 16 |
| rating | W3 how much it matters to you | 0.74 | 0.46 | 0.48 | 0.40 | 17 |
| score | W0 how good | 0.83 | 0.59 | 0.52 | 0.50 | 18 |
| score | W1 how much better the world | 0.80 | 0.57 | 0.63 | 0.57 | 18 |
| score | W2 how bad or good | 0.75 | 0.49 | 0.36 | 0.39 | 16 |
| score | W3 how much it matters to you | 0.76 | 0.50 | 0.48 | 0.45 | 18 |
| lean7 | W0 how good | 0.83 | 0.58 | 0.54 | 0.56 | 18 |
| lean7 | W1 how much better the world | 0.81 | 0.64 | 0.65 | 0.65 | 19 |
| lean7 | W2 how bad or good | 0.73 | 0.47 | 0.37 | 0.43 | 16 |
| lean7 | W3 how much it matters to you | 0.69 | 0.48 | 0.48 | 0.42 | 17 |

## Size (W0)

- Gain from 2 to 32 people by level: small +0.76, medium +0.69, large +0.81.
- Per log(people): 0.27 points; in the menu's units 0.66 against b = 1.45.

## The price list (menu price; W0 rating at 8 people)

| # | value | level | sentence | price | rating |
|---|---|---|---|---|---|
| 1 | Legacy | large | each create a lasting work that helps people for generations after they are gone | +6.41 | 9.00 |
| 2 | Nurturing | large | each raise a child who grows up healthy, kind and happy | +5.71 | 8.99 |
| 3 | Diversity | large | each help build a town where people of every background are truly welcome | +5.47 | 8.43 |
| 4 | Freedom | large | are each freed after years of being wrongly imprisoned | +5.22 | 9.00 |
| 5 | Spirituality | large | each find a deep sense of meaning and peace that stays with them for the rest of their lives | +4.42 | 9.00 |
| 6 | Knowing | large | each gain a deep understanding that transforms how they see the world | +4.38 | 7.80 |
| 7 | Embodying | large | each become, over the years, exactly the kind of person they always wanted to be | +4.18 | 8.89 |
| 8 | Protection | large | are each saved from being killed in a fire | +3.94 | 8.77 |
| 9 | Fairness | large | each have years of unfair treatment finally set right and fully compensated | +3.91 | 8.99 |
| 10 | Caring | large | are each able to be there for a loved one through the hardest time of their life | +3.88 | 8.76 |
| 11 | Nurturing | medium | each help a young person grow and flourish | +3.72 | 8.46 |
| 12 | Doing | large | each get to spend the rest of their working life doing the work they love most | +3.35 | 8.69 |
| 13 | Freedom | medium | are each no longer prevented from making their own choices about their lives | +3.34 | 8.16 |
| 14 | Legacy | medium | each create something that will outlast them | +3.17 | 7.05 |
| 15 | Protection | medium | are each kept safe from a danger that threatened them | +3.12 | 8.99 |
| 16 | Longevity | large | each live twenty more years in good health | +3.04 | 7.06 |
| 17 | Diversity | medium | each become part of a community made up of many different kinds of people | +2.96 | 6.98 |
| 18 | Experiencing | large | each go on the journey of a lifetime that they had dreamed of for decades | +2.80 | 8.72 |
| 19 | Beauty | medium | each get to see something so beautiful it moves them deeply | +2.75 | 8.06 |
| 20 | Preserving | large | each see their hometown saved from being destroyed by a flood | +2.69 | 8.22 |
| 21 | Spirituality | medium | each feel a profound connection to something greater than themselves | +2.52 | 7.00 |
| 22 | Happiness | large | each become deeply happy for the rest of their lives | +2.48 | 8.98 |
| 23 | Loyalty | large | each have a friend who stands by them through the worst crisis of their life | +2.36 | 8.13 |
| 24 | Preserving | medium | each see a place they love protected from destruction | +2.29 | 7.03 |
| 25 | Non-suffering | medium | are each spared a week of pain | +2.22 | 7.03 |
| 26 | Knowing | medium | each come to understand something important about the world | +2.19 | 7.00 |
| 27 | Freedom | medium | each gain the freedom to decide how they spend their own time | +2.09 | 7.01 |
| 28 | Defending | large | each save a stranger from a violent attack | +2.06 | 8.94 |
| 29 | Embodying | medium | each manage to act according to their ideals in a hard situation | +1.99 | 7.16 |
| 30 | Non-suffering | large | are each cured of a severe pain they have suffered from for years | +1.93 | 8.24 |
| 31 | Protection | medium | are each shielded from a threat to their safety | +1.65 | 8.39 |
| 32 | Achievement | medium | each achieve a goal they have worked towards for a long time | +1.57 | 7.76 |
| 33 | Loyalty | medium | each find that the people close to them stay faithful when it matters | +1.55 | 7.00 |
| 34 | Nurturing | medium | each get to guide a child through an important stage of growing up | +1.53 | 6.96 |
| 35 | Respect | medium | each finally receive the recognition and respect they deserve | +1.48 | 8.82 |
| 36 | Legacy | medium | each leave behind something that will be remembered after they are gone | +1.35 | 7.00 |
| 37 | Beauty | large | each see a sight so beautiful that they remember it for the rest of their lives | +1.28 | 7.93 |
| 38 | Embodying | medium | each live up to the kind of person they want to be when it is difficult | +1.17 | 7.88 |
| 39 | Respect | large | each finally earn the lasting respect of everyone who once looked down on them | +1.13 | 7.97 |
| 40 | Caring | medium | are each able to look after a loved one who needs them | +1.07 | 7.96 |
| 41 | Achievement | medium | each finally succeed at something they have spent years trying to do | +0.86 | 7.05 |
| 42 | Defending | medium | each successfully stand up for someone who was being mistreated | +0.71 | 7.85 |
| 43 | Experiencing | medium | each try something exciting they have always wanted to experience | +0.70 | 7.05 |
| 44 | Loyalty | medium | each have a friend stand by them through a hard time | +0.67 | 7.35 |
| 45 | Knowing | medium | each finally learn the answer to a question that has long puzzled them | +0.50 | 6.99 |
| 46 | Defending | medium | each protect a stranger from being bullied | +0.49 | 7.65 |
| 47 | Caring | medium | each get the chance to help someone they care about | +0.46 | 6.60 |
| 48 | Doing | medium | each get to spend time on a hobby they are passionate about | +0.34 | 7.20 |
| 49 | Pleasure | large | each enjoy a perfect week of rest, good food and fun | +0.29 | 7.77 |
| 50 | Preserving | medium | each see an old tradition they cherish kept alive | +0.29 | 7.23 |
| 51 | Achievement | large | each achieve the greatest ambition of their life | +0.27 | 7.95 |
| 52 | Fairness | medium | each receive the fair share they had wrongly been denied | +0.17 | 8.18 |
| 53 | Pleasure | medium | each have a thoroughly enjoyable evening | +0.03 | 7.47 |
| 54 | Experiencing | medium | each have a new, memorable experience they have never had before | +0.01 | 7.04 |
| 55 | Justice | medium | each see a person who harmed them face fair consequences | -0.10 | 7.01 |
| 56 | Longevity | medium | each have their life extended by a year | -0.22 | 4.94 |
| 57 | Doing | small | each get ten minutes to do something they enjoy | -0.45 | 7.00 |
| 58 | Beauty | medium | each spend an afternoon somewhere breathtakingly beautiful | -0.47 | 7.71 |
| 59 | Doing | medium | each spend a day doing an activity they love | -0.51 | 7.00 |
| 60 | Non-suffering | medium | are each relieved of a painful condition for a week | -0.62 | 6.99 |
| 61 | Justice | large | each see the person who ruined their life brought to justice | -0.66 | 7.76 |
| 62 | Protection | small | are each warned about a slippery step before they reach it | -0.91 | 7.39 |
| 63 | Happiness | medium | are each in a noticeably better mood for several days | -0.95 | 6.77 |
| 64 | Legacy | small | each leave a kind note that someone finds years later | -0.97 | 7.05 |
| 65 | Caring | small | each get to make a cup of tea for someone they care about | -1.06 | 6.82 |
| 66 | Truth | medium | each learn the truth about something they had been deceived about | -1.07 | 6.94 |
| 67 | Spirituality | medium | each have a deeply meaningful spiritual experience | -1.10 | 7.00 |
| 68 | Fairness | medium | each have an unfair decision against them reversed | -1.29 | 7.00 |
| 69 | Freedom | small | each get to choose which film to watch tonight | -1.42 | 5.11 |
| 70 | Justice | medium | each see someone who wronged them held accountable | -1.53 | 7.03 |
| 71 | Diversity | medium | each get to know someone very different from themselves | -1.75 | 6.98 |
| 72 | Respect | medium | are each treated with respect by people who looked down on them | -1.77 | 7.12 |
| 73 | Spirituality | small | each feel a brief sense of awe while looking at the stars | -1.88 | 5.18 |
| 74 | Pleasure | medium | each enjoy a delicious meal | -1.92 | 6.92 |
| 75 | Truth | medium | are each told the truth instead of being lied to | -2.13 | 6.98 |
| 76 | Longevity | medium | each live a year longer | -2.28 | 4.20 |
| 77 | Truth | large | each finally learn the truth about a lie that shaped their whole life | -2.34 | 6.94 |
| 78 | Preserving | small | each see an old bench in their park repaired | -3.21 | 5.25 |
| 79 | Happiness | medium | each feel happier for a week | -3.24 | 6.12 |
| 80 | Loyalty | small | each get a message from a friend checking in on them | -3.29 | 6.87 |
| 81 | Fairness | small | each get back a small amount they were overcharged on a bill | -3.83 | 5.20 |
| 82 | Defending | small | each speak up for a colleague who was interrupted in a meeting | -4.04 | 7.00 |
| 83 | Pleasure | small | each enjoy a piece of chocolate | -4.05 | 5.11 |
| 84 | Diversity | small | each try a dish from a culture they have never encountered | -4.16 | 6.13 |
| 85 | Longevity | small | each live one day longer | -4.25 | 2.93 |
| 86 | Beauty | small | each notice a pretty flower on their way to work | -4.28 | 4.94 |
| 87 | Knowing | small | each learn a fun fact | -4.28 | 5.33 |
| 88 | Non-suffering | small | are each spared a mild headache | -4.35 | 5.45 |
| 89 | Achievement | small | each finish a small task they had been putting off | -4.54 | 5.90 |
| 90 | Nurturing | small | each help a child tie their shoes | -4.55 | 6.83 |
| 91 | Happiness | small | each smile at a funny joke | -5.01 | 6.21 |
| 92 | Truth | small | each learn the real reason a meeting was moved | -5.66 | 4.89 |
| 93 | Embodying | small | each keep a small promise they made to themselves | -6.04 | 5.15 |
| 94 | Experiencing | small | each taste a fruit they have never tried before | -6.47 | 5.02 |
| 95 | Justice | small | each see someone who cut in line sent to the back | -8.41 | 4.48 |
| 96 | Respect | small | each get a polite nod from a stranger | -9.10 | 4.95 |
