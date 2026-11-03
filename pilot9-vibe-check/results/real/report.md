# Pilot 9 results (Qwen/Qwen3-14B)

576 rating prompts; the digits 0-9 take at least 1.000 of the answer probability (median 1.000). Menu prices: pilot 8's per-sentence prices.

## Pre-registered tests (W0: 'How good would it be if the following happened?')

- **T1** rating at 4 people against the menu price, 48 sentences: r = 0.55 (rank 0.54): **fail** (bar 0.7)
- **T2** the rating prefers the same sentence of a value as the menu in **20 of 24** values (one-sided p = 0.001; the 24 differences correlate r = 0.55): **pass** (bar 17)
- **T3** 8 people against 2: the rating rises by +0.43 points on average: **not size-sensitive** (bar 0.5)

**Verdict: partly shared.**

## All wordings

| wording | T1 r | T2 same of 24 | within-value r | 8 vs 2 people | mean (SD) at 4 people |
|---|---|---|---|---|---|
| W0 how good | 0.55 | 20 | 0.55 | +0.43 | 6.99 (0.87) |
| W1 how much better the world | 0.49 | 20 | 0.49 | +0.31 | 6.40 (1.33) |
| W2 how bad or good | 0.47 | 18 | 0.39 | +0.19 | 7.20 (0.74) |
| W3 how much it matters to you | 0.40 | 16 | 0.41 | +0.32 | 3.28 (1.00) |
| mean of W0-W2 | 0.56 | 20 | 0.61 | +0.31 | 6.86 (0.87) |

Between wordings (4 people): W0-W1 0.83, W0-W2 0.72, W0-W3 0.64, W1-W2 0.45, W1-W3 0.60, W2-W3 0.58.
Value level (mean of a value's two sentences against pilot 8's value prices; pilot 8's Q4a was 0.54): W0 0.58, W1 0.54, W2 0.54, W3 0.38.

## Size (W0)

- The rating rises 0.31 points per log(people). In the menu's units (scaled by the spread of prices over the spread of ratings) that is 0.57, against the menu's b = 2.11.
- Per sentence, the rating's slope against the menu's own weight of log(people): r = 0.18.

## The values whose two sentences differ most in the menu

| value | sentence | menu price | W0 rating (4 people) |
|---|---|---|---|
| Diversity | each get to know someone very different from themselves | -2.60 | 6.90 |
|  | each become part of a community made up of many different kinds of people | +2.16 | 6.95 |
| Spirituality | each have a deeply meaningful spiritual experience | -1.88 | 6.99 |
|  | each feel a profound connection to something greater than themselves | +1.40 | 7.00 |
| Non-suffering | are each spared a week of pain | +2.46 | 6.93 |
|  | are each relieved of a painful condition for a week | -0.74 | 6.43 |
| Beauty | each spend an afternoon somewhere breathtakingly beautiful | -0.87 | 7.15 |
|  | each get to see something so beautiful it moves them deeply | +2.14 | 7.95 |
| Respect | are each treated with respect by people who looked down on them | -2.38 | 7.02 |
|  | each finally receive the recognition and respect they deserve | +0.50 | 8.72 |
| Preserving | each see a place they love protected from destruction | +1.82 | 7.00 |
|  | each see an old tradition they cherish kept alive | -0.60 | 7.05 |
| Nurturing | each help a young person grow and flourish | +2.97 | 8.00 |
|  | each get to guide a child through an important stage of growing up | +0.68 | 6.95 |
| Happiness | each feel happier for a week | -3.62 | 5.83 |
|  | are each in a noticeably better mood for several days | -1.36 | 5.95 |

## Where rating and price disagree most (W0)

| | value | sentence | menu price | rating |
|---|---|---|---|---|
| rated higher than its price | Respect | each finally receive the recognition and respect they deserve | +0.50 | 8.72 |
|  | Protection | are each kept safe from a danger that threatened them | +3.03 | 8.85 |
|  | Embodying | each live up to the kind of person they want to be when it is difficult | +0.25 | 7.82 |
|  | Respect | are each treated with respect by people who looked down on them | -2.38 | 7.02 |
|  | Diversity | each get to know someone very different from themselves | -2.60 | 6.90 |
| rated lower than its price | Longevity | each live a year longer | -2.71 | 3.08 |
|  | Longevity | each have their life extended by a year | -0.76 | 4.57 |
|  | Non-suffering | are each spared a week of pain | +2.46 | 6.93 |
|  | Diversity | each become part of a community made up of many different kinds of people | +2.16 | 6.95 |
|  | Happiness | are each in a noticeably better mood for several days | -1.36 | 5.95 |
