# Pilot 14: prediction questions

Answered before any result: Claude first (sealed, only the hash committed), then Jasper.

1. **Q1, a few words set the price:** probability that, for the median sentence, its two most important words carry at least half of the summed word-deletion effects.
2. **Q2, the readout follows the choice word by word:** the correlation, over the 994 single-word deletions, between the change in the price readout and the change in the price the choice shows. Best guess and 80% range.
3. **Q3, a few parts write the price:** the summed share of the readout's spread carried by the 10 largest of the 820 attention heads and MLP layers. Best guess and 80% range. (Shares can be negative, so the top 10 can sum to more than 1.)
4. **Q4, the choice uses them:** probability that silencing those 10 parts at the sentence removes at least 30% of the choice's dependence on its price, and more than each of 5 matched random sets.
