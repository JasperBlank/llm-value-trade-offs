# Pilot 15: prediction questions

Answered before any result: Claude first (sealed, only the hash committed), then Jasper.

1. **Q1, is A's price finished before B is read?** The best readout on option A's own positions (its words, its last word, the line break after it), trained on the 96 intact sentences: its correlation with the choice's change across pilot 14's word deletions (held out). Best guess and 80% range. (Pilot 14's single direction at layer 20 got 0.28.)
2. **Q2, does A's sentence pass through one place?** Probability that swapping the state at one position (A's last word, the line break after A, B's last word or the line break after B; layers 16-32) carries at least half of a swap of A's whole sentence into the choice.
3. **Q3, do pilots 5-6's comparator pieces compare these prices?** How much of a swap of A's sentence pilot 5's top-20 pieces carry at the answer position. (With goods they carried 0.65; with apples 0.54.) Best guess and 80% range.
4. **Q4, pilot 14's Q4 redone:** probability that silencing pilot 14's 10 top MLPs at the sentence removes at least 30% of the choice's dependence on the price and more than each of 5 random sets of 10 other MLPs (now without block 1). In pilot 14 the top 10 removed 32%.
