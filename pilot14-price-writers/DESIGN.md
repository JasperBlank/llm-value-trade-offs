# Pilot 14: what writes the price (the price circuit, part 1)

## 1. The question

Jasper: *why does the model pick the options it picks?* Pilots 10-13 found the following:
- As Qwen reads an outcome, it writes a price onto the outcome's own words, before it sees the alternative (readable at layers 16-28; pushes work up to about layer 24).
- The choice partly uses that price.
- Named features describe the price only up to about r 0.8. About half of a new outcome's price belongs to its exact wording.

This pilot asks **what writes the price**:
1. **Which words** of a sentence set its price?
2. **Which parts of the network** (attention heads, MLP layers) write it onto the sentence?
3. **Does the choice depend on those parts?** Silencing them at the sentence should make the choice blind to its price.

Pilot 15 takes the next step: how the two prices are compared at the answer. It will test whether the generic comparator that pilots 5-6 found (blocks 24-27 at the answer position, shared with "which pile has more apples") is the same one that compares these prices.

## 2. Material and measures

- **Model:** Qwen3-14B (bf16, thinking off).
- **Sentences:** pilot 10's 96, with their menu prices. Menus use pilot 10's prompt at 8:8.
- **The price readout:** pilot 11's price direction v at layer 20 (difference of means between the highest- and lowest-priced thirds, menu context). A sentence's readout is its projection on v, averaged over its tokens, read as option A before B. Held-out r with the price: 0.78 (pilot 11).
- **The choice's price test:** each sentence as option A against 8 fixed partners spanning the price range (the sentences nearest the 1/16, 3/16, … 15/16 price quantiles; a sentence is not paired with itself).
  - The choice's dependence on the price is the slope of a sentence's mean logit P(A) on its price, across the 96.
  - Silencing parts of the network at A's tokens and measuring the slope again gives the fraction of the dependence lost.

### Words: occlusion

- Every word of every sentence is deleted in turn: 994 variants.
- For each variant, two changes are measured:
  - the change in the readout;
  - the change in its mean logit against the same 8 partners. In price units this is the change in its price.

### Components: direct attribution

- The state at layer 20 is exactly the sum of the token embeddings and the outputs of every attention head and MLP in blocks 1-20.
- So the readout splits exactly into 820 parts (20 × 40 heads + 20 MLPs) plus the embedding.
- A part's **share** is the covariance of its contribution with the readout across the 96 sentences, divided by the readout's variance. The shares, together with the embedding's, sum to 1.

### Ablation

- At A's tokens only, a set of parts is replaced by its mean output over all sentences' tokens (mean ablation). The choice's price slope is then measured again.
- The embedding is never ablated: that would erase the words themselves.

## 3. Pre-registered questions

| | Question | Measure | Pass |
|---|---|---|---|
| **Q1** | Do a few words set the price? | Per sentence, the two largest word-deletion effects on the price as a share of the summed absolute effects of all its words. Median over the 96. (Equal words would give about 0.2.) | median share ≥ 0.5 |
| **Q2** | Does the readout follow the choice word by word? | r between the change in readout and the change in price, over the 994 deletions | r ≥ 0.6 |
| **Q3** | Do a few parts write the price? | Summed share of the 10 parts with the largest shares (of the 820 heads and MLPs) | ≥ 0.5 |
| **Q4** | Does the choice use them? | Mean-ablating those 10 parts at A's tokens: the fraction of the choice's price slope lost. Compared with 5 random sets of 10 parts, matched in type and place, none of them among the 10: a head from the same block, or an MLP from a block within ±3 | lost ≥ 0.3, and more than all 5 random sets |

## 4. Also reported (not pre-registered)

- **The running price:** the readout token by token along each sentence. Where does it jump?
- **The top words:** each sentence's most important word, listed by value, and the words that matter most across all sentences.
- **Where the writers sit:**
  - the shares summed by block and by type (heads, MLPs), and the embedding's share;
  - the choice's price slope with all heads, or the MLP, of one block mean-ablated at A's tokens, for blocks 1-24 (4 partners per sentence).
- **What the top heads read:** for the heads among the top parts, the words they attend to from the sentence's tokens. Does a word's attention weight follow its deletion effect?

## 5. Running it

- One GPU pod (L40S, $1.09/h): about 20 minutes, about $0.40.
- About 1,100 passes for the readout and attribution, 8,000 for the occlusion, 5,400 for the Q4 ablations, and 18,000 for the block-by-block map.
- The pod pushes its results to `pod-results/pilot14` and stops itself.
- Predictions: Claude's sealed first, then Jasper's (`predictions/`).
