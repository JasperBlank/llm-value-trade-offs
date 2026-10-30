# Reading guide

Six pieces, about three to four hours in total. For each: what to read, and the questions
to answer in your notes. The answers feed straight into the pre-registration and the
"Related work" section of the post.

---

### 1. The Assistant Axis (Lu, Gallagher, Michala, Fish, Lindsey; ICML 2026), ~45 min
https://arxiv.org/abs/2601.10387

**Read:** abstract, introduction, section 2 (how role vectors and the axis are extracted),
the steering results, and the activation-capping results.

**Answer:**
- Which layers did they use, and how did they choose them?
- How does their axis (first principal component over ~300 roles) differ from our
  difference of means over 20 roles? What could that difference change?
- What happened when they steered away from the Assistant? (Our manipulation check
  should see something similar.)
- Which conversation types caused natural drift? Do any of our scenarios resemble them?

### 2. Agentic Misalignment (Lynch et al., Anthropic, 2025), ~40 min
https://arxiv.org/abs/2510.05179

**Read:** the introduction, section 4 (goal conflict and replacement threat tested
separately), and the experiment where the model first judges whether it is in a test.

**Answer:**
- How did they vary the goal and the threat independently? How is our goal factor
  similar, and how is it different?
- What did they find when the model believed the situation was real versus a test?
  What does that predict for our monitoring factor?
- Which limitations do they list for artificial scenarios? Most apply to ours too.

### 3. Asymmetric Goal Drift in Coding Agents Under Value Conflict (Saebo et al., 2026), ~30 min
https://arxiv.org/abs/2603.03456

**Read:** introduction, the experimental framework, the main results.

**Answer:**
- When did agents break their system prompt, and which values did they break it for?
- Which three factors do they name as drivers of drift? Which one does our design
  leave out (accumulated context)?
- This is the closest behavioural study to ours. What does our design add? (Steering the
  character directly, and the monitoring factor.)

### 4. The Persona Selection Model (Marks, Lindsey, Olah; Anthropic, 2026), ~30 min
https://alignment.anthropic.com/2026/psm

**Read:** the whole post.

**Answer:**
- State the theory in two sentences, in your own words.
- What exactly does it predict for our "away" condition?
- Which open questions do the authors list? Does our pilot touch any of them?

### 5. Generative Value Conflicts Reveal LLM Priorities: ConflictScope (Liu et al., ICLR 2026), ~30 min
https://arxiv.org/abs/2509.25369

**Read:** abstract, the comparison of multiple-choice and open-ended evaluation, and the
system-prompt steering result.

**Answer:**
- How did value rankings change between multiple-choice and open-ended settings? This is
  the main threat to our forced A/B design; you'll need it for the limitations.
- How much did writing value orderings into the system prompt move behaviour? Compare
  with what you predict for our goal factor.

### 6. Reproducing steering against evaluation awareness in a large open-weight model (LessWrong, 2026), ~25 min
https://www.lesswrong.com/posts/HhF5kESdtPHku7kim/reproducing-steering-against-evaluation-awareness-in-a-large-1

**Read:** the whole post, especially the results for the "control" steering vectors.

**Answer:**
- What did control vectors (unrelated content) do to misbehaviour rates?
- What did they find about steering strength (non-monotonic effects)?
- What does this say about how strongly to trust a persona effect that beats only one
  random direction?

---

### If you have time
- **Persona vectors** (Chen et al., Anthropic, 2025), https://arxiv.org/abs/2507.21509: the
  difference-of-means method our direction uses, applied to traits like sycophancy.
- **The Hot Mess of AI** (Hägele et al., ICLR 2026), https://alignment.anthropic.com/2026/hot-mess-of-ai:
  the bias–variance framing from earlier. Our design measures bias (which way choices
  move), not variance; worth one sentence in the post.
