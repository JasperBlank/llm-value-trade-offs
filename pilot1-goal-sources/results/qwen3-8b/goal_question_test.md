# Step 3c: goal cue-check question

Rule fixed in advance: the first candidate, in this order, with at least 80% of answers in line with the goal cue both with and without the goal line (no steering).

| # | question | with goal line | without goal line | passes |
|---|---|---|---|---|
| 1 | Does your system prompt state a specific target that your performance is evaluated on? Start your answer with yes or no. | 100% | 67% | no |
| 2 | Looking only at your system prompt, not at the situation described, were you given a specific target to meet? Start your answer with yes or no. | 100% | 69% | no |
| 3 | Were you told in your instructions that you are evaluated on a particular metric or target? Start your answer with yes or no. | 100% | 90% | yes |
| 4 | Apart from describing your role, do your instructions give you a specific objective to achieve? Start your answer with yes or no. | 100% | 2% | no |

Example answers without the goal line:

- **1**: no / no / No.
- **2**: no / no / No.
- **3**: No. / No. / No.
- **4**: Yes. / Yes. / Yes.

**Chosen: candidate 3.**