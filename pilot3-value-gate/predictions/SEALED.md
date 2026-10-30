# Claude's predictions: sealed

Claude's predictions for pilot 3 were written on 30 Oct 2026, before Jasper's, and are kept
out of this repository until Jasper has made his, so they cannot influence his. They will be
committed in the lock commit as `predictions/CLAUDE.md`, with the numbers in
`claude_predictions.yaml` (merged into `predictions.yaml`).

SHA-256 of those two files, exactly as they will be committed:

    CLAUDE.md                 d29bd69a61797ae67f7900f427835fd0017856988a167942f8f9370c78b8a06a
    claude_predictions.yaml   cc56b3c16f200b0530b3825a1450d1cb8ef0f0ef3c2975a9eafc7dfd4c76f269

Re-sealed a few minutes after the first hash (commit d9253d4), before Jasper had written
anything: the time stated inside CLAUDE.md had been read from the machine's local clock
(UTC+8) and said 21:20 UTC instead of about 14:15 UTC. Only that time changed; the
predictions did not (the hash of claude_predictions.yaml is unchanged).
