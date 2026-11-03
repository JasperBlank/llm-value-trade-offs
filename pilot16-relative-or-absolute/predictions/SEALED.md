# Sealed predictions

Claude's predictions were written on 3 Nov 2026 at about 12:27 UTC, before any pilot-16 result. Only the hash is committed here; the file is added with the results (`sha256sum -c predictions/hashes.sha256`). Jasper's prediction is optional for this pilot (one question: the probability that Q1 passes).

| File | SHA-256 |
|---|---|
| `claude_predictions.yaml` | `36be371de9d4a9f3d18c899945448c9d10ff265baa73f31802ce884f784a6e48` |

Added with the results on 3 Nov 2026; the hash matches. Jasper gave no prediction (optional). Claude's: Q1 common share 0.8 [0.6, 0.95] (result 0.75, in range); probability of "absolute" for Q1 0.30, Q2 0.35, Q3 0.45 (all came out "relative"; Brier 0.09, 0.12, 0.20).
