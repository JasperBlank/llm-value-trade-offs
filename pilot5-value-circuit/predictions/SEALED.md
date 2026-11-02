# Sealed predictions

Claude's predictions were written on 2 Nov 2026 at about 07:44 UTC, before Jasper's and before any
pilot-5 result. Only their SHA-256 hashes are committed here; the files are added after Jasper
has answered, and can be checked with `sha256sum`.

| File | SHA-256 |
|---|---|
| `CLAUDE.md` | `6390570a81817f799c249b2bee0281f561b2e75a6974813045cf68cc6b6869af` |
| `claude_predictions.yaml` | `65cfced8434c214eb90f37a0b548616d9bb45f82f5f1e70dc5153537f8ded00a` |
| `CLAUDE_formats.md` (extension, sealed 2 Nov ~09:43 UTC, before any result was seen) | `44a0dcf2d5a1aa6e93ec2cff4e8da9f2b392bfbe3835d7a505d7597fbb30c934` |
| `claude_formats.yaml` | `b1d725123f79703be987480d7f8caa93b86cce3e04b93d260988b123a2fdf33e` |

The main run (everything except the formats extension) ran on 2 Nov from 09:23 to 09:36 UTC on
an RTX PRO 6000 (Runpod, US-NE-1), after Claude's main predictions were sealed and before Jasper's;
its outputs stay hidden on the pod (the log shows progress only) until both sets are committed.
