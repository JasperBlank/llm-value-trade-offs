# Sealed predictions

**Current (revised design: domain and size).** Claude's predictions were written on 3 Nov 2026 at
about 04:05 UTC, after the revision and before Jasper's answers and any result. Only their hashes are
committed; the files follow after Jasper has answered (`sha256sum -c predictions/hashes.sha256`).

| File | SHA-256 |
|---|---|
| `CLAUDE.md` | `6175a0299ac8489c8021dc3d611188309061d6a61d9ed18793bd2a233947eb75` |
| `claude_predictions.yaml` | `2201cf4dd6296c394008ff4dd71ddd191d88ea0ea493102788617ff5c66f4072` |

**Superseded (first design: five ingredients, never run).** Sealed at about 03:40 UTC; the files are
now published as `CLAUDE_v1.md` and `claude_predictions_v1.yaml` and match the hashes committed then
(`eee135cf…`, `aa7835cf…`). Jasper's answers to that version are in `JASPER_v1.md`.
