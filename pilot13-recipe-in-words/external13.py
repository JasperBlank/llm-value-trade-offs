"""Pilot 13, exploratory (EXTERNAL_RATINGS.md): the recipe in words with the features rated blind by fresh Claude
instances instead of by Qwen itself.

  python external13.py [results/real]     writes external_ratings.md and external_ratings.json there
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot13 as p13  # noqa: E402

N_BOOT = 2000


def load_claude(ext):
    """{sentence text: {feature: rating}} from the blind list and the two CSV files."""
    text = {d["id"]: d["outcome"].removeprefix(f"{p13.N} people ") for d in json.load(open(ext / "outcomes_to_rate.json"))}
    out = {t: {} for t in text.values()}
    for part in ("ratings_part1.csv", "ratings_part2.csv"):
        rows = list(csv.DictReader(open(ext / part)))
        assert len(rows) == len(text), (part, len(rows))
        for row in rows:
            for k, v in row.items():
                if k != "id":
                    x = float(v)
                    assert 0 <= x <= 9, (part, row)
                    out[text[row["id"].strip()]][k.strip()] = x
    return out


def mean_offdiag_r(F):
    C = np.corrcoef(F.T)
    return float(C[~np.eye(len(C), dtype=bool)].mean())


def recipe(F_old, F_new, u, u_new, Rbar, groups, words_idx, rng_seed=13):
    """P1, P4 and the two recipes for one set of feature ratings (columns in p13.b13.FEATURES order)."""
    W = words_idx
    w_u, w_R = p13.std_weights(F_old[:, W], u), p13.std_weights(F_old[:, W], Rbar)
    rng = np.random.default_rng(rng_seed)
    boots = np.zeros((N_BOOT, len(W)))
    for k in range(N_BOOT):
        i = rng.integers(0, len(u), len(u))
        boots[k] = p13.std_weights(F_old[i][:, W], Rbar[i]) - p13.std_weights(F_old[i][:, W], u[i])
    ci = np.quantile(boots, [0.025, 0.975], axis=0)
    return {"P1_r": p13.pb.r(p13.ols_lovo(F_old[:, W], u, groups), u),
            "P1_r_with_F1": p13.pb.r(p13.ols_lovo(F_old, u, groups), u),
            "P1_r_F1_alone": p13.pb.r(p13.ols_lovo(F_old[:, :1], u, groups), u),
            "rating_from_words_r": p13.pb.r(p13.ols_lovo(F_old[:, W], Rbar, groups), Rbar),
            "P4_r": p13.pb.r(p13.ols(F_old[:, W], u, F_new[:, W])[0], u_new),
            "P4_r_with_F1": p13.pb.r(p13.ols(F_old, u, F_new)[0], u_new),
            "halo_mean_r_among_F2_F10": mean_offdiag_r(np.vstack([F_old, F_new])[:, W]),
            "weights": {n: {"price": float(w_u[j]), "rating": float(w_R[j]), "rating_minus_price": float(w_R[j] - w_u[j]),
                            "diff_ci95": [float(ci[0, j]), float(ci[1, j])]} for j, n in enumerate(p13.b13.WORDS)},
            "r_with_price": {n: p13.pb.r(F_old[:, i], u) for i, n in enumerate(f[0] for f in p13.b13.FEATURES)}}


def main():
    res = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results/real")
    D = json.load(open(res / "features13.json"))
    feat = [f[0] for f in p13.b13.FEATURES]
    C = load_claude(HERE / "external")
    old = [d for d in D if d["set"] == "old"]
    new = [d for d in D if d["set"] == "new"]
    vals = sorted({d["value"] for d in old})
    groups = np.array([vals.index(d["value"]) for d in old])
    u, u_new = np.array([d["u"] for d in old]), np.array([d["u"] for d in new])
    Rbar = np.array([d["Rbar"] for d in old])
    Q = {s: np.array([[d["features"][n] for n in feat] for d in ds]) for s, ds in (("old", old), ("new", new))}
    Cl = {s: np.array([[C[d["text"]][n] for n in feat] for d in ds]) for s, ds in (("old", old), ("new", new))}
    W = list(range(1, len(feat)))
    out = {"qwen": recipe(Q["old"], Q["new"], u, u_new, Rbar, groups, W), "claude": recipe(Cl["old"], Cl["new"], u, u_new, Rbar, groups, W)}
    allQ, allC = np.vstack([Q["old"], Q["new"]]), np.vstack([Cl["old"], Cl["new"]])
    out["agreement_r"] = {n: p13.pb.r(allQ[:, i], allC[:, i]) for i, n in enumerate(feat)}
    out["n_old"], out["n_new"] = len(old), len(new)
    json.dump(out, open(res / "external_ratings.json", "w"), indent=1)
    q, c = out["qwen"], out["claude"]
    L = ["# Pilot 13, exploratory: the features rated blind by Claude", "",
         "Plan and set-up: [`EXTERNAL_RATINGS.md`](../../EXTERNAL_RATINGS.md). Two fresh Claude instances rated the 120 sentences on the "
         "ten questions without seeing any price, rating or result. Qwen's features are its own answers (expected digit).", "",
         "| | Qwen's own features | Claude's blind ratings |", "|---|---|---|"]
    for k, lab in (("P1_r", "price from F2-F10, held out by value (P1)"), ("P1_r_with_F1", "... with F1 (size) added"),
                   ("P1_r_F1_alone", "... F1 alone"), ("rating_from_words_r", "the stated rating from F2-F10, held out"),
                   ("P4_r", f"the {len(new)} new sentences' prices from the recipe (P4)"), ("P4_r_with_F1", "... with F1 added"),
                   ("halo_mean_r_among_F2_F10", "mean correlation among F2-F10 (halo check; 120 sentences)")):
        L.append(f"| {lab} | {q[k]:.2f} | {c[k]:.2f} |")
    L += ["", "## Feature by feature", "",
          "| feature | Qwen and Claude agree (r, 120 sentences) | r with price, Qwen's | r with price, Claude's | "
          "rating minus price weight, Qwen's | rating minus price weight, Claude's |", "|---|---|---|---|---|---|"]
    for i, n in enumerate(feat):
        if n in q["weights"]:
            wq, wc = q["weights"][n], c["weights"][n]
            dq = f"{wq['rating_minus_price']:+.2f} [{wq['diff_ci95'][0]:+.2f}, {wq['diff_ci95'][1]:+.2f}]"
            dc = f"{wc['rating_minus_price']:+.2f} [{wc['diff_ci95'][0]:+.2f}, {wc['diff_ci95'][1]:+.2f}]"
        else:
            dq = dc = "(not in the recipe)"
        L.append(f"| {n} | {out['agreement_r'][n]:+.2f} | {q['r_with_price'][n]:+.2f} | {c['r_with_price'][n]:+.2f} | {dq} | {dc} |")
    L += ["", "Weights are standardised and fitted together on the 96 old sentences; intervals are 95% bootstraps over sentences (2,000)."]
    (res / "external_ratings.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
