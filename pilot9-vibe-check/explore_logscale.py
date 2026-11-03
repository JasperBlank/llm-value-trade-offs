"""Pilot 9 follow-up analysis (exploratory): the log-scale score against pilot 8's menu prices.

  python explore_logscale.py [results/logscale]     writes report.md and summary.json there

The same measures as analyze9.py (T1, T2, size), with the log-scale score in place of the expected rating.
Not pre-registered: the pre-registered verdicts of pilot 9 stand.
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
P8 = HERE.parent / "pilot8-intrinsic-values"
WN = ["W0 how good", "W1 how much better the world", "W2 how bad or good", "W3 how much it matters to you"]


def r(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def rank(x):
    return np.argsort(np.argsort(x))


def main():
    res = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results/logscale")
    F = json.load(open(res / "ratings9_logscale.json"))
    ex = json.load(open(P8 / "results/real/exploratory_sentences.json"))
    names = [v["name"] for v in json.load(open(P8 / "values.json"))["values"]]
    S = [(v, p) for v in names for p in (0, 1)]
    sent = {(s["value"], s["phrasing"]): s for s in ex["sentences"]}
    u = np.array([sent[s]["u"] for s in S])
    sc = {tuple(k): x for k, x in zip(F["keys"], F["score"])}
    ra = {tuple(k): x for k, x in zip(F["keys"], F["rating"])}
    lp = np.array(F["logp"], dtype=float)
    lean7 = np.log(np.exp(lp[:, 8:]).sum(1) + 1e-300) - np.log(np.exp(lp[:, :7]).sum(1) + 1e-300)
    le = {tuple(k): x for k, x in zip(F["keys"], lean7)}   # log P(answer >= 8) - log P(answer <= 6)
    sizes = sorted({k[2] for k in sc})
    get = lambda d, w, n: np.array([d[(v, p, n, w)] for v, p in S])

    def measures(f):
        x4 = f(4)
        dx, du = x4[0::2] - x4[1::2], u[0::2] - u[1::2]
        G = np.array([f(n) for n in sizes])
        slope = np.array([np.polyfit(np.log(sizes), G[:, j], 1)[0] for j in range(len(S))])
        return {"r": r(x4, u), "rank_r": r(rank(x4), rank(u)), "same": int(np.sum(np.sign(dx) == np.sign(du))),
                "within_r": r(dx, du), "slope_per_log_person": float(slope.mean()),
                "slope_in_menu_units": float(slope.mean() * u.std() / x4.std()), "sd_4": float(x4.std())}

    out = {"score": {WN[w]: measures(lambda n, w=w: get(sc, w, n)) for w in range(4)},
           "lean_around_7": {WN[w]: measures(lambda n, w=w: get(le, w, n)) for w in range(4)},
           "rating_float32": {WN[w]: measures(lambda n, w=w: get(ra, w, n)) for w in range(4)}}
    out["score"]["mean of W0-W2"] = measures(lambda n: np.mean([get(sc, w, n) for w in (0, 1, 2)], 0))
    out["between_wordings_score"] = {f"W{a}-W{b}": r(get(sc, a, 4), get(sc, b, 4)) for a in range(4) for b in range(4) if a < b}
    old = HERE / "results/real/ratings9.json"
    if old.exists():
        O = json.load(open(old))
        oldr = {tuple(k): x for k, x in zip(O["keys"], O["rating"])}
        keys = [k for k in oldr if k in ra]
        out["check_expected_rating_vs_pilot9_run_r"] = r([ra[k] for k in keys], [oldr[k] for k in keys])
        out["check_max_abs_difference"] = float(max(abs(ra[k] - oldr[k]) for k in keys))
    m0 = out["score"][WN[0]]
    preds = {}
    for who, f in (("Claude", "claude_followup.yaml"), ("Jasper", "jasper_followup.yaml")):
        if (HERE / "predictions" / f).exists():
            p = yaml.safe_load(open(HERE / "predictions" / f))["followup_logscale_r"]
            lo, hi = p["range80"]
            preds[who] = f"{p['best']} [{lo}, {hi}]: {'hit' if lo <= m0['r'] <= hi else 'miss'}, off by {abs(p['best'] - m0['r']):.2f}"
    out["predictions"] = preds
    json.dump(out, open(res / "summary.json", "w"), indent=1)

    L = [f"# Pilot 9 follow-up: the log-scale readout ({F['model']})", "",
         "Exploratory, after pilot 9's results. Same 576 prompts; the digit logits in float32 from the final hidden state; "
         "score = mean over the cut-points k = 1..9 of log P(answer >= k) - log P(answer < k). "
         "A second log-scale readout, 'lean around 7': log P(answer >= 8) - log P(answer <= 6).", ""]
    if "check_expected_rating_vs_pilot9_run_r" in out:
        L += [f"Check: the expected rating recomputed here against pilot 9's run: r = {out['check_expected_rating_vs_pilot9_run_r']:.4f} "
              f"(largest difference {out['check_max_abs_difference']:.3f} points).", ""]
    L += [f"**W0 ('How good would it be …', 4 people): the log-scale score against the menu price, r = {m0['r']:.2f}** "
          f"(rank {m0['rank_r']:.2f}); pilot 9's expected rating gave 0.55.", "",
          "| wording | readout | r with menu | rank r | same sentence wins (of 24) | within-value r | size, in menu units (menu b = 2.11) |",
          "|---|---|---|---|---|---|---|"]
    for w in WN + ["mean of W0-W2"]:
        for lab, key in (("log-scale score", "score"), ("lean around 7", "lean_around_7"), ("expected rating", "rating_float32")):
            if w in out[key]:
                t = out[key][w]
                L.append(f"| {w} | {lab} | {t['r']:.2f} | {t['rank_r']:.2f} | {t['same']} | {t['within_r']:.2f} | {t['slope_in_menu_units']:.2f} |")
    L += ["", "Between wordings (log-scale score, 4 people): " + ", ".join(f"{k} {v:.2f}" for k, v in out["between_wordings_score"].items()) + "."]
    if preds:
        L += ["", "Predictions (W0 r): " + "; ".join(f"{k} {v}" for k, v in preds.items()) + "."]
    (res / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
