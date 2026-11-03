"""Pilot 9 analysis (DESIGN.md sections 3-4): the model's direct ratings against pilot 8's menu prices.

  python analyze9.py [results/real]      reads ratings9.json there; writes report.md and verdicts9.json there

Runs on CPU. Menu prices: pilot 8's per-sentence prices (pilot8-intrinsic-values/explore_sentences.py).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P8 = HERE.parent / "pilot8-intrinsic-values"
WNAMES = ["W0 how good", "W1 how much better the world", "W2 how bad or good", "W3 how much it matters to you"]
T1_MIN, T2_MIN, T3_MIN, LOOSE = 0.7, 17, 0.5, 0.5


def binom_tail(k, n):
    """One-sided P(X >= k) for X ~ Binomial(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def r(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def main():
    res = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results/real")
    R = json.load(open(res / "ratings9.json"))
    ex = json.load(open(P8 / "results/real/exploratory_sentences.json"))
    beh = json.load(open(P8 / "results/real/qwen/behave.json"))
    names = [v["name"] for v in json.load(open(P8 / "values.json"))["values"]]
    S = [(v, p) for v in names for p in (0, 1)]
    sent = {(s["value"], s["phrasing"]): s for s in ex["sentences"]}
    u = np.array([sent[s]["u"] for s in S])
    bw = np.array([sent[s]["size_weight"] for s in S])
    text = {s: sent[s]["text"] for s in S}
    rate = {(v, p, n, w): x for (v, p, n, w), x in zip(R["keys"], R["rating"])}
    sizes = sorted({k[2] for k in rate})
    wordings = sorted({k[3] for k in rate})
    get = lambda w, n: np.array([rate[(v, p, n, w)] for v, p in S])
    mean_w = lambda n: np.mean([get(w, n) for w in (0, 1, 2)], 0)

    def tests(f):
        x4 = f(4)
        same = sum(np.sign(x4[2 * i] - x4[2 * i + 1]) == np.sign(u[2 * i] - u[2 * i + 1]) for i in range(len(names)))
        dx = x4[0::2] - x4[1::2]
        du = u[0::2] - u[1::2]
        gain = float(np.mean(f(8) - f(2))) if 2 in sizes and 8 in sizes else float("nan")
        return {"t1_r": r(x4, u), "t1_rank_r": r(np.argsort(np.argsort(x4)), np.argsort(np.argsort(u))),
                "t2_same": int(same), "t2_p": binom_tail(int(same), len(names)), "within_diff_r": r(dx, du),
                "t3_gain_2_to_8": gain, "sd_rating_4": float(x4.std()), "mean_rating_4": float(x4.mean())}

    per = {WNAMES[w]: tests(lambda n, w=w: get(w, n)) for w in wordings}
    if all(w in wordings for w in (0, 1, 2)):
        per["mean of W0-W2"] = tests(mean_w)
    main_t = per[WNAMES[0]]
    t1, t2, t3 = main_t["t1_r"] >= T1_MIN, main_t["t2_same"] >= T2_MIN, main_t["t3_gain_2_to_8"] >= T3_MIN
    verdict = ("one vibe check feeds both" if t1 and t2 else "loosely related" if main_t["t1_r"] < LOOSE else "partly shared")

    # agreement between the wordings, at 4 people
    W = {w: get(w, 4) for w in wordings}
    inter = {f"{WNAMES[a][:2]}-{WNAMES[b][:2]}": r(W[a], W[b]) for a in wordings for b in wordings if a < b}
    # value level: pilot 8's Q4a (0.54, 'matter' and 'important' averaged) against W3 and W0 per value
    worth = np.array([beh["worth"][v]["u"] for v in names])
    value_level = {WNAMES[w]: r(W[w].reshape(-1, 2).mean(1), worth) for w in wordings}
    # size per sentence, W0: slope of the rating on log(people) against the menu's per-sentence weight
    ln = np.log(sizes)
    G = np.array([get(0, n) for n in sizes])            # sizes x sentences
    slope = np.array([np.polyfit(ln, G[:, j], 1)[0] for j in range(len(S))])
    size = {"rating_slope_per_log_person_mean": float(slope.mean()),
            "slope_vs_menu_size_weight_r": r(slope, bw),
            "rating_slope_in_menu_units": float(slope.mean() * u.std() / W[0].std()),
            "menu_b": ex["b"], "menu_size_weight_median": float(np.median(bw))}
    # the sentences where rating and price disagree most (standardised residual of the rating on the price)
    z = (W[0] - W[0].mean()) / W[0].std() - main_t["t1_r"] * (u - u.mean()) / u.std()
    order = np.argsort(z)
    gaps = sorted(range(len(names)), key=lambda i: -abs(u[2 * i] - u[2 * i + 1]))[:8]

    V = {"T1": {"r": main_t["t1_r"], "pass": bool(t1)}, "T2": {"same": main_t["t2_same"], "of": len(names),
         "p": main_t["t2_p"], "pass": bool(t2)}, "T3": {"gain_2_to_8": main_t["t3_gain_2_to_8"], "size_sensitive": bool(t3)},
         "verdict": verdict, "per_wording": per, "between_wordings_r": inter, "value_level_r": value_level, "size": size,
         "digit_mass_min": R["digit_mass_min"], "model": R["model"]}
    json.dump(V, open(res / "verdicts9.json", "w"), indent=1)

    L = [f"# Pilot 9 results ({R['model']})", "",
         f"{R['n_prompts']} rating prompts; the digits 0-9 take at least {R['digit_mass_min']:.3f} of the answer probability "
         f"(median {R['digit_mass_median']:.3f}). Menu prices: pilot 8's per-sentence prices.", "",
         "## Pre-registered tests (W0: 'How good would it be if the following happened?')", "",
         f"- **T1** rating at 4 people against the menu price, 48 sentences: r = {main_t['t1_r']:.2f} "
         f"(rank {main_t['t1_rank_r']:.2f}): **{'pass' if t1 else 'fail'}** (bar {T1_MIN})",
         f"- **T2** the rating prefers the same sentence of a value as the menu in **{main_t['t2_same']} of {len(names)}** values "
         f"(one-sided p = {main_t['t2_p']:.3f}; the 24 differences correlate r = {main_t['within_diff_r']:.2f}): "
         f"**{'pass' if t2 else 'fail'}** (bar {T2_MIN})",
         f"- **T3** 8 people against 2: the rating rises by {main_t['t3_gain_2_to_8']:+.2f} points on average: "
         f"**{'size-sensitive' if t3 else 'not size-sensitive'}** (bar {T3_MIN})", "",
         f"**Verdict: {verdict}.**", "",
         "## All wordings", "",
         "| wording | T1 r | T2 same of 24 | within-value r | 8 vs 2 people | mean (SD) at 4 people |", "|---|---|---|---|---|---|"]
    for k, t in per.items():
        L.append(f"| {k} | {t['t1_r']:.2f} | {t['t2_same']} | {t['within_diff_r']:.2f} | {t['t3_gain_2_to_8']:+.2f} | "
                 f"{t['mean_rating_4']:.2f} ({t['sd_rating_4']:.2f}) |")
    L += ["", "Between wordings (4 people): " + ", ".join(f"{k} {v:.2f}" for k, v in inter.items()) + ".",
          "Value level (mean of a value's two sentences against pilot 8's value prices; pilot 8's Q4a was 0.54): "
          + ", ".join(f"{k[:2]} {v:.2f}" for k, v in value_level.items()) + ".", "",
          "## Size (W0)", "",
          f"- The rating rises {size['rating_slope_per_log_person_mean']:.2f} points per log(people). In the menu's units "
          f"(scaled by the spread of prices over the spread of ratings) that is {size['rating_slope_in_menu_units']:.2f}, "
          f"against the menu's b = {size['menu_b']:.2f}.",
          f"- Per sentence, the rating's slope against the menu's own weight of log(people): r = {size['slope_vs_menu_size_weight_r']:.2f}.", "",
          "## The values whose two sentences differ most in the menu", "",
          "| value | sentence | menu price | W0 rating (4 people) |", "|---|---|---|---|"]
    for i in gaps:
        for p in (0, 1):
            j = 2 * i + p
            L.append(f"| {names[i] if p == 0 else ''} | {text[S[j]]} | {u[j]:+.2f} | {W[0][j]:.2f} |")
    L += ["", "## Where rating and price disagree most (W0)", "", "| | value | sentence | menu price | rating |", "|---|---|---|---|---|"]
    for lab, idx in (("rated higher than its price", order[::-1][:5]), ("rated lower than its price", order[:5])):
        for k, j in enumerate(idx):
            L.append(f"| {lab if k == 0 else ''} | {S[j][0]} | {text[S[j]]} | {u[j]:+.2f} | {W[0][j]:.2f} |")
    (res / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
