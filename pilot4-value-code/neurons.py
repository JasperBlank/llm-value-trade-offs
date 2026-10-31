"""Pilot 4, step 3: MLP neurons, the Padoa-Schioppa way (from the running sums in neurons.npz).

  python neurons.py

For every neuron at every summarised position: its correlation with each variable, and, for
values, the partial correlation after removing the number of people (so a neuron that only
counts people is not called a value neuron). Each neuron is classified by its best variable
(|r| >= 0.3) and counted per layer. Writes results/<run>/neurons.json and fig_neurons.png.
"""
from __future__ import annotations

import json

import numpy as np

from common import ROOT, acts_path, load_config, log, parse_args

THRESH = 0.3          # |r| for a neuron to count as tracking a variable
PARTIAL = 0.2         # monkey criterion: partial correlation needed with the number AND with the good's value
NONE = 0.05           # ...and below this, a neuron counts as not tracking that variable
N_SHUFFLES = 50       # shuffle control for the value neurons


def main():
    args = parse_args(__doc__)
    cfg = load_config(args)
    z = np.load(acts_path(cfg, "neurons.npz"))
    n, s1, s2, sy, y = int(z["n"]), z["s1"], z["s2"], z["sy"], z["y"]
    var = [str(v) for v in z["vars"]]
    pos = [str(p) for p in z["positions"]]
    vi = {v: i for i, v in enumerate(var)}
    mean_a = s1 / n
    sd_a = np.sqrt(np.maximum(s2 / n - mean_a ** 2, 0))
    ym, ysd = y.mean(0), y.std(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (sy / n - mean_a[..., None] * ym) / (sd_a[..., None] * ysd)
    r = np.nan_to_num(r)
    ryy = np.corrcoef(y.T)

    def partial(v, q):
        """r(neuron, v | q) from the pairwise correlations."""
        a, b, c = r[..., vi[v]], r[..., vi[q]], ryy[vi[v], vi[q]]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (a - b * c) / np.sqrt((1 - b ** 2) * (1 - c ** 2))
        return np.nan_to_num(out)

    measures = {
        "offer value A (number removed)": partial("v_a", "log_qa"),
        "offer value B (number removed)": partial("v_b", "log_qb"),
        "number of people A": r[..., vi["log_qa"]],
        "number of people B": r[..., vi["log_qb"]],
        "chosen value": r[..., vi["chosen_value"]],
        "choice": r[..., vi["choice"]],
    }
    candidates = {
        "a_num": ["offer value A (number removed)", "number of people A"],
        "a_end": ["offer value A (number removed)", "number of people A"],
        "b_num": ["offer value B (number removed)", "number of people B", "offer value A (number removed)"],
        "b_end": ["offer value B (number removed)", "number of people B", "offer value A (number removed)"],
        "t-1": ["offer value A (number removed)", "offer value B (number removed)", "chosen value", "choice"],
    }
    n_layers = r.shape[0]
    alive = sd_a > 1e-6
    out = {"threshold": THRESH, "n_prompts": n, "n_layers": n_layers, "neurons_per_layer": int(r.shape[2]),
           "counts": {}, "top": {}}
    for pj, p in enumerate(pos):
        cands = candidates.get(p, [])
        if not cands:
            continue
        M = np.stack([np.abs(measures[c][:, pj]) for c in cands], -1)          # (layers, neurons, classes)
        best = M.argmax(-1)
        strong = (M.max(-1) >= THRESH) & alive[:, pj]
        out["counts"][p] = {c: [int(((best[li] == ci) & strong[li]).sum()) for li in range(n_layers)]
                            for ci, c in enumerate(cands)}
        out["top"][p] = {}
        for ci, c in enumerate(cands):
            m = np.where((best == ci) & strong, M[..., ci], 0)
            flat = np.argsort(m.ravel())[::-1][:20]
            out["top"][p][c] = [{"layer": int(f // m.shape[1]), "neuron": int(f % m.shape[1]),
                                 "r": float(measures[c][f // m.shape[1], pj, f % m.shape[1]])}
                                for f in flat if m.ravel()[f] > 0]
        log(f"  {p}: " + ", ".join(f"{c}: {sum(v)}" for c, v in out["counts"][p].items()))
    # The monkey criterion at the option positions, over prompts whose option is a real good.
    # Each neuron's activity is related to log(number of people) and to the good's value u
    # together (partial correlations from the per-good running sums). A value neuron needs both:
    # it follows the number of people with the good held fixed, AND the good's value with the
    # number held fixed, in the same direction (value = u + b * log(number)). A neuron that only
    # recognises goods fails the first; one that only counts people fails the second.
    u = json.load(open(ROOT / "values.json"))["u"]
    keys = [str(k) for k in z["keys"]]
    real = np.array([not k.startswith("N_") for k in keys])
    uk = np.array([u[k] for k in keys])
    KS = z["key_sums"].astype(np.float64)                          # (layers, option positions, neurons, keys, 3)
    out["monkey"] = {"partial_r_threshold": PARTIAL, "none_threshold": NONE}
    for oj, p in enumerate([str(x) for x in z["key_positions"]]):
        st = z["key_stats_" + p[0]][real]                          # per real key: n, sum log q, sum log q^2
        nk, sq, sqq, ur = st[:, 0], st[:, 1], st[:, 2], uk[real]
        sa, saa, saq = (KS[:, oj][:, :, real, j] for j in range(3))     # each (layers, neurons, real keys)
        n = nk.sum()
        b_val = json.load(open(ROOT / "values.json"))["b"]

        def classify(ur):
            Sa, Saa, Saq, Sau = sa.sum(-1), saa.sum(-1), saq.sum(-1), (sa * ur).sum(-1)
            Sq, Sqq, Su, Suu, Squ = sq.sum(), sqq.sum(), (nk * ur).sum(), (nk * ur ** 2).sum(), (sq * ur).sum()
            Caa, Caq, Cau = Saa - Sa ** 2 / n, Saq - Sa * Sq / n, Sau - Sa * Su / n
            Cqq, Cuu, Cqu = Sqq - Sq ** 2 / n, Suu - Su ** 2 / n, Squ - Sq * Su / n
            with np.errstate(divide="ignore", invalid="ignore"):
                r_aq, r_au, r_qu = Caq / np.sqrt(Caa * Cqq), Cau / np.sqrt(Caa * Cuu), Cqu / np.sqrt(Cqq * Cuu)
                r_q = np.nan_to_num((r_aq - r_au * r_qu) / np.sqrt((1 - r_au ** 2) * (1 - r_qu ** 2)))   # number, good fixed
                r_u = np.nan_to_num((r_au - r_aq * r_qu) / np.sqrt((1 - r_aq ** 2) * (1 - r_qu ** 2)))   # value, number fixed
                det = Cqq * Cuu - Cqu ** 2                                 # two-regressor least squares
                beta_q = (Cuu * Caq - Cqu * Cau) / det
                beta_u = (Cqq * Cau - Cqu * Caq) / det
                ratio = np.nan_to_num(beta_q / beta_u)                     # a value neuron: about b
            value = ((np.abs(r_q) >= PARTIAL) & (np.abs(r_u) >= PARTIAL) & (np.sign(r_q) == np.sign(r_u))
                     & (ratio >= b_val / 2) & (ratio <= b_val * 2))
            return r_q, r_u, ratio, value

        r_q, r_u, ratio, value = classify(ur)
        number = (np.abs(r_q) >= PARTIAL) & (np.abs(r_u) < NONE)
        identity = (np.abs(r_u) >= PARTIAL) & (np.abs(r_q) < NONE)
        # Shuffle control: the goods' values permuted among the real goods; how many pass by chance?
        rng = np.random.default_rng(5)
        null = []
        for _ in range(N_SHUFFLES):
            perm = rng.permutation(len(ur))
            while abs(np.corrcoef(ur[perm], ur)[0, 1]) > 0.3:     # shuffles unrelated to the true order only
                perm = rng.permutation(len(ur))
            null.append(classify(ur[perm])[3].sum(1))
        null = np.array(null)                                              # (shuffles, layers)
        score = np.minimum(np.abs(r_q), np.abs(r_u)) * value
        flat = np.argsort(score.ravel())[::-1][:20]
        out["monkey"][p] = {
            "value_neurons": value.sum(1).tolist(), "number_only": number.sum(1).tolist(),
            "good_value_only": identity.sum(1).tolist(),
            "value_neurons_shuffled_mean": null.mean(0).tolist(), "value_neurons_shuffled_95th": np.percentile(null, 95, axis=0).tolist(),
            "value_neurons_total": int(value.sum()), "shuffled_totals": null.sum(1).tolist(),
            "top_value_neurons": [{"layer": int(f // value.shape[1]), "neuron": int(f % value.shape[1]),
                                   "r_number_good_fixed": float(r_q.ravel()[f]), "r_value_number_fixed": float(r_u.ravel()[f]),
                                   "number_to_value_ratio": float(ratio.ravel()[f])}
                                  for f in flat if value.ravel()[f]]}
        log(f"  monkey criterion at {p}: value neurons {int(value.sum())} (shuffled goods: {null.sum(1).mean():.1f} on average, "
            f"95th percentile {np.percentile(null.sum(1), 95):.0f}), number only {int(number.sum())}, good's value only {int(identity.sum())}")
    json.dump(out, open(f"{cfg['res']}/neurons.json", "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps = list(out["counts"])
    fig, axes = plt.subplots(2, len(ps), figsize=(4 * len(ps), 7.5), sharey="row")
    for ax, p in zip(axes[0], ps):
        for c, v in out["counts"][p].items():
            ax.plot(v, label=c)
        ax.set_title(f"position {p}", fontsize=10)
        ax.legend(fontsize=6)
    axes[0][0].set_ylabel(f"neurons with |r| ≥ {THRESH}, by best variable")
    for ax, p in zip(axes[1], ps):
        if p in out["monkey"]:
            m = out["monkey"][p]
            ax.plot(m["value_neurons"], label="value (number AND good's value, right mix)")
            ax.plot(m["value_neurons_shuffled_95th"], ":", color="grey", label="same, goods' values shuffled (95th pct)")
            ax.plot(m["number_only"], label="number of people only")
            ax.plot(m["good_value_only"], label="good's value only (not the number)")
            ax.legend(fontsize=6)
        ax.set_xlabel("layer")
    axes[1][0].set_ylabel("neurons, monkey criterion")
    fig.suptitle("Neurons classified the monkey way", x=0.02, ha="left")
    fig.savefig(f"{cfg['res']}/fig_neurons.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"Done: {cfg['res']}/neurons.json")


if __name__ == "__main__":
    main()
