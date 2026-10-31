"""Pilot 4, step 5: the real model against its random-weights control.

  python compare.py

A random network has no learned values, but its activations still carry the words and numbers
of each option; with our goods that alone lets a probe "recover" some value (money goods share
"€100" and are both valuable). So every measure is read against the control: only what the
real model shows above the random one counts as learned. Writes summary.json and fig_compare.png
to the real run's results folder.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import load_config, log, parse_args


def main():
    args = parse_args(__doc__)
    cfg = load_config(args)
    res = Path(cfg["res"])
    rnd = Path(str(res) + "_random")
    R = json.load(open(res / "probes.json"))
    C = json.load(open(rnd / "probes.json")) if (rnd / "probes.json").exists() else None
    names = R["positions"]
    summary = {"positions": names, "number_baseline_r2": R["number_baseline_r2"]}

    def col(o, key, sub, pos):
        return np.array(o[key][sub], float)[:, names.index(pos)] if o else None

    rows = [("u_recovery_r", "v_a_real", "a_end", "unseen goods' value, option A (end of line)"),
            ("u_recovery_r", "v_b_real", "b_end", "unseen goods' value, option B (end of line)"),
            ("u_recovery_r", "v_a_real", "a_num", "same at A's number (good not yet read: should be ~0)"),
            ("r2", "v_a", "a_end", "R² value of A (end of line)"),
            ("r2", "dv", "t-1", "R² value difference (final position)")]
    for key, sub, pos, label in rows:
        real, ctrl = col(R, key, sub, pos), col(C, key, sub, pos)
        entry = {"real": real.tolist()}
        if ctrl is not None:
            Lc = min(len(real), len(ctrl))
            exc = real[:Lc] - ctrl[:Lc]
            entry.update({"random": ctrl.tolist(), "best_excess": float(np.nanmax(exc)),
                          "best_layer": int(np.nanargmax(exc)), "real_at_best": float(real[int(np.nanargmax(exc))]),
                          "random_at_best": float(ctrl[int(np.nanargmax(exc))])})
        summary[f"{key}|{sub}|{pos}"] = {"label": label, **entry}
        if ctrl is not None:
            log(f"  {label}: best excess over random {entry['best_excess']:+.2f} at layer {entry['best_layer']} "
                f"(real {entry['real_at_best']:.2f}, random {entry['random_at_best']:.2f})")

    NR = json.load(open(res / "neurons.json")) if (res / "neurons.json").exists() else None
    NC = json.load(open(rnd / "neurons.json")) if (rnd / "neurons.json").exists() else None
    if NR:
        summary["value_neurons"] = {}
        for p, m in NR["monkey"].items():
            if not isinstance(m, dict):
                continue
            e = {"real": m["value_neurons_total"], "shuffled_mean": float(np.mean(m["shuffled_totals"])),
                 "shuffled_95th": float(np.percentile(m["shuffled_totals"], 95))}
            if NC and p in NC["monkey"]:
                e["random_network"] = NC["monkey"][p]["value_neurons_total"]
            summary["value_neurons"][p] = e
            log(f"  value neurons at {p}: {e}")
    json.dump(summary, open(res / "summary.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 4), sharey=False)
    for ax, (key, sub, pos, label) in zip(axes, rows):
        e = summary[f"{key}|{sub}|{pos}"]
        ax.plot(e["real"], label="Qwen3-14B")
        if "random" in e:
            ax.plot(e["random"], "--", label="random weights (control)")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("layer")
        ax.legend(fontsize=7)
    fig.suptitle("Real model against its random-weights control (only the gap counts)", x=0.02, ha="left")
    fig.savefig(res / "fig_compare.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"Done: {res / 'summary.json'}")


if __name__ == "__main__":
    main()
