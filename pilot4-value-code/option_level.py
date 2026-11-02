"""Pilot 4, read after the run: the unseen-goods test at the level of options.

  python option_level.py [--results results/real]

The prompt-level measure in probes.json (u_recovery_r) mixes two things: whether the probe
knows how valuable a held-out good is, and how it handles the number of people within a good.
The question that matters is the first, and it has only as many data points as there are
options (9 real options in pilot 3's design). This script takes, per layer and position, the
probe's held-out estimate for each option (its prediction with the number of people removed,
stored in probes.json under "u_recovery") and correlates the 9 estimates with the 9 true
values. Real model and random-weights control side by side. Writes option_level.md next to
the real run's results.

Added on 2 Nov 2026, after the results came in (not part of the planned battery).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def option_corr(est: dict, u: dict) -> tuple[float, int]:
    ks = [k for k in est if not k.startswith("N_") and est[k] is not None and np.isfinite(est[k])]
    if len(ks) < 4:
        return float("nan"), len(ks)
    x = np.array([u[k] for k in ks])
    y = np.array([est[k] for k in ks])
    if y.std() == 0:
        return float("nan"), len(ks)
    return float(np.corrcoef(x, y)[0, 1]), len(ks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results" / "real"))
    a = ap.parse_args()
    res = Path(a.results)
    rnd = Path(str(res) + "_random")
    u = json.load(open(ROOT / "values.json"))["u"]
    R = json.load(open(res / "probes.json"))
    C = json.load(open(rnd / "probes.json")) if (rnd / "probes.json").exists() else None
    layers = list(range(len(R["r2"]["v_a"])))
    show = [li for li in layers if li % 4 == 0]
    rows = [("v_a", "a_end"), ("v_a", "reply_end"), ("v_a", "t-1"),
            ("v_b", "b_end"), ("v_b", "reply_end"), ("v_b", "t-1")]
    out = {"layers": layers, "real": {}, "random": {}}
    lines = ["# Unseen goods, option by option",
             "",
             "Correlation, over the 9 real options, between each option's held-out value estimate",
             "(probe trained without that option, number of people removed) and its true value from",
             "pilot 3. With 9 points, |r| needs to be above about 0.67 for p < 0.05 (two-sided).",
             "",
             "| target | position | model | " + " | ".join(f"L{li}" for li in show) + " |",
             "|---|---|---|" + "---|" * len(show)]
    for tgt, pos in rows:
        for tag, O in [("real", R), ("random", C)]:
            if O is None:
                continue
            vals = [option_corr(O["u_recovery"].get(f"{tgt}|{li}|{pos}", {}), u)[0] for li in layers]
            out[tag][f"{tgt}|{pos}"] = vals
            lines.append(f"| {tgt} | {pos} | {tag} | " + " | ".join(
                "–" if not np.isfinite(vals[li]) else f"{vals[li]:+.2f}" for li in show) + " |")
    best = max(layers, key=lambda li: np.nan_to_num(out["real"]["v_a|t-1"][li], nan=-9))
    est = R["u_recovery"][f"v_a|{best}|t-1"]
    est_c = C["u_recovery"][f"v_a|{best}|t-1"] if C else {}
    ks = sorted([k for k in est if not k.startswith("N_")], key=lambda k: u[k])
    lines += ["", f"## Option A's value at the final position, layer {best}, each option held out",
              "", "| option | true value u | estimate, Qwen3-14B | estimate, random weights |", "|---|---|---|---|"]
    for k in ks:
        lines.append(f"| {k} | {u[k]:+.2f} | {est[k]:+.2f} | {est_c.get(k, float('nan')):+.2f} |")
    lines += ["", "Estimates are on an arbitrary offset (the probe's training mean is removed per fold);",
              "only their order and spacing matter."]
    (res / "option_level.md").write_text("\n".join(lines) + "\n")
    json.dump(out, open(res / "option_level.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
