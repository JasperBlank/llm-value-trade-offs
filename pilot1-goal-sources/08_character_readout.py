"""Step 8: do the goal line and the logging line move the model's character?

A first, cheap piece of mechanism. For every unsteered prompt of the main experiment we
read where the model sits along the Assistant direction (from step 1) at the moment it is
about to answer: the activation at the last prompt token, at the steering layer, projected
onto the direction. Higher = more like its default Assistant.

Questions (exploratory, not pre-registered as tests):
  - Does the goal line move the model along the Assistant direction?
  - Does "logging is disabled" move it?
Shifts are expressed as a fraction of the push our "away" steering applies, so
"-0.25" means the cue moves the model a quarter as far toward the characters as the
steering does.

If the main run exists, we also check whether prompt cells that sit further from the
Assistant are the ones where the model protects the third party less (correlation across
cells, within each version so that differences between stories don't drive it). That is
consistent with the cue working THROUGH the character, though not proof of it.

Outputs: character_readout.json, character_readout.md
"""
import json

import numpy as np

from common import (decision_system, decision_user, get_backend, load_config, load_jsonl, log, out_path,
                    parse_args, parse_decision, protect_from, save_json)


def main():
    args = parse_args(__doc__)
    cfg = load_config(args)
    dp = out_path(cfg, "direction.npz")
    if not dp.exists():
        raise SystemExit("Missing direction.npz. Run 01_extract_direction.py first.")
    d = np.load(dp)
    unit, layer, mean_norm = d["unit"], int(d["layer"]), float(d["mean_norm"])
    alpha = cfg["main"].get("alpha")
    ap = out_path(cfg, "alpha_choice.json")
    if alpha is None and ap.exists():
        alpha = json.loads(ap.read_text()).get("alpha")
    push = float(alpha) * mean_norm if alpha else None
    stakes = {}
    sp = out_path(cfg, "stakes_choice.json")
    if sp.exists():
        stakes = json.loads(sp.read_text())

    backend = get_backend(cfg)
    rows = []
    for sc in cfg["scenarios"]:
        level = int(stakes.get(sc["id"], {}).get("level", 1))
        prompts, metas = [], []
        for goal in (0, 1):
            for mon in (0, 1):
                for order in (0, 1):
                    prompts.append(backend.chat(decision_system(cfg, sc, goal, mon), decision_user(cfg, sc, level, order)))
                    metas.append({"goal": goal, "monitored": mon, "order": order, "level": level,
                                  "harm_dimension": sc.get("harm_dimension", sc["id"])})
        acts = backend.last_token_activations(prompts, layer, batch_size=int(cfg["batch_size"]), metas=metas)
        for m, a in zip(metas, acts):
            rows.append({"scenario": sc["id"], "family": sc.get("family", sc["id"]), "goal": m["goal"],
                         "monitored": m["monitored"], "order": m["order"], "projection": float(a @ unit)})
    log(f"Read {len(rows)} prompts at layer {layer}")

    versions = sorted({r["scenario"] for r in rows})
    fam_of = {r["scenario"]: r["family"] for r in rows}
    families = sorted(set(fam_of.values()))

    def cell(sc, goal=None, mon=None):
        v = [r["projection"] for r in rows if r["scenario"] == sc and (goal is None or r["goal"] == goal)
             and (mon is None or r["monitored"] == mon)]
        return float(np.mean(v))

    def effects(vs):
        g = np.mean([cell(v, goal=1) - cell(v, goal=0) for v in vs])
        u = np.mean([cell(v, mon=0) - cell(v, mon=1) for v in vs])
        return g, u

    g, u = effects(versions)
    rng = np.random.default_rng(cfg["seed"])
    draws = []
    for _ in range(int(cfg["prereg"]["bootstrap_reps"])):
        picked = rng.choice(len(families), size=len(families), replace=True)
        vs = [v for i in picked for v in versions if fam_of[v] == families[i]]
        draws.append(effects(vs))
    ci = np.percentile(np.array(draws), [2.5, 97.5], axis=0)

    def scaled(x):
        return x / push if push else float("nan")

    result = {"layer": layer, "steering_push": push,
              "goal_shift": g, "goal_shift_ci": ci[:, 0].tolist(), "goal_shift_vs_steering": scaled(g),
              "unmonitored_shift": u, "unmonitored_shift_ci": ci[:, 1].tolist(),
              "unmonitored_shift_vs_steering": scaled(u)}

    # Link to behaviour: per (version, goal, monitored) cell, care rate (unsteered) vs projection.
    main_rows = load_jsonl(out_path(cfg, "main.jsonl"))
    if main_rows:
        care = {}
        for r in main_rows:
            if r.get("persona") != "none":
                continue
            p = protect_from(parse_decision(r["text"]), r["order"])
            if p is not None:
                care.setdefault((r["scenario"], r["goal"], r["monitored"]), []).append(p)
        xs, ys = [], []
        for v in versions:
            pts = [(cell(v, gg, mm), np.mean(care[(v, gg, mm)])) for gg in (0, 1) for mm in (0, 1)
                   if (v, gg, mm) in care]
            if len(pts) >= 3:
                px, py = np.array(pts).T
                xs += list(px - px.mean())
                ys += list(py - py.mean())
        if len(xs) >= 6 and np.std(xs) > 0 and np.std(ys) > 0:
            result["within_version_correlation_projection_vs_care"] = float(np.corrcoef(xs, ys)[0, 1])
            result["n_cells"] = len(xs)
    save_json(result, out_path(cfg, "character_readout.json"))

    md = ["# Do the goal and logging lines move the model's character?", "",
          f"Projection onto the Assistant direction at the last prompt token, layer {layer}, no steering. "
          "Negative = further from the default Assistant.", "",
          "| cue | shift | 95% interval | as a fraction of the away-steering push |", "|---|---|---|---|",
          f"| Goal line | {g:+.3f} | [{ci[0, 0]:+.3f}, {ci[1, 0]:+.3f}] | {scaled(g):+.2f} |",
          f"| Logging disabled | {u:+.3f} | [{ci[0, 1]:+.3f}, {ci[1, 1]:+.3f}] | {scaled(u):+.2f} |", ""]
    if "within_version_correlation_projection_vs_care" in result:
        md += [f"Within-version correlation between a cell's position and its care rate: "
               f"r = {result['within_version_correlation_projection_vs_care']:+.2f} over {result['n_cells']} cells. "
               "A positive r means cells closer to the Assistant protect the third party more. "
               "Exploratory: consistent with the cues acting through the character, not proof of it."]
    out_path(cfg, "character_readout.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
