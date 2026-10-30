"""Step 1: find the "Assistant direction" in the model's activations.

The model answers neutral questions (a) as its default self and (b) while playing
20 benign characters. At one middle layer we average the activations over the
answer tokens. The direction is:

    mean(default Assistant)  minus  mean(all characters),  scaled to length 1

This is a simplified version of the "Assistant Axis" (Lu et al., 2026), which is
the first principal component of many character directions. As a sanity check we
also compute that principal component and report how well the two agree.

A random direction of the same length is saved too; it is the control condition.

Outputs (in results/<run>/):
  direction_answers.jsonl   every generated answer
  direction.npz             unit, random_unit, mean_norm, layer, pc1
  direction_report.json     agreement with PC1 and where each character sits
"""
from collections import defaultdict

import numpy as np

from common import (Steer, fill, get_backend, load_config, load_jsonl, log, out_path, parse_args,
                    resolve_layer, run_jobs, save_json)


def main():
    args = parse_args(__doc__)
    cfg = load_config(args)
    D, P = cfg["direction"], cfg["prompts"]
    backend = get_backend(cfg)
    layer = resolve_layer(cfg, backend)
    log(f"Using layer {layer} of {backend.n_layers}")

    questions = P["neutral_questions"][: D["n_questions"]]
    jobs = []
    for qi, q in enumerate(questions):
        for s in range(D["samples_default"]):
            jobs.append({"id": f"default|{qi}|{s}", "steer": "none", "system": P["default_system"], "user": q,
                         "meta": {"kind": "free", "who": "default", "q": qi,
                                  "system": P["default_system"], "user": q}})
        for ri, role in enumerate(P["roles"]):
            system = fill(P["role_system"], role=role)
            jobs.append({"id": f"role{ri}|{qi}", "steer": "none", "system": system, "user": q,
                         "meta": {"kind": "free", "who": f"role{ri}", "role": role, "q": qi,
                                  "system": system, "user": q}})
    answers_path = out_path(cfg, "direction_answers.jsonl")
    gen = dict(cfg["generation"], max_new_tokens=D["max_new_tokens"])
    run_jobs(backend, jobs, {"none": Steer("none", 0.0, None, layer)}, answers_path, cfg, gen,
             label="character answers")

    log("Measuring activations on the answers ...")
    sums, counts, norms = defaultdict(float), defaultdict(int), []
    recs = [r for r in load_jsonl(answers_path) if r.get("text", "").strip()]
    for i, r in enumerate(recs):
        mean, norm = backend.response_mean(backend.chat(r["system"], r["user"]), r["text"], layer)
        if mean is None:
            continue
        sums[r["who"]] = sums[r["who"]] + mean
        counts[r["who"]] += 1
        if r["who"] == "default":
            norms.append(norm)
        if (i + 1) % 100 == 0:
            log(f"  {i + 1}/{len(recs)}")

    means = {who: sums[who] / counts[who] for who in sums}
    default = means["default"]
    role_keys = sorted(k for k in means if k != "default")
    roles_pooled = np.mean([means[k] for k in role_keys], axis=0)
    diff = default - roles_pooled
    unit = diff / np.linalg.norm(diff)

    # Sanity check: first principal component over the character means plus the default.
    M = np.stack([means[k] for k in role_keys] + [default])
    Mc = M - M.mean(0)
    _, svals, vt = np.linalg.svd(Mc, full_matrices=False)
    pc1 = vt[0] if (default - M.mean(0)) @ vt[0] > 0 else -vt[0]
    cos_pc1 = float(unit @ pc1)
    var_pc1 = float(svals[0] ** 2 / (svals ** 2).sum())

    rng = np.random.default_rng(cfg["seed"])
    rnd = rng.standard_normal(unit.shape[0])
    rnd /= np.linalg.norm(rnd)
    mean_norm = float(np.mean(norms))

    np.savez(out_path(cfg, "direction.npz"), unit=unit.astype(np.float32), random_unit=rnd.astype(np.float32),
             mean_norm=mean_norm, layer=layer, pc1=pc1.astype(np.float32), gap=float(np.linalg.norm(diff)))

    role_names = {r["who"]: r.get("role", "default Assistant") for r in recs}
    proj = {role_names[k]: float((means[k] - roles_pooled) @ unit) for k in role_keys + ["default"]}
    report = {
        "layer": layer,
        "n_answers": {k: counts[k] for k in sorted(counts)},
        "mean_activation_norm": mean_norm,
        "cosine_direction_vs_pc1": cos_pc1,
        "variance_explained_by_pc1": var_pc1,
        "cosine_random_vs_direction": float(rnd @ unit),
        "projection_on_direction": dict(sorted(proj.items(), key=lambda kv: -kv[1])),
    }
    save_json(report, out_path(cfg, "direction_report.json"))

    log(f"Saved direction.npz (layer {layer}, typical activation size {mean_norm:.1f})")
    log(f"Agreement with the first principal component: cosine {cos_pc1:.2f} "
        f"(PC1 explains {var_pc1:.0%} of the variance between characters)")
    print("\nWhere each character sits along the direction (default Assistant should be highest):")
    for name, v in report["projection_on_direction"].items():
        print(f"  {v:8.2f}  {name}")
    if cos_pc1 < 0.5:
        log("NOTE: weak agreement with PC1. The direction may mix several things; mention this in the post.")


if __name__ == "__main__":
    main()
