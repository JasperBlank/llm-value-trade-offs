"""Step 6: do different kinds of harm share one signal inside the network?

For every scenario version we read the model's activation at the moment it is about
to answer (the last prompt token, at the same layer used for steering), once with the
mildest harm level (0) and once with the most serious (3). Every other part of the
prompt is identical, so

    harm direction = mean activation at level 3  minus  mean activation at level 0

isolates "more of this harm". We average over the four prompt cells (goal x monitored)
and both option orders. Then we compare the directions of all versions by cosine
similarity (1 identical, 0 unrelated, -1 opposite).

Pre-registered questions (see prereg/PREREGISTRATION.md):
  6a  Are versions of the SAME harm type more similar to each other than to versions
      of DIFFERENT harm types? Only pairs from different stories are compared, because
      two versions of the same story share most of their text. Tested by shuffling the
      versions 10,000 times while keeping the story structure (permutation test; see
      situation_preserving_permutation in common.py for why).
  6b  Do ALL harm types point roughly the same way? Mean cosine over pairs of different
      harm types from different stories, compared with 0.3.

Known confound, to state in the post: serious harm levels may share features that are
not "harm" at all, such as being longer or sounding sadder. A control for that (the same
bad detail happening to someone the decision does not affect) is planned for the
follow-up, not this pilot.

This step needs no generation, only one forward pass per prompt, so it takes minutes.
Outputs: harm_directions.npz, harm_directions.md, fig_harm_similarity.png
"""
import itertools

import numpy as np

from common import (decision_system, decision_user, get_backend, load_config, log, out_path, parse_args,
                    permutation_p, resolve_layer, save_json, situation_preserving_permutation)

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def main():
    args = parse_args(__doc__)
    cfg = load_config(args)
    backend = get_backend(cfg)
    dpath = out_path(cfg, "direction.npz")
    layer = int(np.load(dpath)["layer"]) if dpath.exists() else resolve_layer(cfg, backend)
    log(f"Reading activations at layer {layer}")

    versions = cfg["scenarios"]
    dirs = {}
    for si, sc in enumerate(versions):
        means = {}
        for level in (0, len(sc["harm_levels"]) - 1):
            prompts, metas = [], []
            for goal in (0, 1):
                for mon in (0, 1):
                    for order in (0, 1):
                        prompts.append(backend.chat(decision_system(cfg, sc, goal, mon),
                                                    decision_user(cfg, sc, level, order)))
                        metas.append({"level": level, "harm_dimension": sc.get("harm_dimension", sc["id"])})
            acts = backend.last_token_activations(prompts, layer, batch_size=int(cfg["batch_size"]), metas=metas)
            means[level] = acts.mean(0)
        d = means[max(means)] - means[min(means)]
        dirs[sc["id"]] = d / np.linalg.norm(d)
        log(f"  {sc['id']}: done")

    ids = [sc["id"] for sc in versions]
    types = {sc["id"]: sc.get("harm_dimension", sc["id"]) for sc in versions}
    fams = {sc["id"]: sc.get("family", sc["id"]) for sc in versions}
    M = np.stack([dirs[i] for i in ids])
    C = M @ M.T
    np.savez(out_path(cfg, "harm_directions.npz"), ids=np.array(ids), directions=M, cosine=C, layer=layer)

    pairs = [(a, b) for a, b in itertools.combinations(range(len(ids)), 2) if fams[ids[a]] != fams[ids[b]]]

    def split(Cm):
        same = [Cm[a, b] for a, b in pairs if types[ids[a]] == types[ids[b]]]
        diff = [Cm[a, b] for a, b in pairs if types[ids[a]] != types[ids[b]]]
        return same, diff

    def gap_of(Cm):
        s, d = split(Cm)
        return np.mean(s) - np.mean(d) if s and d else float("nan")

    same, diff = split(C)
    gap = gap_of(C)
    rng = np.random.default_rng(cfg["seed"])
    fam_list = [fams[i] for i in ids]
    null = []
    for _ in range(int(cfg["prereg"].get("permutations", 10000))):
        p = situation_preserving_permutation(fam_list, rng)
        null.append(gap_of(C[np.ix_(p, p)]))
    p_6a = permutation_p(null, gap)
    mean_cross = float(np.mean(diff)) if diff else float("nan")

    md = ["# Do different harms share one signal?", "",
          f"Layer {layer}. Directions: activation at the last prompt token, most serious minus mildest "
          "harm level. Only pairs from different stories are compared.", "",
          "| question | result | pre-registered rule | supported |", "|---|---|---|---|",
          f"| 6a. same harm type more alike | same-type mean {np.mean(same):.2f} vs different-type "
          f"{np.mean(diff):.2f} (gap {gap:+.2f}, permutation p = {p_6a:.3f}, {len(same)} same-type pairs) "
          f"| gap > 0 and p < 0.05 | {'yes' if gap > 0 and p_6a < 0.05 else 'no'} |",
          f"| 6b. one shared signal | mean cosine across harm types {mean_cross:.2f} | above 0.30 "
          f"| {'yes' if mean_cross > 0.3 else 'no'} |", "",
          "Caveat: serious harm levels may share features that are not harm itself (longer or sadder "
          "text), which would raise 6b. The control for that is planned for the follow-up.", "",
          "![similarity](fig_harm_similarity.png)", "", "Cosine similarity between harm directions:", ""]
    short = [f"{i} ({types[i]})" for i in ids]
    md.append("| | " + " | ".join(str(k + 1) for k in range(len(ids))) + " |")
    md.append("|---" * (len(ids) + 1) + "|")
    for a, name in enumerate(short):
        md.append(f"| {a + 1}. {name} | " + " | ".join(f"{C[a, b]:.2f}" for b in range(len(ids))) + " |")
    out_path(cfg, "harm_directions.md").write_text("\n".join(md), encoding="utf-8")
    save_json({"layer": layer, "same_type_mean": float(np.mean(same)) if same else None,
               "different_type_mean": float(np.mean(diff)) if diff else None, "gap": float(gap),
               "permutation_p": p_6a, "same_type_more_alike": bool(gap > 0 and p_6a < 0.05),
               "mean_cosine_across_types": mean_cross, "one_shared_signal": bool(mean_cross > 0.3)},
              out_path(cfg, "harm_directions.json"))

    plot(C, short, cfg)
    print("\n".join(md[4:8]))
    log(f"Details: {out_path(cfg, 'harm_directions.md')}")


def plot(C, names, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])  # red - gray - blue
    n = len(names)
    fig, ax = plt.subplots(figsize=(0.55 * n + 4.5, 0.55 * n + 1.5), facecolor=SURFACE)
    im = ax.imshow(C, cmap=cmap, vmin=-1, vmax=1)
    for a in range(n):
        for b in range(n):
            ax.text(b, a, f"{C[a, b]:.2f}", ha="center", va="center", fontsize=7,
                    color=INK if abs(C[a, b]) < 0.6 else SURFACE)
    ax.set_xticks(range(n), [str(k + 1) for k in range(n)], color=INK2)
    ax.set_yticks(range(n), [f"{k + 1}. {nm}" for k, nm in enumerate(names)], color=INK, fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, length=0)
    ax.set_title("How similar are the harm directions? (1 = same, 0 = unrelated)", color=INK, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path(cfg, "fig_harm_similarity.png"), dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


if __name__ == "__main__":
    main()
