"""Stage `circuit_analyze` (exploratory, CPU): read the decomposition.

Writers. For A's direction at layer L (and B's), over prompts whose option is a real good, each
piece c (embedding, head (block, h), MLP of a block) gets
  share_c = cov(piece_c, V) / var(V)          (sums to the readout's slope on the true value)
  beta_q, beta_u: its regression on log(number) and on the good's value u together
                  (a piece that carries value itself has beta_q / beta_u = b = 1.46).
Reported: the top pieces, how many pieces carry half of the total, heads against MLPs per block,
and for the top heads, which part of the prompt their contribution was read from.
Readers. Each piece's direct contribution to logit(A) - logit(B), against the model's log-odds:
top reader heads, their attention to the "A" and "B" labels, and whether that attention follows
the decision ("correct letter" heads).
Writes results/<run>/circuit.json, writers.json (the pieces patched in stage `patch`), figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import REGIONS, acts_path, log, save_json, values


def piece_name(blk, h, H):
    return f"MLP {blk}" if h == H else f"head {blk}.{h}"


def cov_share(X, y):
    """X (n, k) pieces, y (n,): cov(X_k, y) / var(y)."""
    yc = y - y.mean()
    return (X - X.mean(0)).T @ yc / (yc @ yc)


def two_reg(X, q, u):
    """Least squares of each column of X on [q, u] (centred): returns beta_q, beta_u (k,)."""
    Z = np.column_stack([q - q.mean(), u - u.mean()])
    B = np.linalg.lstsq(Z, X - X.mean(0), rcond=None)[0]
    return B[0], B[1]


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run(ctx):
    cfg, rows = ctx.cfg, ctx.rows
    b_val, u = values()
    cm = json.load(open(acts_path(cfg, "circuit_meta.json")))
    dnames, snames, nL, H = cm["directions"], cm["source_directions"], cm["n_blocks"], cm["heads"]
    comp = np.load(acts_path(cfg, "comp.npy"), mmap_mode="r")
    emb = np.load(acts_path(cfg, "emb.npy"))
    fin = np.load(acts_path(cfg, "final.npz"))
    test = [rows[i] for i in fin["test_index"]]
    N = len(test)
    P = int(cfg["primary_layer"])
    out = {"primary_layer": P, "b": b_val, "writers": {}, "readers": {}, "regions": REGIONS}
    C = np.asarray(comp, np.float32)                                    # (N, nL, H+1, D) ~ fits in memory
    src = np.load(acts_path(cfg, "src.npy"))                           # ~1.2 GB, read once (network storage)
    att = np.load(acts_path(cfg, "att.npy"))
    ridx = {r: i for i, r in enumerate(REGIONS)}

    # ------------------------------------------------------------------ writers
    for L in cfg["layers"]:
        for slot in ("a", "b"):
            di = dnames.index(f"v_{slot}@{L}")
            keys = np.array([r["key_" + slot] for r in test])
            real = np.array([not k.startswith("N_") for k in keys])
            y = np.array([r["v_" + slot] for r in test])[real]
            q = np.log([r["q_" + slot] for r in test])[real]
            uu = np.array([u[k] for k in keys[real]])
            X = np.concatenate([emb[real, di:di + 1], C[real, :L, :, di].reshape(int(real.sum()), -1)], 1)
            names = ["embedding"] + [piece_name(blk, h, H) for blk in range(L) for h in range(H + 1)]
            share = cov_share(X, y)
            bq, bu = two_reg(X, q, uu)
            total = float(share.sum())
            order = np.argsort(-share)
            pos = np.cumsum(share[order])
            n50 = int(np.searchsorted(pos, 0.5 * total) + 1)
            n80 = int(np.searchsorted(pos, 0.8 * total) + 1)
            heads_by_blk = share[1:].reshape(L, H + 1)[:, :H].sum(1)
            mlp_by_blk = share[1:].reshape(L, H + 1)[:, H]
            top = [{"piece": names[i], "share": float(share[i]), "share_of_total": float(share[i] / total),
                    "beta_q": float(bq[i]), "beta_u": float(bu[i]),
                    "ratio_q_to_u": float(bq[i] / bu[i]) if abs(bu[i]) > 1e-9 else None} for i in order[:20]]
            rec = {"total_slope": total, "pieces_for_half": n50, "pieces_for_80pct": n80, "n_pieces": len(names),
                   "embedding_share": float(share[0] / total), "heads_share": float(heads_by_blk.sum() / total),
                   "mlps_share": float(mlp_by_blk.sum() / total),
                   "heads_by_block": (heads_by_blk / total).tolist(), "mlps_by_block": (mlp_by_blk / total).tolist(),
                   "total_beta_q": float(bq.sum()), "total_beta_u": float(bu.sum()),
                   "total_ratio_q_to_u": float(bq.sum() / bu.sum()) if abs(bu.sum()) > 1e-9 else None,
                   "top": top}
            # which parts of the prompt the top heads' contributions come from (primary layer only)
            if L == P and f"v_{slot}@{P}" in snames:
                si = snames.index(f"v_{slot}@{P}")
                top_heads = [i for i in order if names[i].startswith("head")][:10]
                srcs = []
                for i in top_heads:
                    blk, h = (int(x) for x in names[i].split()[1].split("."))
                    S_ = np.asarray(src[:, blk, h, :, si], np.float32)[real]       # (n, R)
                    A_ = np.asarray(att[:, blk, h, :], np.float32)[real]
                    srcs.append({"piece": names[i], "share_of_total": float(share[i] / total),
                                 "share_by_region": dict(zip(REGIONS, (cov_share(S_, y) / total).tolist())),
                                 "attention_by_region": dict(zip(REGIONS, A_.mean(0).tolist()))})
                all_heads = np.zeros(len(REGIONS))
                for blk in range(L):
                    S_ = np.asarray(src[:, blk, :, :, si], np.float32)[real].sum(1)  # all heads of a block
                    all_heads += cov_share(S_, y)
                rec["top_heads_sources"] = srcs
                rec["all_heads_share_by_region"] = dict(zip(REGIONS, (all_heads / total).tolist()))
                # check: the per-token split adds up to the heads' totals
                tot_src = sum(np.asarray(src[:, blk, :, :, si], np.float32).sum((1, 2)) for blk in range(L))
                tot_comp = C[:, :L, :H, di].sum((1, 2))
                rec["source_split_rel_err"] = float(np.abs(tot_src - tot_comp).max() / (np.abs(tot_comp).max() + 1e-9))
            out["writers"][f"v_{slot}@{L}"] = rec
            log(f"  writers of {slot.upper()}'s value at layer {L}: slope {total:.2f}; heads {rec['heads_share']:.0%}, "
                f"MLPs {rec['mlps_share']:.0%}, embedding {rec['embedding_share']:.0%}; half carried by {n50} pieces; "
                f"top {top[0]['piece']} ({top[0]['share_of_total']:.0%})")

    # ------------------------------------------------------------------ readers
    li = dnames.index("logit")
    lo = fin["logodds"]
    DLA = C[:, :, :, li] / fin["rms"][:, None, None]                    # (N, nL, H+1)
    X = DLA.reshape(N, -1)
    names = [piece_name(blk, h, H) for blk in range(nL) for h in range(H + 1)]
    g = cov_share(X, lo)
    tot = float(g.sum() + cov_share(emb[:, li:li + 1] / fin["rms"][:, None], lo)[0])
    order = np.argsort(-g)
    head_order = [i for i in order if names[i].startswith("head")]
    dv = np.array([r["v_a"] - r["v_b"] for r in test])
    readers = []
    si = snames.index("logit")
    for i in head_order[:int(cfg["circuit"]["n_readers"])]:
        blk, h = (int(x) for x in names[i].split()[1].split("."))
        A_ = np.asarray(att[:, blk, h, :], np.float32)
        diff = A_[:, ridx["a_label"]] - A_[:, ridx["b_label"]]
        S_ = np.asarray(src[:, blk, h, :, si], np.float32) / fin["rms"][:, None]
        readers.append({"piece": names[i], "block": blk, "head": h, "share_of_logodds": float(g[i] / tot),
                        "attention_a_label": float(A_[:, ridx["a_label"]].mean()),
                        "attention_b_label": float(A_[:, ridx["b_label"]].mean()),
                        "corr_label_attention_diff_with_logodds": corr(diff, lo),
                        "corr_label_attention_diff_with_value_diff": corr(diff, dv),
                        "share_by_region": dict(zip(REGIONS, (cov_share(S_, lo) / tot).tolist()))})
    thr = float(cfg["circuit"]["letter_head_corr"])
    out["readers"] = {"total": tot, "heads_share": float(g.reshape(nL, H + 1)[:, :H].sum() / tot),
                      "mlps_share": float(g.reshape(nL, H + 1)[:, H].sum() / tot),
                      "by_block_heads": (g.reshape(nL, H + 1)[:, :H].sum(1) / tot).tolist(),
                      "by_block_mlps": (g.reshape(nL, H + 1)[:, H] / tot).tolist(),
                      "top_heads": readers,
                      "letter_heads": [r["piece"] for r in readers if r["corr_label_attention_diff_with_logodds"] >= thr]}
    log(f"  readers: heads {out['readers']['heads_share']:.0%} of the log-odds, MLPs {out['readers']['mlps_share']:.0%}; "
        f"letter heads among the top {len(readers)}: {out['readers']['letter_heads']}")

    # ------------------------------------------------------------------ pieces to patch (stage `patch`)
    w = out["writers"][f"v_a@{P}"]
    di = dnames.index(f"v_a@{P}")
    keys = np.array([r["key_a"] for r in test])
    real = np.array([not k.startswith("N_") for k in keys])
    q = np.log([r["q_a"] for r in test])[real]
    uu = np.array([u[k] for k in keys[real]])
    Xp = C[real, :P, :, di].reshape(int(real.sum()), -1)
    _, bu = two_reg(Xp, q, uu)
    rank = np.argsort(-np.abs(bu))
    kmax = max(cfg["circuit"]["topk"])
    top = [(int(i // (H + 1)), int(i % (H + 1))) for i in rank[:kmax]]
    rng = np.random.default_rng(int(cfg["circuit"]["seed"]))
    rest = rank[kmax:]
    rand = {str(k): [(int(i // (H + 1)), int(i % (H + 1))) for i in rng.choice(rest, size=k, replace=False)]
            for k in cfg["circuit"]["topk"]}
    save_json({"layer": P, "ranked_by": "|beta_u| on A's direction (the good's part)", "H": H,
               "top": top, "top_names": [piece_name(a, h, H) for a, h in top], "random": rand},
              Path(cfg["res"]) / "writers.json")
    out["patch_writers"] = [piece_name(a, h, H) for a, h in top[:20]]
    save_json(out, Path(cfg["res"]) / "circuit.json")
    figures(cfg, out, nL)
    log("circuit_analyze done")


def figures(cfg, out, nL):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = out["primary_layer"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for ax, key, title in ((axes[0], f"v_a@{P}", "A's value"), (axes[1], f"v_b@{P}", "B's value")):
        w = out["writers"][key]
        x = np.arange(len(w["heads_by_block"]))
        ax.bar(x - 0.2, w["heads_by_block"], 0.4, label="attention heads")
        ax.bar(x + 0.2, w["mlps_by_block"], 0.4, label="MLP")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"Who writes {title} (read at layer {P}), share per block", fontsize=9)
        ax.set_xlabel("block")
        ax.legend(fontsize=7)
    r = out["readers"]
    x = np.arange(len(r["by_block_heads"]))
    axes[2].bar(x - 0.2, r["by_block_heads"], 0.4, label="attention heads")
    axes[2].bar(x + 0.2, r["by_block_mlps"], 0.4, label="MLP")
    axes[2].axhline(0, color="grey", lw=0.5)
    axes[2].set_title("Who writes the answer (direct share of the log-odds)", fontsize=9)
    axes[2].set_xlabel("block")
    axes[2].legend(fontsize=7)
    fig.savefig(Path(cfg["res"]) / "fig_writers_readers.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    w = out["writers"][f"v_a@{P}"]
    if w.get("top_heads_sources"):
        S = np.array([[h["share_by_region"][g] for g in out["regions"]] for h in w["top_heads_sources"]])
        fig, ax = plt.subplots(figsize=(10, 4.5))
        im = ax.imshow(S, cmap="RdBu_r", vmin=-np.abs(S).max(), vmax=np.abs(S).max(), aspect="auto")
        ax.set_yticks(range(len(S)), [h["piece"] for h in w["top_heads_sources"]], fontsize=7)
        ax.set_xticks(range(len(out["regions"])), out["regions"], rotation=60, fontsize=7)
        ax.set_title(f"Top heads writing A's value: where in the prompt it comes from (share of the total)", fontsize=9)
        fig.colorbar(im, ax=ax)
        fig.savefig(Path(cfg["res"]) / "fig_value_sources.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
