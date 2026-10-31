"""Pilot 4, step 4: a first causal look (exploratory). Needs the model (GPU).

  python steer.py

Takes the layer where a probe best reads the value of goods it never saw (option A, end of its
line), refits that probe without a held-out set of steering prompts, and adds the probe's
direction to every token of option A's line while the model reads it, at several strengths
(in the probe's value units). If the direction carries value the way the model uses it, the
log-odds of choosing A should rise by about the added value (slope about 1). A random
direction of the same length is the control. Writes results/<run>/steering.json and a figure.
"""
from __future__ import annotations

import json

import numpy as np

from common import acts_path, char_landmarks, load_config, load_prompts, log, parse_args, token_at


def main():
    args = parse_args(__doc__)
    cfg = load_config(args)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    S = cfg["steering"]
    rows = load_prompts(cfg)
    meta = np.load(acts_path(cfg, "meta.npz"))
    resid = np.load(acts_path(cfg, "resid.npy"), mmap_mode="r")
    pr = json.load(open(f"{cfg['res']}/probes.json"))
    names = pr["positions"]
    pe = names.index("a_end")
    urec = np.array(pr["u_recovery_r"]["v_a_real"], float)[:, pe]
    r2 = np.array(pr["r2"]["v_a"], float)[:, pe]
    lo, hi = S["layer_range"]
    hi = min(hi, len(urec) - 2)                       # the last recorded layer is after the final norm
    cand = list(range(max(1, min(lo, hi)), hi + 1))
    layer = max(cand, key=lambda li: (np.nan_to_num(urec[li], nan=-9), np.nan_to_num(r2[li], nan=-9)))
    log(f"Steering layer {layer}: unseen-goods partial r {urec[layer]:.2f}, held-out R2 {r2[layer]:.2f}")

    # Steering prompts: pairs of real goods (and gain wordings) the model is undecided about.
    pa = meta["p_a"] / np.maximum(meta["p_a"] + meta["p_b"], 1e-9)
    pool = [r["index"] for r in rows if r["group"] == "test" and r["kind"] in ("within", "cross", "mirror")]
    rng = np.random.default_rng(31)
    near = [i for i in pool if 0.2 < pa[i] < 0.8]
    if len(near) < S["n_prompts"]:
        near = sorted(pool, key=lambda i: abs(pa[i] - 0.5))[:S["n_prompts"]]
    steer_idx = sorted(rng.choice(near, size=min(S["n_prompts"], len(near)), replace=False).tolist())
    held = set(steer_idx)

    # Refit the value probe at that layer without the steering prompts.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train = [r["index"] for r in rows if r["group"] == "test" and r["index"] not in held]
    X = torch.tensor(np.asarray(resid[train, layer, pe], dtype=np.float32), device=dev)
    y = torch.tensor([rows[i]["v_a"] for i in train], dtype=torch.float32, device=dev)
    c = pr["best_c"].get(f"v_a|{layer}|a_end", 0.01)
    xm, ym = X.mean(0), y.mean()
    Xc = X - xm
    G = Xc.T @ Xc
    w = torch.linalg.solve(G + c * G.diagonal().mean() * torch.eye(G.shape[0], device=dev), Xc.T @ (y - ym))
    direction = (w / (w @ w)).float()                 # adding alpha * direction raises the readout by alpha
    resid_norm = float(X.norm(dim=1).mean())
    g = torch.Generator(device="cpu").manual_seed(7)
    rand = torch.randn(direction.shape, generator=g).to(dev)
    rand = rand / rand.norm() * direction.norm()
    log(f"Direction length {float(direction.norm()):.3g} (mean residual length {resid_norm:.3g})")

    log(f"Loading {cfg['model']} ...")
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    dtype = getattr(torch, cfg["dtype"]) if dev == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=dtype)
    model = model.to(dev).eval()
    letter_ids = []
    for letter in "AB":
        ids = {e[0] for v in (letter, " " + letter) if len(e := tok.encode(v, add_special_tokens=False)) == 1}
        letter_ids.append(sorted(ids))

    encs, spans = [], []
    for i in steer_idx:
        r = rows[i]
        msgs = ([{"role": "system", "content": r["system"]}] if r["system"] else []) + [{"role": "user", "content": r["user"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        marks = char_landmarks(text, r["user"])
        s0 = token_at(enc["offset_mapping"], marks["a_label"][0])
        s1 = token_at(enc["offset_mapping"], marks["a_text"][1] - 1)
        encs.append(enc["input_ids"])
        spans.append((s0, s1))
    state = {"vec": None, "mask": None}

    def hook(module, inputs, output):
        if state["vec"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = h + state["mask"][..., None].to(h.dtype) * state["vec"].to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handle = model.model.layers[layer - 1].register_forward_hook(hook)    # hidden_states[layer] = output of layers[layer-1]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    bs = int(cfg["batch_size"])

    def run(vec):
        out = np.zeros(len(steer_idx))
        for k0 in range(0, len(steer_idx), bs):
            sl = list(range(k0, min(k0 + bs, len(steer_idx))))
            T = max(len(encs[j]) for j in sl)
            ids = torch.full((len(sl), T), pad_id, dtype=torch.long)
            att = torch.zeros((len(sl), T), dtype=torch.long)
            mask = torch.zeros((len(sl), T))
            for b, j in enumerate(sl):
                pad = T - len(encs[j])
                ids[b, pad:] = torch.tensor(encs[j])
                att[b, pad:] = 1
                mask[b, pad + spans[j][0]: pad + spans[j][1] + 1] = 1
            state["vec"], state["mask"] = vec, mask.to(dev)
            with torch.no_grad():
                try:
                    o = model(input_ids=ids.to(dev), attention_mask=att.to(dev), logits_to_keep=1)
                except TypeError:
                    o = model(input_ids=ids.to(dev), attention_mask=att.to(dev))
            p = torch.softmax(o.logits[:, -1, :].float(), -1)
            a, b_ = p[:, letter_ids[0]].sum(-1), p[:, letter_ids[1]].sum(-1)
            pn = (a / (a + b_)).clamp(1e-4, 1 - 1e-4)
            out[sl] = torch.log(pn / (1 - pn)).cpu().numpy()
        state["vec"] = None
        return out

    base = run(None)
    res = {"layer": int(layer), "n_prompts": len(steer_idx), "direction_norm": float(direction.norm()),
           "mean_residual_norm": resid_norm, "predicted_slope": 1.0, "value_direction": {}, "random_direction": {}}
    for a in S["strengths"]:
        d_ = run(direction * a) - base if a != 0 else np.zeros_like(base)
        res["value_direction"][str(a)] = {"mean_shift": float(d_.mean()), "sd": float(d_.std())}
        log(f"  value direction x {a:+}: log-odds of A shift {d_.mean():+.3f}")
    for a in (min(S["strengths"]), max(S["strengths"])):
        d_ = run(rand * a) - base
        res["random_direction"][str(a)] = {"mean_shift": float(d_.mean()), "sd": float(d_.std())}
        log(f"  random direction x {a:+}: log-odds of A shift {d_.mean():+.3f}")
    handle.remove()
    xs = np.array([float(k) for k in res["value_direction"]])
    ys = np.array([v["mean_shift"] for v in res["value_direction"].values()])
    res["observed_slope"] = float(np.polyfit(xs, ys, 1)[0])
    xr = np.array([float(k) for k in res["random_direction"]])
    yr = np.array([v["mean_shift"] for v in res["random_direction"].values()])
    res["random_slope"] = float((yr[-1] - yr[0]) / (xr[-1] - xr[0]))
    log(f"Slope: value direction {res['observed_slope']:.3f} (predicted about 1), random {res['random_slope']:.3f}")
    json.dump(res, open(f"{cfg['res']}/steering.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(xs, ys, "o-", label="value direction")
    ax.plot(xr, yr, "s--", label="random direction, same length")
    ax.plot(xs, xs, ":", color="grey", label="predicted (slope 1)")
    ax.set_xlabel("value added to option A (probe units)")
    ax.set_ylabel("shift in log-odds of choosing A")
    ax.set_title(f"Steering at layer {layer}, on option A's tokens", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(f"{cfg['res']}/fig_steering.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
