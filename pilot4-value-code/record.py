"""Pilot 4, step 1: run every prompt through the model once and record its insides.

  python record.py            (Qwen3-14B, all ~16,000 prompts; about 15-25 minutes on an A6000)
  python record.py --random   (the control: same architecture, random weights, test prompts only)

Writes to acts_dir (the network volume):
  resid.npy        residual stream, (prompts, layers+1, positions, width), float16
                   positions: see config.yaml; layer 0 = embeddings, layer L = after the final norm
  resid_full.npy   every token of the 500 "full" prompts, (tokens, layers+1, width), float16
  meta.npz         answer probabilities, token positions, lengths, the full-prompt index and token regions
  neurons.npz      running sums for every MLP neuron at the neuron positions (test prompts only):
                   n, sum a, sum a^2, sum a*y for y in NEURON_VARS
"""
from __future__ import annotations

import json

import numpy as np

from common import (REGIONS, acts_path, char_landmarks, label_arrays, load_config, load_prompts, log, parse_args,
                    positions_for, token_regions)

NEURON_VARS = ["v_a", "v_b", "chosen_value", "choice", "dv", "log_qa", "log_qb"]


def main():
    args = parse_args(__doc__)
    cfg = load_config(args)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_prompts(cfg)
    N, names = len(rows), cfg["positions"]
    P = len(names)
    npos = [names.index(n) for n in cfg["neuron_positions"]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading {cfg['model']} on {dev} ...")
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    dtype = getattr(torch, cfg["dtype"]) if dev == "cuda" else torch.float32
    if cfg["random"]:
        # The control: identical architecture and tokenizer, weights drawn at random (seeded).
        from transformers import AutoConfig
        torch.manual_seed(20261031)
        mcfg = AutoConfig.from_pretrained(cfg["model"])
        try:
            model = AutoModelForCausalLM.from_config(mcfg, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_config(mcfg, torch_dtype=dtype)
        log("Random-weights control model built from the config")
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=dtype)
    model = model.to(dev).eval()
    n_layers, d = model.config.num_hidden_layers, model.config.hidden_size
    L = n_layers + 1
    letter_ids = []
    for letter in "AB":
        ids = {e[0] for v in (letter, " " + letter) if len(e := tok.encode(v, add_special_tokens=False)) == 1}
        letter_ids.append(sorted(ids))
    log(f"{N} prompts, {n_layers} layers, width {d}, letter tokens {letter_ids}")

    # Tokenise everything first: positions, lengths, regions of the full prompts.
    enc_ids, positions, regions_full = [], np.zeros((N, P), np.int32), {}
    for i, r in enumerate(rows):
        msgs = ([{"role": "system", "content": r["system"]}] if r["system"] else []) + [{"role": "user", "content": r["user"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        marks = char_landmarks(text, r["user"])
        positions[i] = positions_for(enc["offset_mapping"], marks, len(enc["input_ids"]), names)
        enc_ids.append(enc["input_ids"])
        if r["full"]:
            regions_full[i] = token_regions(enc["offset_mapping"], marks)
        if i < 3:
            shown = {nm: tok.decode([enc["input_ids"][p]]) for nm, p in zip(names, positions[i])}
            log(f"  positions of prompt {i}: {shown}")
    lengths = np.array([len(x) for x in enc_ids])
    full_rows = sorted(regions_full)
    full_off = np.zeros(len(full_rows) + 1, np.int64)
    full_off[1:] = np.cumsum([lengths[i] for i in full_rows])

    resid = np.lib.format.open_memmap(acts_path(cfg, "resid.npy"), mode="w+", dtype=np.float16, shape=(N, L, P, d))
    resid_full = np.lib.format.open_memmap(acts_path(cfg, "resid_full.npy"), mode="w+", dtype=np.float16,
                                           shape=(int(full_off[-1]), L, d))
    log(f"Allocated {resid.nbytes / 1e9:.1f} GB + {resid_full.nbytes / 1e9:.1f} GB in {cfg['acts_dir']}")

    # MLP neurons: capture the input of every down_proj (the neuron activations) at the neuron positions.
    layers = model.model.layers
    n_neur = layers[0].mlp.down_proj.in_features
    state = {"gather": None, "captured": {}}

    def make_hook(li):
        def hook(module, inputs):
            if state["gather"] is not None:
                x = inputs[0]
                b = torch.arange(x.shape[0], device=x.device)[:, None]
                state["captured"][li] = x[b, state["gather"]].float()
        return hook

    hooks = [layer.mlp.down_proj.register_forward_pre_hook(make_hook(li)) for li, layer in enumerate(layers)]
    K = len(NEURON_VARS)
    S1 = torch.zeros(n_layers, len(npos), n_neur, dtype=torch.float64, device=dev)
    S2 = torch.zeros_like(S1)
    SY = torch.zeros(n_layers, len(npos), n_neur, K, dtype=torch.float64, device=dev)
    lab = label_arrays(rows)
    # Per-good sums at the option positions (for the monkey criterion: a value neuron must follow
    # the number of people within a good AND the value across goods).
    keys = sorted({r["key_a"] for r in rows if r["group"] == "test"} | {r["key_b"] for r in rows if r["group"] == "test"})
    kidx = {k: i for i, k in enumerate(keys)}
    opt_pos = [j for j, p in enumerate(cfg["neuron_positions"]) if p[0] in "ab"]          # a_num, a_end, b_num, b_end
    slot_of = {j: cfg["neuron_positions"][j][0] for j in opt_pos}
    KS = torch.zeros(n_layers, len(opt_pos), n_neur, len(keys), 3, dtype=torch.float64, device=dev)  # sum a, a^2, a*logq
    p_a, p_b = np.zeros(N), np.zeros(N)
    y_rows = np.zeros((N, K))

    order = sorted(range(N), key=lambda i: (rows[i]["group"] != "test", lengths[i]))   # test prompts first
    bs, t0 = int(cfg["batch_size"]), __import__("time").time()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    full_pos = {i: k for k, i in enumerate(full_rows)}
    for k0 in range(0, N, bs):
        idx = order[k0:k0 + bs]
        T = max(lengths[i] for i in idx)
        ids = torch.full((len(idx), T), pad_id, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        pads = []
        for j, i in enumerate(idx):
            pad = T - lengths[i]
            ids[j, pad:] = torch.tensor(enc_ids[i])
            att[j, pad:] = 1
            pads.append(pad)
        pads_t = torch.tensor(pads)
        gpos = torch.tensor(positions[idx]) + pads_t[:, None]                     # (B, P) in padded coordinates
        is_test = [rows[i]["group"] == "test" for i in idx]
        state["gather"] = gpos[:, npos].to(dev) if any(is_test) else None
        state["captured"] = {}
        with torch.no_grad():
            kw = dict(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True)
            try:
                out = model(**kw, logits_to_keep=1)
            except TypeError:
                out = model(**kw)
        probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
        pa = probs[:, letter_ids[0]].sum(-1).cpu().numpy()
        pb = probs[:, letter_ids[1]].sum(-1).cpu().numpy()
        p_a[idx], p_b[idx] = pa, pb
        bidx = torch.arange(len(idx))[:, None]
        g = gpos.to(dev)
        stacked = torch.stack([h[bidx.to(dev), g] for h in out.hidden_states], dim=1)   # (B, L, P, d)
        resid[idx] = stacked.to(torch.float16).cpu().numpy()
        for j, i in enumerate(idx):
            if i in full_pos:
                k = full_pos[i]
                tok_states = torch.stack([h[j, pads[j]:] for h in out.hidden_states], dim=1)   # (T_i, L, d)
                resid_full[full_off[k]:full_off[k + 1]] = tok_states.to(torch.float16).cpu().numpy()
        if state["gather"] is not None:
            sel = [j for j, t in enumerate(is_test) if t]
            pa_n = pa[sel] / np.maximum(pa[sel] + pb[sel], 1e-9)
            pa_c = np.clip(pa_n, 1e-4, 1 - 1e-4)
            ii = [idx[j] for j in sel]
            Y = np.stack([lab["v_a"][ii], lab["v_b"][ii], pa_n * lab["v_a"][ii] + (1 - pa_n) * lab["v_b"][ii],
                          np.log(pa_c / (1 - pa_c)), lab["dv"][ii], lab["log_qa"][ii], lab["log_qb"][ii]], axis=1)
            y_rows[ii] = Y
            Yt = torch.tensor(Y, dtype=torch.float64, device=dev)
            sel_t = torch.tensor(sel, device=dev)
            onehot = {}
            for s_ in "ab":
                kk = [kidx[rows[i]["key_" + s_]] for i in ii]
                oh = torch.zeros(len(ii), len(keys), dtype=torch.float64, device=dev)
                oh[torch.arange(len(ii)), torch.tensor(kk)] = 1
                q = torch.tensor(lab["log_q" + s_][ii], dtype=torch.float64, device=dev)
                onehot[s_] = (oh, oh * q[:, None])
            for li, a in state["captured"].items():
                a = a[sel_t].double()                                    # (b, Pn, I)
                S1[li] += a.sum(0)
                S2[li] += (a * a).sum(0)
                SY[li] += torch.einsum("bpi,bk->pik", a, Yt)
                for oj, pj in enumerate(opt_pos):
                    oh, ohq = onehot[slot_of[pj]]
                    ap = a[:, pj]                                        # (b, I)
                    KS[li, oj, :, :, 0] += ap.T @ oh
                    KS[li, oj, :, :, 1] += (ap * ap).T @ oh
                    KS[li, oj, :, :, 2] += ap.T @ ohq
        done = k0 + len(idx)
        if (k0 // bs) % 50 == 0 or done == N:
            rate = done / max(__import__("time").time() - t0, 1e-6)
            log(f"  {done}/{N} prompts ({rate:.1f}/s, about {(N - done) / max(rate, 1e-6) / 60:.0f} min left)")
    for h in hooks:
        h.remove()
    resid.flush()
    resid_full.flush()
    test_mask = np.array([r["group"] == "test" for r in rows])
    trows = [r for r in rows if r["group"] == "test"]
    key_stats = {s_: np.array([[sum(1 for r in trows if r["key_" + s_] == k),
                                sum(np.log(r["q_" + s_]) for r in trows if r["key_" + s_] == k),
                                sum(np.log(r["q_" + s_]) ** 2 for r in trows if r["key_" + s_] == k)] for k in keys])
                 for s_ in "ab"}
    np.savez(acts_path(cfg, "neurons.npz"), n=int(test_mask.sum()), s1=S1.cpu().numpy(), s2=S2.cpu().numpy(),
             sy=SY.cpu().numpy(), y=y_rows[test_mask], vars=np.array(NEURON_VARS),
             positions=np.array(cfg["neuron_positions"]), key_sums=KS.float().cpu().numpy(), keys=np.array(keys),
             key_positions=np.array([cfg["neuron_positions"][j] for j in opt_pos]),
             key_stats_a=key_stats["a"], key_stats_b=key_stats["b"])
    regions = np.concatenate([regions_full[i] for i in full_rows]) if full_rows else np.zeros(0, np.int16)
    np.savez(acts_path(cfg, "meta.npz"), p_a=p_a, p_b=p_b, positions=positions, lengths=lengths,
             full_rows=np.array(full_rows), full_offsets=full_off, full_regions=regions,
             region_names=np.array(REGIONS), position_names=np.array(names), n_layers=n_layers, width=d,
             model=cfg["model"])
    json.dump({"ids": [r["id"] for r in rows]}, open(acts_path(cfg, "row_ids.json"), "w"))
    mass = p_a + p_b
    log(f"Done. Probability on A/B: mean {mass.mean():.3f}, min {mass.min():.3f}. Files in {cfg['acts_dir']}")


if __name__ == "__main__":
    main()
