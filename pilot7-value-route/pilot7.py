"""Pilot 7: where does the model form the values it puts through the decision stage?

  python pilot7.py                         both models, all stages
  python pilot7.py --models qwen           one model
  python pilot7.py --quick                 Qwen3-0.6B and a few pairs (a pipeline test)

Stages, per model (DESIGN.md):
  fit     the model's own values: its logit difference fitted as u(A's good) - u(B's good) + b log(nA / nB)
  map     patching one site (a part of the prompt) at one layer, donor into recipient, for four kinds
          of swap: A's good, A's number of people, A's number of apples, A's object (weight)
  das     the monkey test at the site chosen by a pre-registered rule: one direction trained on GOOD
          swaps only, tested on NUMBER swaps it never saw (one currency?), on the exchange rate between
          the two, and on apple counts (value or just magnitude?)
  report  verdicts and report.md
The answer is the logit difference logit("A") - logit("B"). Transfer = slope through zero.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
P5, P6 = HERE.parent / "pilot5-value-circuit", HERE.parent / "pilot6-value-subspace"
sys.path.insert(0, str(P5))
from common import (Hooks, batches, deep_update, forward, layer_output, load_model, log,  # noqa: E402
                    main_letter_ids, model_shape, replace_layer_output, save_json)

SITES_ALL = ["A_end", "A_nl", "newlines", "B_line", "B_end", "B_nl", "reply", "template", "after_B", "between", "after_A", "last"]


def through_zero(obs, eff, reps=1000, seed=0):
    obs, eff = np.asarray(obs, float), np.asarray(eff, float)
    den = (eff * eff).sum()
    if den == 0:
        return {"value": float("nan"), "ci95": [float("nan")] * 2}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(reps):
        i = rng.integers(0, len(obs), len(obs))
        d = (eff[i] * eff[i]).sum()
        bs.append((obs[i] * eff[i]).sum() / d if d > 0 else np.nan)
    return {"value": float((obs * eff).sum() / den),
            "ci95": [float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))]}


# --------------------------------------------------------------------------
# Settings, data, sites
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", default=None, help="comma-separated names from config.models")
    ap.add_argument("--stages", default="fit,map,das,report")
    ap.add_argument("--results-dir", default=None)
    return ap.parse_args()


def load_cfg(args):
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.quick:
        q = cfg.get("quick", {})
        cfg = deep_update(cfg, {k: v for k, v in q.items() if k != "models"})
        cfg["models"] = q["models"]
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    cfg["quick_mode"] = bool(args.quick)
    res = Path(cfg["results_dir"])
    cfg["res"] = str(res if res.is_absolute() else HERE / res)
    return cfg


def chat_text(tok, user):
    msgs = [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode_sites(tok, user):
    """Token ids and the sites, as negative positions (-1 = the last token). Everything after option A's
    line is the same text in a recipient and its donor, so negative positions line up between them."""
    s = chat_text(tok, user)
    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    u0 = s.index(user)
    spans, pos = {}, u0
    for ln in user.split("\n"):
        key = "A" if ln.startswith("A) ") else "B" if ln.startswith("B) ") else "reply" if ln.startswith("Reply") else None
        if key:
            spans[key] = (pos, pos + len(ln))
        pos += len(ln) + 1
    n = len(ids)

    def toks(a, b):
        return [i for i, (x, y) in enumerate(offs) if y > a and x < b and y > x]
    a_end = max(toks(*spans["A"]))
    B, R = toks(*spans["B"]), toks(*spans["reply"])
    r_end = max(R)
    sites = {"A_end": [a_end], "A_nl": list(range(a_end + 1, min(B))), "B_line": B, "B_end": [max(B)],
             "B_nl": list(range(max(B) + 1, min(R))), "reply": R,
             "newlines": [i for i in range(a_end + 1, r_end + 1) if i not in B and i not in R],
             "template": list(range(r_end + 1, n - 1)), "after_A": list(range(a_end + 1, n)), "last": [n - 1],
             "between": list(range(a_end + 1, n - 1)),            # everything after A's line except the answer position
             "after_B": list(range(max(B) + 1, n - 1))}           # everything after B's line except the answer position
    return {"ids": ids, "a_end": a_end, "b_span": (min(B), max(B)), "sites": {k: [i - n for i in v] for k, v in sites.items()}}


class Data:
    def __init__(self, cfg):
        self.cfg = cfg
        lim = cfg.get("limit") or {}
        rng = np.random.default_rng(0)
        pairs = [json.loads(l) for l in open(HERE / "pairs.jsonl", encoding="utf-8")]
        self.sets = {}
        for s in ("good_train", "number_train", "good_test", "number_test", "bnumber_test", "quantity", "weight"):
            ps = [p for p in pairs if p["set"] == s]
            if s in lim:
                ps = [ps[i] for i in sorted(rng.choice(len(ps), size=min(lim[s], len(ps)), replace=False))]
            self.sets[s] = ps


class Model:
    """One model with its tokenizer, and the pairs encoded and checked for it."""
    def __init__(self, cfg, name, data):
        self.cfg, self.name = cfg, name
        mc = cfg["models"][name]
        self.tok, self.model, self.dev = load_model({"model": mc["path"], "dtype": cfg["dtype"]})
        self.layers = self.model.model.layers
        self.sh = model_shape(self.model)
        self.ia, self.ib = main_letter_ids(self.tok)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.map_layers = [L for L in mc["map_layers"] if L <= len(self.layers)]
        self.res = Path(cfg["res"]) / name
        self.res.mkdir(parents=True, exist_ok=True)
        self._enc = {}
        self.pairs, dropped = {}, {}
        for s, ps in data.sets.items():
            keep = []
            for p in ps:
                er, ed = self.enc(p["user"]), self.enc(p["donor_user"])
                if p.get("swap") == "bnumber":          # B's line changes: same length, the same outside B's line
                    b0, b1 = er["b_span"]
                    ok = len(er["ids"]) == len(ed["ids"]) and er["ids"][:b0] == ed["ids"][:b0] and er["ids"][b1 + 1:] == ed["ids"][b1 + 1:]
                else:                                   # A's line changes: the same after A's line
                    ok = er["ids"][er["a_end"] + 1:] == ed["ids"][ed["a_end"] + 1:]
                if ok:
                    keep.append(p)
            self.pairs[s], dropped[s] = keep, len(ps) - len(keep)
        log(f"[{name}] pairs (dropped where the text after A does not tokenize the same): "
            + ", ".join(f"{s} {len(v)} (-{dropped[s]})" for s, v in self.pairs.items()))

    def enc(self, user):
        if user not in self._enc:
            self._enc[user] = encode_sites(self.tok, user)
        return self._enc[user]

    def batch(self, users):
        import torch
        es = [self.enc(u) for u in users]
        T = max(len(e["ids"]) for e in es)
        ids = torch.full((len(es), T), self.pad, dtype=torch.long)
        att = torch.zeros((len(es), T), dtype=torch.long)
        for j, e in enumerate(es):
            ids[j, T - len(e["ids"]):] = torch.tensor(e["ids"])
            att[j, T - len(e["ids"]):] = 1
        return ids.to(self.dev), att.to(self.dev), es

    def ydiff(self, o):
        lg = o.logits[:, -1].float()
        return (lg[:, self.ia] - lg[:, self.ib]).detach().cpu().numpy()

    def site_index(self, es, site):
        """(rows, negative positions) of a site over a batch."""
        import torch
        rows, ks = [], []
        for j, e in enumerate(es):
            for k in e["sites"][site]:
                rows.append(j)
                ks.append(k)
        return torch.tensor(rows, device=self.dev), torch.tensor(ks, device=self.dev)


def patch_hook(rows, ks, src, W=None):
    """Replace the block output at (rows, T + ks) by src (the donor's), or only its part in the span of W's rows."""
    def hook(m, i, o):
        h = layer_output(o)
        T = h.shape[1]
        hn = h.clone()
        cur = h[rows, T + ks].float()
        new = src.float() if W is None else cur + ((src.float() - cur) @ W.T) @ W
        hn[rows, T + ks] = new.to(h.dtype)
        return replace_layer_output(o, hn)
    return hook


# --------------------------------------------------------------------------
# Stage: fit (the model's own values)
# --------------------------------------------------------------------------
def stage_fit(M):
    import torch
    cfg = M.cfg
    out = {}
    for split in ("good_train", "good_test"):
        ps = M.pairs[split]
        y = np.zeros(len(ps))
        with torch.no_grad():
            for ch in batches(len(ps), int(cfg["batch_size"])):
                ids, att, _ = M.batch([ps[i]["user"] for i in ch])
                y[ch] = M.ydiff(forward(M.model, ids, att, torch))
        out[split] = y
    keys = sorted({p["key_a"] for s in ("good_train", "good_test") for p in M.pairs[s]} |
                  {p["key_b"] for s in ("good_train", "good_test") for p in M.pairs[s]} |
                  {p["donor_key_a"] for s in ("good_train", "good_test") for p in M.pairs[s]})
    ref = "G_dental"
    free = [k for k in keys if k != ref]

    def design(ps):
        X = np.zeros((len(ps), len(free) + 2))
        for i, p in enumerate(ps):
            if p["key_a"] in free:
                X[i, free.index(p["key_a"])] += 1
            if p["key_b"] in free:
                X[i, free.index(p["key_b"])] -= 1
            X[i, -2] = math.log(p["q_a"]) - math.log(p["q_b"])
            X[i, -1] = 1.0
        return X
    Xtr, ytr = design(M.pairs["good_train"]), out["good_train"]
    lam = float(cfg["fit"]["ridge"]) * np.eye(Xtr.shape[1])
    lam[-1, -1] = lam[-2, -2] = 0
    beta = np.linalg.solve(Xtr.T @ Xtr + lam, Xtr.T @ ytr)
    u = {ref: 0.0, **{k: float(beta[i]) for i, k in enumerate(free)}}
    b, bias = float(beta[-2]), float(beta[-1])
    pred = design(M.pairs["good_test"]) @ beta
    r_test = float(np.corrcoef(pred, out["good_test"])[0, 1])
    r_train = float(np.corrcoef(Xtr @ beta, ytr)[0, 1])
    q6 = json.load(open(HERE / "values.json"))
    common = [k for k in u if k in q6["u"]]
    r_pilot3 = float(np.corrcoef([u[k] for k in common], [q6["u"][k] for k in common])[0, 1]) if len(common) > 2 else None
    usable = r_test >= cfg["prereg"]["usable_min_corr"] and b > 0
    fit = {"u": u, "b": b, "bias_towards_A": bias, "corr_train": r_train, "corr_heldout": r_test,
           "corr_u_with_pilot3": r_pilot3, "usable": bool(usable), "n_train": len(ytr),
           "mean_abs_logit_diff": float(np.abs(out["good_test"]).mean())}
    save_json(fit, M.res / "fit.json")
    log(f"  [{M.name}] fit: b = {b:.2f} (logit units per log person), held-out r = {r_test:.2f}, "
        f"u vs pilot 3 r = {r_pilot3}, usable = {usable}")


def values_for(M):
    f = json.load(open(M.res / "fit.json"))
    u, b = f["u"], f["b"]

    def dv(p):
        if p["swap"] == "good":
            return u.get(p["donor_key_a"], np.nan) - u.get(p["key_a"], np.nan)
        if p["swap"] == "number":
            return b * (math.log(p["donor_q_a"]) - math.log(p["q_a"]))
        if p["swap"] == "bnumber":                       # the change of B's value
            return b * (math.log(p["donor_q_b"]) - math.log(p["q_b"]))
        return np.nan
    return dv


# --------------------------------------------------------------------------
# Stage: map (site x layer patching)
# --------------------------------------------------------------------------
MAP_SETS = {"good": "good_test", "number": "number_test", "bnumber": "bnumber_test", "quantity": "quantity", "weight": "weight"}


def stage_map(M):
    import torch
    cfg = M.cfg
    sites = cfg["map"]["sites"]
    Ls = M.map_layers
    out = {"layers": Ls, "sites": sites, "swaps": {}}
    for swap, split in MAP_SETS.items():
        ps = M.pairs[split]
        rng = np.random.default_rng(int(cfg["map"]["seed"]))
        if len(ps) > cfg["map"]["n_pairs"]:
            ps = [ps[i] for i in sorted(rng.choice(len(ps), size=int(cfg["map"]["n_pairs"]), replace=False))]
        n = len(ps)
        y_r, y_d = np.zeros(n), np.zeros(n)
        obs = np.zeros((len(sites), len(Ls), n))
        for ch in batches(n, int(cfg["batch_size"])):
            with torch.no_grad():
                cap = {}
                with Hooks() as hk:
                    for L in Ls:
                        hk.fwd(M.layers[L - 1], (lambda L: lambda m, i, o: cap.__setitem__(L, layer_output(o)))(L))
                    idd, attd, _ = M.batch([ps[i]["donor_user"] for i in ch])
                    y_d[ch] = M.ydiff(forward(M.model, idd, attd, torch))
                Td = idd.shape[1]
                ids, att, es = M.batch([ps[i]["user"] for i in ch])
                y_r[ch] = M.ydiff(forward(M.model, ids, att, torch))
                for si, site in enumerate(sites):
                    rows, ks = M.site_index(es, site)
                    if len(rows) == 0:
                        obs[si, :, ch] = 0
                        continue
                    for li, L in enumerate(Ls):
                        src = cap[L][rows, Td + ks]
                        with Hooks() as hk:
                            hk.fwd(M.layers[L - 1], patch_hook(rows, ks, src))
                            obs[si, li, ch] = M.ydiff(forward(M.model, ids, att, torch)) - y_r[ch]
        eff = y_d - y_r
        rec = {"n_pairs": n, "mean_abs_effect": float(np.abs(eff).mean()), "transfer": {}}
        for si, site in enumerate(sites):
            rec["transfer"][site] = {str(L): through_zero(obs[si, li], eff) for li, L in enumerate(Ls)}
        out["swaps"][swap] = rec
        best = max(((s, L) for s in sites if s not in ("after_A", "last") for L in Ls),
                   key=lambda t: rec["transfer"][t[0]][str(t[1])]["value"])
        log(f"  [{M.name}] map {swap}: {n} pairs, mean |effect| {rec['mean_abs_effect']:.2f}; "
            f"best upstream site {best[0]}@{best[1]}")
    save_json(out, M.res / "map.json")


def choose_sites(M):
    """The pre-registered rule (DESIGN.md, section 4)."""
    cfg = M.cfg
    mp = json.load(open(M.res / "map.json"))
    T = mp["swaps"]["good"]["transfer"]
    Ls = mp["layers"]
    arrival = next((L for L in Ls if T["last"][str(L)]["value"] >= cfg["prereg"]["arrival_min"]), Ls[-1])
    best = lambda cands: max(cands, key=lambda t: (T[t[0]][str(t[1])]["value"], -t[1]))
    up_to = [L for L in Ls if L <= arrival]
    before = [L for L in Ls if L < arrival] or Ls[:1]
    a_site = best([(s, L) for s in cfg["map"]["a_sites"] for L in up_to])
    b_site = best([(s, L) for s in cfg["map"]["b_sites"] for L in before])
    return {"arrival_layer": arrival, "a_site": list(a_site), "b_site": list(b_site),
            "a_site_share": T[a_site[0]][str(a_site[1])]["value"], "b_site_share": T[b_site[0]][str(b_site[1])]["value"]}


# --------------------------------------------------------------------------
# Stage: das (the monkey test)
# --------------------------------------------------------------------------
def prep_site(M, ps, site, L):
    """Per pair: recipient and donor states at the site (layer L), and the answers: recipient, donor,
    and recipient with the whole site swapped."""
    import torch
    cfg = M.cfg
    n = len(ps)
    hs_r, hs_d = [None] * n, [None] * n
    y_r, y_d, y_s = np.zeros(n), np.zeros(n), np.zeros(n)
    with torch.no_grad():
        for ch in batches(n, int(cfg["batch_size"])):
            cap = {}
            with Hooks() as hk:
                hk.fwd(M.layers[L - 1], lambda m, i, o: cap.__setitem__("d", layer_output(o)))
                idd, attd, ed = M.batch([ps[i]["donor_user"] for i in ch])
                y_d[ch] = M.ydiff(forward(M.model, idd, attd, torch))
            with Hooks() as hk:
                hk.fwd(M.layers[L - 1], lambda m, i, o: cap.__setitem__("r", layer_output(o)))
                ids, att, es = M.batch([ps[i]["user"] for i in ch])
                y_r[ch] = M.ydiff(forward(M.model, ids, att, torch))
            Td, Tr = idd.shape[1], ids.shape[1]
            rows, ks = M.site_index(es, site)
            src = cap["d"][rows, Td + ks]
            with Hooks() as hk:
                hk.fwd(M.layers[L - 1], patch_hook(rows, ks, src))
                y_s[ch] = M.ydiff(forward(M.model, ids, att, torch))
            for j, i in enumerate(ch):
                kk = torch.tensor(es[j]["sites"][site], device=M.dev)
                hs_d[i] = cap["d"][j, Td + kk].to(torch.bfloat16)
                hs_r[i] = cap["r"][j, Tr + kk].to(torch.bfloat16)
    return {"ps": ps, "hd": hs_d, "hr": hs_r, "y_r": y_r, "y_d": y_d, "y_s": y_s}


def das_train(M, D, site, L, seed):
    import torch
    cfg = M.cfg
    C = cfg["das"]
    d = M.sh["d"]
    torch.manual_seed(seed)
    lin = torch.nn.Linear(d, int(C["k"]), bias=False).to(M.dev)
    torch.nn.utils.parametrizations.orthogonal(lin)
    opt = torch.optim.Adam(lin.parameters(), lr=float(C["lr"]))
    tgt = torch.tensor(D["y_s"], dtype=torch.float32, device=M.dev)
    ps, n = D["ps"], len(D["ps"])
    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(int(C["epochs"])):
        perm = rng.permutation(n)
        losses = []
        for s0 in range(0, n, int(C["batch_size"])):
            ch = perm[s0:s0 + int(C["batch_size"])].tolist()
            ids, att, es = M.batch([ps[i]["user"] for i in ch])
            rows, ks = M.site_index(es, site)
            src = torch.cat([D["hd"][i] for i in ch], 0)
            with Hooks() as hk:
                hk.fwd(M.layers[L - 1], patch_hook(rows, ks, src, lin.weight))
                try:
                    o = M.model(input_ids=ids, attention_mask=att, logits_to_keep=1)
                except TypeError:
                    o = M.model(input_ids=ids, attention_mask=att)
            lg = o.logits[:, -1].float()
            loss = ((lg[:, M.ia] - lg[:, M.ib] - tgt[ch]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        hist.append(float(np.mean(losses)))
    return lin.weight.detach().float(), hist


def das_eval(M, D, site, L, W):
    import torch
    ps, n = D["ps"], len(D["ps"])
    y = np.zeros(n)
    with torch.no_grad():
        for ch in batches(n, int(M.cfg["batch_size"])):
            ids, att, es = M.batch([ps[i]["user"] for i in ch])
            rows, ks = M.site_index(es, site)
            src = torch.cat([D["hd"][i] for i in ch], 0)
            with Hooks() as hk:
                hk.fwd(M.layers[L - 1], patch_hook(rows, ks, src, W))
                y[ch] = M.ydiff(forward(M.model, ids, att, torch))
    yr, ys, yd = D["y_r"], D["y_s"], D["y_d"]
    w = W[0].cpu().numpy()
    dc = np.array([float(((D["hd"][i].float() - D["hr"][i].float()).cpu().numpy() @ w).mean()) for i in range(n)])
    return {"of_site_swap": through_zero(y - yr, ys - yr), "of_donor": through_zero(y - yr, yd - yr),
            "site_swap_of_donor": through_zero(ys - yr, yd - yr), "coord_change": dc.tolist(),
            "mean_abs_site_effect": float(np.abs(ys - yr).mean())}


def stage_das(M):
    import torch
    cfg = M.cfg
    ch = choose_sites(M)
    dv = values_for(M)
    fit = json.load(open(M.res / "fit.json"))
    out = {"choice": ch, "usable": fit["usable"], "sites": {}}
    log(f"  [{M.name}] sites chosen")
    todo = [("a_site", tuple(ch["a_site"])), ("b_site", tuple(ch["b_site"]))]
    learned = {}
    mx = int(cfg["das"]["max_train_pairs"])
    for role, (site, L) in todo:
        t0 = time.time()
        rec = {"site": site, "layer": L, "train": {}, "eval": {}}
        sub = lambda ps: [ps[i] for i in sorted(np.random.default_rng(int(cfg["das"]["seed"])).choice(len(ps), size=min(mx, len(ps)), replace=False))]
        Ds = {s: prep_site(M, sub(M.pairs[s]) if s.endswith("train") else M.pairs[s], site, L)
              for s in ("good_train", "number_train", "good_test", "number_test", "bnumber_test", "quantity", "weight")}
        Ws = {}
        for name, split in (("good_trained", "good_train"), ("number_trained", "number_train")):
            W, hist = das_train(M, Ds[split], site, L, int(cfg["das"]["seed"]))
            Ws[name] = W
            rec["train"][name] = {"loss_by_epoch": hist}
            learned[f"{role}|{site}|{L}|{name}"] = W.cpu().numpy()
        for name, W in Ws.items():
            ev = {}
            for split in ("good_test", "number_test", "bnumber_test", "quantity", "weight"):
                e = das_eval(M, Ds[split], site, L, W)
                if split in ("good_test", "number_test", "bnumber_test"):
                    v = np.array([dv(p) for p in Ds[split]["ps"]])
                    c = np.array(e["coord_change"])
                    ok = np.isfinite(v)
                    e["coord_per_value_unit"] = float(np.polyfit(v[ok], c[ok], 1)[0]) if ok.sum() > 2 else float("nan")
                    e["coord_corr_with_value_change"] = float(np.corrcoef(v[ok], c[ok])[0, 1]) if ok.sum() > 2 else float("nan")
                e.pop("coord_change")
                ev[split] = e
            rec["eval"][name] = ev
        a, b_ = Ws["good_trained"][0], Ws["number_trained"][0]
        rec["cos_good_vs_number_trained"] = float(torch.abs(a @ b_) / a.norm() / b_.norm())
        out["sites"][role] = rec
        g = rec["eval"]["good_trained"]
        log(f"  [{M.name}] das {role} {site}@{L}: done in {(time.time() - t0) / 60:.1f} min")
        _ = g
    np.savez(M.res / "directions.npz", **learned)
    save_json(out, M.res / "das.json")


# --------------------------------------------------------------------------
# Stage: report
# --------------------------------------------------------------------------
def verdicts_for(cfg, das):
    pr = cfg["prereg"]
    out = {}
    for role, rec in das["sites"].items():
        g = rec["eval"]["good_trained"]
        good, num = g["good_test"]["of_site_swap"]["value"], g["number_test"]["of_site_swap"]["value"]
        v2 = bool(good >= pr["one_currency_min"] and num >= pr["one_currency_min"])
        cg, cn = g["good_test"]["coord_per_value_unit"], g["number_test"]["coord_per_value_unit"]
        ratio = cn / cg if np.isfinite(cg) and np.isfinite(cn) and cg != 0 else float("nan")
        v3 = bool(np.isfinite(ratio) and 1 / pr["exchange_rate_factor"] <= ratio <= pr["exchange_rate_factor"])
        qsite, qdir = g["quantity"]["site_swap_of_donor"]["value"], g["quantity"]["of_site_swap"]["value"]
        if not np.isfinite(qsite) or qsite < pr["magnitude_site_min"]:
            v4 = "not testable"
        else:
            v4 = "value-specific" if qdir < pr["magnitude_max"] else "generic magnitude"
        bsite, bdir = g["bnumber_test"]["site_swap_of_donor"]["value"], g["bnumber_test"]["of_site_swap"]["value"]
        cb = g["bnumber_test"]["coord_per_value_unit"]
        rho = cb / cg if np.isfinite(cb) and np.isfinite(cg) and cg != 0 else float("nan")
        if role == "a_site":
            v5 = "A's own value (the site comes before B's line)"
        elif not np.isfinite(bsite) or bsite < pr["decision_site_min"] or not np.isfinite(rho):
            v5 = "not testable"
        elif rho <= -pr["decision_rho"]:
            v5 = "the comparison (A minus B)"
        elif rho >= pr["decision_rho"]:
            v5 = "a value code shared by both options"
        else:
            v5 = "A's own value"
        share = das["choice"][f"{role}_share"]
        floor_ok = role != "a_site" or share >= pr["a_site_floor"]
        if not floor_ok:
            head = "A's information is not concentrated here (below the floor)"
        elif v2 and v3:
            head = "one currency" + ("" if role == "a_site" else f"; V5: {v5}")
        else:
            head = "no single value here"
        out[role] = {"site": rec["site"], "layer": rec["layer"], "site_share_of_good_swap": share, "headline": head,
                     "V2_one_currency": v2, "good_share": good, "number_share": num,
                     "V3_exchange_rate": v3, "exchange_ratio": ratio,
                     "V4": v4, "quantity_share": qdir, "quantity_site_share": qsite,
                     "V5": v5, "b_to_a_coord_ratio": rho, "bnumber_share": bdir, "bnumber_site_share": bsite}
    return out


def stage_report(cfg, names):
    lines = ["# Pilot 7 results", ""]
    allv = {}
    for name in names:
        R = Path(cfg["res"]) / name
        if not (R / "fit.json").exists():
            continue
        fit = json.load(open(R / "fit.json"))
        lines += [f"## {name} ({cfg['models'][name]['path']})", "",
                  f"**Its own values.** b = {fit['b']:.2f} logit units per log person; the fit predicts held-out "
                  f"choices with r = {fit['corr_heldout']:.2f} (train {fit['corr_train']:.2f}); its goods' values "
                  f"correlate {fit['corr_u_with_pilot3'] if fit['corr_u_with_pilot3'] is None else round(fit['corr_u_with_pilot3'], 2)} "
                  f"with Qwen3-14B's from pilot 3. Usable for V2-V4: **{fit['usable']}**.", ""]
        if (R / "map.json").exists():
            mp = json.load(open(R / "map.json"))
            Ls = mp["layers"]
            for swap, rec in mp["swaps"].items():
                lines += [f"**Map, {swap} swaps** ({rec['n_pairs']} pairs; mean |effect| {rec['mean_abs_effect']:.2f}; "
                          "transfer of the donor's effect):", "",
                          "| site | " + " | ".join(str(L) for L in Ls) + " |", "|---|" + "---|" * len(Ls)]
                for s in mp["sites"]:
                    lines.append(f"| {s} | " + " | ".join(f"{rec['transfer'][s][str(L)]['value']:.2f}" for L in Ls) + " |")
                lines.append("")
        if (R / "das.json").exists():
            das = json.load(open(R / "das.json"))
            ch = das["choice"]
            lines += [f"**Chosen by the rule:** arrival layer {ch['arrival_layer']}; A site {ch['a_site'][0]} at layer {ch['a_site'][1]} "
                      f"(its swap carries {ch['a_site_share']:.2f} of a good swap); B site {ch['b_site'][0]} at layer {ch['b_site'][1]} "
                      f"({ch['b_site_share']:.2f}).", "",
                      "Shares of the site's own whole-swap effect, held-out pairs:", "",
                      "| site | direction | good swaps | number swaps | B's number | apples | weight | coord per value unit: good / number / B's number | cos(good-, number-trained) |",
                      "|---|---|---|---|---|---|---|---|---|"]
            f = lambda e: f"{e['of_site_swap']['value']:+.2f}"
            for role, rec in das["sites"].items():
                for nm, ev in rec["eval"].items():
                    lines.append(f"| {role}: {rec['site']}@{rec['layer']} | {nm} | {f(ev['good_test'])} | {f(ev['number_test'])} | "
                                 f"{f(ev['bnumber_test'])} | {f(ev['quantity'])} | {f(ev['weight'])} | "
                                 f"{ev['good_test']['coord_per_value_unit']:+.3f} / {ev['number_test']['coord_per_value_unit']:+.3f} / "
                                 f"{ev['bnumber_test']['coord_per_value_unit']:+.3f} | {rec['cos_good_vs_number_trained']:.2f} |")
                ev = rec["eval"]["good_trained"]
                lines.append(f"| (the whole site, share of the donor's effect) | | {ev['good_test']['site_swap_of_donor']['value']:.2f} | "
                             f"{ev['number_test']['site_swap_of_donor']['value']:.2f} | {ev['bnumber_test']['site_swap_of_donor']['value']:.2f} | "
                             f"{ev['quantity']['site_swap_of_donor']['value']:.2f} | {ev['weight']['site_swap_of_donor']['value']:.2f} | | |")
            V = verdicts_for(cfg, das)
            V["usable"] = das["usable"]
            allv[name] = V
            lines += ["", "Verdicts (DESIGN.md, section 4):", ""]
            for role in ("a_site", "b_site"):
                v = V[role]
                lines.append(f"- **{role}** ({v['site']}@{v['layer']}): **{v['headline']}**. V2 one currency {v['V2_one_currency']} "
                             f"(good {v['good_share']:.2f}, number {v['number_share']:.2f}); V3 exchange rate {v['V3_exchange_rate']} "
                             f"(number / good {v['exchange_ratio']:.2f}); V4 {v['V4']} (apples {v['quantity_share']:.2f}; the site carries "
                             f"{v['quantity_site_share']:.2f} of an apples swap); V5 {v['V5']} (coordinate per unit of B's value / per unit of A's "
                             f"{v['b_to_a_coord_ratio']:.2f}; the direction carries {v['bnumber_share']:.2f} of the site's B-swap effect; the site "
                             f"carries {v['bnumber_site_share']:.2f} of a B swap)")
            if not das["usable"]:
                lines.append("- This model's own values did not predict its held-out choices well enough: its verdicts are reported but not counted.")
            lines.append("")
    (Path(cfg["res"]) / "report.md").write_text("\n".join(lines) + "\n")
    save_json(allv, Path(cfg["res"]) / "verdicts7.json")
    log("report done: " + json.dumps({k: {r: v[r]["headline"] for r in v if r != "usable"} for k, v in allv.items()}))


def main():
    args = parse_args()
    cfg = load_cfg(args)
    names = args.models.split(",") if args.models else list(cfg["models"])
    stages = args.stages.split(",")
    data = Data(cfg)
    log("Pilot 7; pairs: " + ", ".join(f"{k} {len(v)}" for k, v in data.sets.items()))
    for name in names:
        model_stages = [s for s in stages if s != "report"]
        if not model_stages:
            continue
        t0 = time.time()
        log(f"===== model {name} =====")
        M = Model(cfg, name, data)
        for s in model_stages:
            t1 = time.time()
            log(f"===== stage {s} ({name}) =====")
            {"fit": stage_fit, "map": stage_map, "das": stage_das}[s](M)
            log(f"===== stage {s} ({name}) finished in {(time.time() - t1) / 60:.1f} min =====")
        del M
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        log(f"===== model {name} finished in {(time.time() - t0) / 60:.1f} min =====")
    if "report" in stages:
        stage_report(cfg, names)


if __name__ == "__main__":
    main()
