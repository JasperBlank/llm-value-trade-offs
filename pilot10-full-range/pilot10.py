"""Pilot 10: the full range of goodness (DESIGN.md). The GPU part.

  python pilot10.py            Qwen3-14B: writes results/real/menu_pA.npy and results/real/ratings10.json
  python pilot10.py --quick    Qwen3-0.6B on a subset: writes results/quick/ (a pipeline test)

Menu: P(A) for each of the 45,600 trade-offs (the letters' probabilities renormalised over A and B, as in pilot 8).
Ratings: for each of the 1,536 rating prompts, the log-probability of each digit 0-9 (renormalised over the digits;
digit logits in float32 from the final hidden state, as in pilot 9's follow-up), the expected rating, and the
log-scale score (pilot9-vibe-check/pilot9_logscale.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in ("pilot7-value-route", "pilot5-value-circuit", "pilot9-vibe-check", "pilot8-intrinsic-values"):
    sys.path.insert(0, str(ROOT / p))
import pilot7 as p7  # noqa: E402
from build_prompts10 import load_sentences, menu_prompt  # noqa: E402
from common import batches, forward, load_model, log  # noqa: E402
from pilot8 import letter_set  # noqa: E402
from pilot9_logscale import log_scale_score  # noqa: E402


def pad_batch(enc, ch, pad, torch, dev):
    T = max(len(enc[i]) for i in ch)
    ids = torch.full((len(ch), T), pad, dtype=torch.long)
    att = torch.zeros((len(ch), T), dtype=torch.long)
    for j, i in enumerate(ch):
        ids[j, T - len(enc[i]):] = torch.tensor(enc[i])
        att[j, T - len(enc[i]):] = 1
    return ids.to(dev), att.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    a = ap.parse_args()
    import torch
    name = "Qwen/Qwen3-0.6B" if a.quick else "Qwen/Qwen3-14B"
    out = HERE / "results" / ("quick" if a.quick else "real")
    out.mkdir(parents=True, exist_ok=True)
    S = load_sentences()
    keys = json.load(open(HERE / "menu_keys10.json"))
    rrows = [json.loads(l) for l in open(HERE / "ratings10.jsonl")]
    if a.quick:
        keys, rrows = keys[::400], rrows[::16]
    t0 = time.time()
    tok, model, dev = load_model({"model": name, "dtype": "bfloat16"})
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # the menu
    AB = letter_set(tok, "AB")
    enc = [tok(p7.chat_text(tok, menu_prompt(S, k)), add_special_tokens=False)["input_ids"] for k in keys]
    pA = np.zeros(len(keys), dtype=np.float32)
    mass = np.zeros(len(keys), dtype=np.float32)
    for bi, ch in enumerate(batches(len(keys), a.batch_size)):
        ids, att = pad_batch(enc, ch, pad, torch, dev)
        o = forward(model, ids, att, torch)
        pr = torch.softmax(o.logits[:, -1].float(), -1)
        pa, pb = pr[:, AB[0]].sum(-1), pr[:, AB[1]].sum(-1)
        pA[ch] = (pa / (pa + pb)).cpu().numpy()
        mass[ch] = (pa + pb).cpu().numpy()
        if bi % 300 == 0:
            log(f"menu {ch[-1] + 1} of {len(keys)}")
    np.save(out / "menu_pA.npy", pA)
    log(f"menu done: {len(keys)} trade-offs; letter probability min {mass.min():.3f}, median {np.median(mass):.3f}")

    # the ratings
    digits = [sorted({e[0] for s in (d, " " + d) if len(e := tok.encode(s, add_special_tokens=False)) == 1})
              for d in "0123456789"]
    flat = sorted({t for g in digits for t in g})
    col = {t: j for j, t in enumerate(flat)}
    W = model.lm_head.weight[flat].float()
    enc = [tok(p7.chat_text(tok, r["user"]), add_special_tokens=False)["input_ids"] for r in rrows]
    LP = np.zeros((len(rrows), 10))
    with torch.no_grad():
        for ch in batches(len(rrows), a.batch_size):
            ids, att = pad_batch(enc, ch, pad, torch, dev)
            h = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, -1].float()
            lg = h @ W.T
            L = torch.stack([torch.logsumexp(lg[:, [col[t] for t in g]], -1) for g in digits], 1)
            LP[ch] = torch.log_softmax(L, -1).double().cpu().numpy()
    res = {"model": name, "minutes": (time.time() - t0) / 60, "n_menu": len(keys), "n_ratings": len(rrows),
           "letter_mass_min": float(mass.min()), "letter_mass_median": float(np.median(mass)),
           "keys_menu_subset_every": 400 if a.quick else 1, "keys_ratings_subset_every": 16 if a.quick else 1,
           "rating": [round(float(x), 4) for x in np.exp(LP) @ np.arange(10)],
           "score": [round(float(x), 4) for x in log_scale_score(LP)],
           "logp": [[round(float(x), 3) for x in p] for p in LP],
           "keys": [[r["s"], r["n"], r["wording"]] for r in rrows]}
    json.dump(res, open(out / "ratings10.json", "w"))
    log(f"{name}: {len(keys)} trade-offs and {len(rrows)} ratings in {res['minutes']:.1f} min")


if __name__ == "__main__":
    main()
