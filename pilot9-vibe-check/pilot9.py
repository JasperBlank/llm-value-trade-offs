"""Pilot 9: ask the model directly how good each outcome would be (DESIGN.md).

  python pilot9.py            Qwen3-14B, writes results/real/ratings9.json
  python pilot9.py --quick    Qwen3-0.6B on 48 prompts, writes results/quick/ratings9.json (a pipeline test)

For every prompt in ratings9.jsonl: the probability of each digit 0-9 as the first answer token (with or
without a leading space), renormalised over the digits; the expected rating; the digits' total probability.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot7-value-route"))
sys.path.insert(0, str(HERE.parent / "pilot5-value-circuit"))
import pilot7 as p7  # noqa: E402
from common import batches, forward, load_model, log  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args()
    import torch
    name = "Qwen/Qwen3-0.6B" if a.quick else "Qwen/Qwen3-14B"
    out = HERE / "results" / ("quick" if a.quick else "real")
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(HERE / "ratings9.jsonl")]
    if a.quick:
        rows = rows[::12]
    t0 = time.time()
    tok, model, dev = load_model({"model": name, "dtype": "bfloat16"})
    digits = [sorted({e[0] for s in (d, " " + d) if len(e := tok.encode(s, add_special_tokens=False)) == 1})
              for d in "0123456789"]
    assert all(digits), digits
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    enc = [tok(p7.chat_text(tok, r["user"]), add_special_tokens=False)["input_ids"] for r in rows]
    P = np.zeros((len(rows), 10))
    mass = np.zeros(len(rows))
    top = []
    for ch in batches(len(rows), a.batch_size):
        T = max(len(enc[i]) for i in ch)
        ids = torch.full((len(ch), T), pad, dtype=torch.long)
        att = torch.zeros((len(ch), T), dtype=torch.long)
        for j, i in enumerate(ch):
            ids[j, T - len(enc[i]):] = torch.tensor(enc[i])
            att[j, T - len(enc[i]):] = 1
        o = forward(model, ids.to(dev), att.to(dev), torch)
        pr = torch.softmax(o.logits[:, -1].float(), -1)
        d = torch.stack([pr[:, g].sum(-1) for g in digits], 1).cpu().numpy()
        P[ch] = d / d.sum(1, keepdims=True)
        mass[ch] = d.sum(1)
        top += [tok.decode([t]) for t in pr.argmax(-1).tolist()]
    rating = P @ np.arange(10)
    res = {"model": name, "n_prompts": len(rows), "minutes": (time.time() - t0) / 60,
           "digit_mass_min": float(mass.min()), "digit_mass_median": float(np.median(mass)),
           "top_tokens": {t: top.count(t) for t in sorted(set(top))},
           "rating": [round(float(x), 4) for x in rating],
           "probs": [[round(float(x), 5) for x in p] for p in P],
           "keys": [[r["value"], r["phr"], r["n"], r["wording"]] for r in rows]}
    json.dump(res, open(out / "ratings9.json", "w"))
    log(f"{name}: {len(rows)} prompts in {res['minutes']:.1f} min; digit probability min {mass.min():.3f}, "
        f"median {np.median(mass):.3f}; top tokens {res['top_tokens']}")


if __name__ == "__main__":
    main()
