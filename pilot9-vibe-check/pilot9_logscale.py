"""Pilot 9 follow-up (exploratory, after the results): the same 576 rating prompts, read on a log scale.

  python pilot9_logscale.py            Qwen3-14B, writes results/logscale/ratings9_logscale.json
  python pilot9_logscale.py --quick    Qwen3-0.6B on 48 prompts, writes results/quick/ratings9_logscale.json

Under W0 more than half the answers put about 99% on "7", so the expected rating barely moves. Here, for every
prompt: the log-probability of each digit 0-9 (renormalised over the digits; the digit logits are computed in
float32 from the final hidden state), the expected rating, and a log-scale score:

    score = mean over k = 1..9 of  log P(answer >= k) - log P(answer < k)

In an ordered-logit picture this is the location of the answer on a latent scale (in logits), and it keeps the
ratios between small probabilities (0.1% against 0.4% on "8") that the expected rating flattens.
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
from common import batches, load_model, log  # noqa: E402


def logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def log_scale_score(logp):
    """logp: (n, 10) log-probabilities over the digits. Mean cumulative log-odds over the nine cut-points."""
    return np.mean([logsumexp(logp[:, k:]) - logsumexp(logp[:, :k]) for k in range(1, 10)], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args()
    import torch
    name = "Qwen/Qwen3-0.6B" if a.quick else "Qwen/Qwen3-14B"
    out = HERE / "results" / ("quick" if a.quick else "logscale")
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(HERE / "ratings9.jsonl")]
    if a.quick:
        rows = rows[::12]
    t0 = time.time()
    tok, model, dev = load_model({"model": name, "dtype": "bfloat16"})
    digits = [sorted({e[0] for s in (d, " " + d) if len(e := tok.encode(s, add_special_tokens=False)) == 1})
              for d in "0123456789"]
    assert all(digits), digits
    flat = sorted({t for g in digits for t in g})
    col = {t: j for j, t in enumerate(flat)}
    W = model.lm_head.weight[flat].float()                    # (m, d) the digit rows of the unembedding
    bias = getattr(model.lm_head, "bias", None)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    enc = [tok(p7.chat_text(tok, r["user"]), add_special_tokens=False)["input_ids"] for r in rows]
    LP = np.zeros((len(rows), 10))
    with torch.no_grad():
        for ch in batches(len(rows), a.batch_size):
            T = max(len(enc[i]) for i in ch)
            ids = torch.full((len(ch), T), pad, dtype=torch.long)
            att = torch.zeros((len(ch), T), dtype=torch.long)
            for j, i in enumerate(ch):
                ids[j, T - len(enc[i]):] = torch.tensor(enc[i])
                att[j, T - len(enc[i]):] = 1
            h = model.model(input_ids=ids.to(dev), attention_mask=att.to(dev)).last_hidden_state[:, -1].float()
            lg = h @ W.T
            if bias is not None:
                lg = lg + bias[flat].float()
            L = torch.stack([torch.logsumexp(lg[:, [col[t] for t in g]], -1) for g in digits], 1)
            LP[ch] = torch.log_softmax(L, -1).double().cpu().numpy()
    rating = np.exp(LP) @ np.arange(10)
    score = log_scale_score(LP)
    res = {"model": name, "n_prompts": len(rows), "minutes": (time.time() - t0) / 60,
           "score": [round(float(x), 4) for x in score], "rating": [round(float(x), 4) for x in rating],
           "logp": [[round(float(x), 3) for x in p] for p in LP],
           "keys": [[r["value"], r["phr"], r["n"], r["wording"]] for r in rows]}
    json.dump(res, open(out / "ratings9_logscale.json", "w"))
    log(f"{name}: {len(rows)} prompts in {res['minutes']:.1f} min; score range {score.min():.2f} to {score.max():.2f}")


if __name__ == "__main__":
    main()
