"""Exploratory, after pilot 12's results: how different are the two recipes in the model's states?
For each layer and prompt: ridge readouts of the price and of the rating (held out by value), each scored on its own
target and on the other; and the cosine between the two readouts' weight vectors (all sentences, penalty 1 x d)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe12 import RidgeLOVO, kernel, r  # noqa: E402

res = HERE / "results/real"
Z = np.load(res / "states12.npz")
layers, u, R = list(Z["layers"]), Z["u"], Z["Rbar"]
D = json.load(open(res / "disagreement12.json"))
vals = sorted({x["value"] for x in D})
groups = np.array([vals.index(x["value"]) for x in D])
levels = np.array([["small", "medium", "large"].index(x["level"]) for x in D])
wl = lambda x: x - np.array([x[levels == L].mean() for L in levels])
L_ = ["# Exploratory: how different are the two recipes? (pilot 12's states)", "",
      f"Price and rating (mean of W0-W2, 8 people) correlate r = {r(u, R):.2f}; within stakes level {r(wl(u), wl(R)):.2f}.", "",
      "| layer | prompt | price readout: price | price readout: rating | rating readout: rating | rating readout: price | cosine of the two readouts |",
      "|---|---|---|---|---|---|---|"]
for Lr in (16, 20, 24, 28):
    li = layers.index(Lr)
    for ctx in ("menu_mean", "rating_mean"):
        X = Z[ctx][:, li].astype(np.float64)
        K = kernel(X)
        m = RidgeLOVO(K, groups, X.shape[1])
        pu, pR = m.predict(u), m.predict(R)
        A = (X - X.mean(0)) / (X.std(0) + 1e-6)
        inv = np.linalg.inv(K + X.shape[1] * np.eye(len(K)))
        wu, wR = A.T @ inv @ (u - u.mean()), A.T @ inv @ (R - R.mean())
        cos = float(wu @ wR / np.linalg.norm(wu) / np.linalg.norm(wR))
        L_.append(f"| {Lr} | {'menu (as option A)' if ctx == 'menu_mean' else 'rating'} | {r(pu, u):.2f} | {r(pu, R):.2f} | {r(pR, R):.2f} | {r(pR, u):.2f} | {cos:.2f} |")
(res / "exploratory_recipes.md").write_text("\n".join(L_) + "\n")
print("\n".join(L_))
