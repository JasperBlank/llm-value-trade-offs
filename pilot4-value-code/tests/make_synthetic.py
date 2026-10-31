"""Synthetic activations with a planted value code, to test probes.py and neurons.py.

Planted at layer 3 (of 0..5), width 64:
  option A's line end / number:  V_A * e1 + log(q_A) * e2 (+ noise)   -> one common value direction e1
  option B's line end / number:  V_B * e1 + log(q_B) * e2
  final positions:               (V_A - V_B) * e3
  goal module: version 1 (objective on a real good) adds +3 value to the target at its line end
               (extra value); version 2 (objective on files) adds +3 along e4 at the end only, so
               the choice moves but no value readout does (a bypass).
  full prompts: the a_text tokens carry V_A * e1 at layer 3.
Every other layer and position is noise. probes.py must find all of this.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REGIONS, label_arrays  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(Path(__file__).resolve().parents[1] / "prompts.jsonl")]
N, L, d = len(rows), 6, 64
names = ["a_num", "a_end", "b_num", "b_end", "reply_end", "t-5", "t-4", "t-3", "t-2", "t-1"]
P = len(names)
rng = np.random.default_rng(1)
lab = label_arrays(rows)
e = np.linalg.qr(rng.normal(size=(d, 4)))[0].T * 4.0
X = rng.normal(0, 1, size=(N, L, P, d)).astype(np.float32)
tslot = np.array([0 if r["key_a"] == r["goal_good"] else 1 for r in rows])
ext_a = np.array([3.0 if (r["kind"] == "goal1" and t == 0) else 0.0 for r, t in zip(rows, tslot)])
ext_b = np.array([3.0 if (r["kind"] == "goal1" and t == 1) else 0.0 for r, t in zip(rows, tslot)])
va, vb = lab["v_a"] + ext_a, lab["v_b"] + ext_b
for pos, v, q in (("a_end", va, lab["log_qa"]), ("a_num", va, lab["log_qa"]), ("b_end", vb, lab["log_qb"]), ("b_num", vb, lab["log_qb"])):
    X[:, 3, names.index(pos)] += np.outer(v, e[0]) + np.outer(q, e[1])
dv = va - vb
bypass = np.array([3.0 * (1 if t == 0 else -1) if r["kind"] == "goal2" else 0.0 for r, t in zip(rows, tslot)])
for pos in ("t-1", "t-2", "reply_end"):
    X[:, 3, names.index(pos)] += np.outer(dv, e[2]) + np.outer(bypass, e[3])
z = dv + bypass + rng.normal(0, 0.3, N)
p_a = 1 / (1 + np.exp(-z))
np.save(out / "resid.npy", X.astype(np.float16))
full_rows = np.array([i for i, r in enumerate(rows) if r["full"]])
T = 40
regs_one = np.repeat(np.arange(len(REGIONS)), 4)[:T].astype(np.int16)
F = rng.normal(0, 1, size=(len(full_rows) * T, L, d)).astype(np.float32)
for k, i in enumerate(full_rows):
    m = np.where(regs_one == REGIONS.index("a_text"))[0] + k * T
    F[m, 3] += va[i] * e[0]
np.save(out / "resid_full.npy", F.astype(np.float16))
np.savez(out / "meta.npz", p_a=p_a, p_b=1 - p_a, positions=np.zeros((N, P), np.int32), lengths=np.full(N, T),
         full_rows=full_rows, full_offsets=np.arange(len(full_rows) + 1) * T, full_regions=np.tile(regs_one, len(full_rows)),
         region_names=np.array(REGIONS), position_names=np.array(names), n_layers=L - 1, width=d, model="synthetic")
# Neuron running sums (test prompts only): 4 layers x 5 positions x 50 neurons; neurons 0-9 code V_A at a_end
# in layer 2, neurons 10-19 code the choice at t-1 in layer 3; the rest are noise.
test = np.array([r["group"] == "test" for r in rows])
pc = np.clip(p_a[test], 1e-4, 1 - 1e-4)
Y = np.stack([lab["v_a"][test], lab["v_b"][test], pc * lab["v_a"][test] + (1 - pc) * lab["v_b"][test],
              np.log(pc / (1 - pc)), lab["dv"][test], lab["log_qa"][test], lab["log_qb"][test]], 1)
A = rng.normal(0, 1, size=(test.sum(), 4, 5, 50))
realA = np.array([not r["key_a"].startswith("N_") for r in rows])[test]
A[:, 2, 1, :10] += 0.8 * (Y[:, [0]] - Y[realA, 0].mean()) / Y[realA, 0].std()     # value neurons (scaled to real goods)
A[:, 3, 4, 10:20] += 0.8 * (Y[:, [3]] - Y[:, 3].mean()) / Y[:, 3].std()
A[:, 2, 1, 20:30] += 0.8 * (Y[:, [5]] - Y[:, 5].mean()) / Y[:, 5].std()
trows = [r for r in rows if r["group"] == "test"]
keys = sorted({r["key_a"] for r in trows} | {r["key_b"] for r in trows})
KS = np.zeros((4, 4, 50, len(keys), 3))
for oj, (pj, s_) in enumerate([(0, "a"), (1, "a"), (2, "b"), (3, "b")]):
    kk = np.array([keys.index(r["key_" + s_]) for r in trows])
    q = np.log([r["q_" + s_] for r in trows])
    for k in range(len(keys)):
        m = kk == k
        KS[:, oj, :, k, 0] = A[m, :, pj].sum(0)
        KS[:, oj, :, k, 1] = (A[m, :, pj] ** 2).sum(0)
        KS[:, oj, :, k, 2] = (A[m, :, pj] * q[m, None, None]).sum(0)
ks = {s_: np.array([[sum(1 for r in trows if r["key_" + s_] == k), sum(np.log(r["q_" + s_]) for r in trows if r["key_" + s_] == k),
                     sum(np.log(r["q_" + s_]) ** 2 for r in trows if r["key_" + s_] == k)] for k in keys]) for s_ in "ab"}
np.savez(out / "neurons.npz", n=int(test.sum()), s1=A.sum(0), s2=(A ** 2).sum(0), sy=np.einsum("nlpi,nk->lpik", A, Y), y=Y,
         vars=np.array(["v_a", "v_b", "chosen_value", "choice", "dv", "log_qa", "log_qb"]),
         positions=np.array(["a_num", "a_end", "b_num", "b_end", "t-1"]), key_sums=KS.astype(np.float32), keys=np.array(keys),
         key_positions=np.array(["a_num", "a_end", "b_num", "b_end"]), key_stats_a=ks["a"], key_stats_b=ks["b"])
print("synthetic activations written to", out, X.shape)
