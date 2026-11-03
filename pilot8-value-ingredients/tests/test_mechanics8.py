"""Checks for pilot 8.

  python tests/test_mechanics8.py                     the fitting checks only (no model needed)
  python tests/test_mechanics8.py --config <cfg> --run   also checks a finished run's files

1. The maximum-likelihood fit recovers known worths, b and bias from simulated answers on the real
   behaviour design.
2. The ingredient regression recovers known effects exactly (R² = 1, held out = 1).
3. (--run) The run's files exist; the worth direction has length 1; at the top layer, swapping the
   answer position carries 1 for every ingredient.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def main():
    import pilot8 as p8
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        print(f"{'OK ' if cond else 'BAD'} {name} {info}")
        ok = ok and bool(cond)

    O = json.load(open(HERE / "outcomes.json"))["outcomes"]
    ids = [o["id"] for o in O]
    idx = {k: i for i, k in enumerate(ids)}
    rows = [json.loads(l) for l in open(HERE / "behaviour.jsonl")]
    ia = np.array([idx[r["a"]] for r in rows])
    ib = np.array([idx[r["b"]] for r in rows])
    dlog = np.array([math.log(r["na"] / r["nb"]) for r in rows])
    rng = np.random.default_rng(1)
    X = p8.features(O)
    beta = np.array([0.4, -0.8, 0.6, -0.3, 1.2, 0.5, 0.9])
    u_true = np.column_stack([np.ones(len(O)), X]) @ beta + rng.normal(0, 0.2, len(O))
    u_true -= u_true[0]
    b_true, c_true = 1.5, 0.3
    p = 1 / (1 + np.exp(-(u_true[ia] - u_true[ib] + b_true * dlog + c_true)))
    u, b, c, _ = p8.fit_worth(ia, ib, dlog, p, len(O))
    check("1. fit recovers the worths", np.abs(u - u_true).max() < 1e-3, f"max error {np.abs(u - u_true).max():.1e}")
    check("1b. fit recovers b and the bias", abs(b - b_true) < 1e-3 and abs(c - c_true) < 1e-3, f"b {b:.4f}, c {c:.4f}")
    u_exact = np.column_stack([np.ones(len(O)), X]) @ beta
    bt, r2, r2l = p8.ingredient_regression(X, u_exact)
    check("2. regression recovers the effects", np.abs(bt - beta).max() < 1e-9 and r2 > 0.999999 and r2l > 0.999999,
          f"R² {r2:.6f}, held out {r2l:.6f}")
    if "--run" in sys.argv:
        cfg_path = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else str(HERE / "config.yaml")
        import yaml
        cfg = yaml.safe_load(open(cfg_path))
        if "--quick" in sys.argv:
            cfg["models"] = cfg["quick"]["models"]
            cfg["results_dir"] = cfg["quick"]["results_dir"]
        res = Path(cfg["results_dir"])
        res = res if res.is_absolute() else HERE / res
        R = res / list(cfg["models"])[0]
        check("3a. behave, map, das files exist", all((R / f).exists() for f in ("behave.json", "map.json", "das.json")))
        w = np.load(R / "worth_direction.npy")
        check("3b. the worth direction has length 1", abs(np.linalg.norm(w) - 1) < 1e-3, f"{np.linalg.norm(w):.4f}")
        mp = json.load(open(R / "map.json"))
        top = str(mp["layers"][-1])
        vals = [rec["transfer"]["last"][top]["value"] for rec in mp["swaps"].values()]
        n_layers = json.load(open(R / "das.json")).get("n_layers")
        check("3c. top-layer swap of the answer position carries 1 (if the top layer is mapped)",
              all(abs(v - 1) < 0.02 for v in vals) or (n_layers is not None and int(top) != n_layers), str(np.round(vals, 3)))
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
