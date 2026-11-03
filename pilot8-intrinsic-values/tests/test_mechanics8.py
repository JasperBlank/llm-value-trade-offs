"""Checks for pilot 8 (intrinsic values).

  python tests/test_mechanics8.py                              fitting and counting checks (no model)
  python tests/test_mechanics8.py --config <cfg> --run [--quick]  also the tokenizer sites and a run's files

1. The maximum-likelihood fit recovers known worths, b and bias on the real trade-off design.
2. The cycle count: a transitive order has none; one reversed pair inside an order makes cycles.
3. (--run) Rating prompts: R_end is the outcome line's last token, R_nl lies between it and the reply
   line, and everything after R_end is the same text in every rating pair (so positions line up).
4. (--run) The run's files exist and the learned directions have length 1.
"""
import itertools
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

    names = [v["name"] for v in json.load(open(HERE / "values.json"))["values"]]
    idx = {k: i for i, k in enumerate(names)}
    rows = [json.loads(l) for l in open(HERE / "behaviour.jsonl")]
    ia, ib = np.array([idx[r["a"]] for r in rows]), np.array([idx[r["b"]] for r in rows])
    dlog = np.array([math.log(r["na"] / r["nb"]) for r in rows])
    rng = np.random.default_rng(1)
    u_true = rng.normal(0, 1.5, len(names))
    u_true -= u_true[0]
    p = 1 / (1 + np.exp(-(u_true[ia] - u_true[ib] + 1.3 * dlog - 0.4)))
    u, b, c, _ = p8.fit_worth(ia, ib, dlog, p, len(names))
    check("1. fit recovers worths, b and bias", np.abs(u - u_true).max() < 1e-3 and abs(b - 1.3) < 1e-3 and abs(c + 0.4) < 1e-3,
          f"max error {np.abs(u - u_true).max():.1e}, b {b:.4f}, c {c:.4f}")
    order = {v: k for k, v in enumerate(names)}
    pref = {(x, y): 0.9 if order[x] < order[y] else 0.1 for x, y in itertools.permutations(names, 2)}
    c0, _ = p8.cyclic_triads(names, pref)
    pref[(names[0], names[5])], pref[(names[5], names[0])] = 0.1, 0.9
    c1, _ = p8.cyclic_triads(names, pref)
    check("2. cycles: none in a transitive order, some after reversing one pair", c0 == 0 and c1 == 4, f"{c0}, {c1}")
    if "--run" in sys.argv:
        import yaml
        cfg = yaml.safe_load(open(sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else HERE / "config.yaml"))
        if "--quick" in sys.argv:
            cfg["models"], cfg["results_dir"] = cfg["quick"]["models"], cfg["quick"]["results_dir"]
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(list(cfg["models"].values())[0]["path"])

        class _M:
            pass
        M = _M()
        M.tok = tok
        R = p8.RatingEnc(M)
        bad = 0
        for pr in [json.loads(l) for l in open(HERE / "rating_pairs.jsonl")][:200]:
            er, ed = R.enc(pr["user"]), R.enc(pr["donor_user"])
            line = pr["user"].split("\n")[2]
            last_tok = tok.decode([er["ids"][er["a_end"]]]).strip()
            bad += not (line.endswith(last_tok) and len(er["sites"]["R_nl"]) >= 1 and er["ids"][er["a_end"] + 1:] == ed["ids"][ed["a_end"] + 1:])
        check("3. rating sites well formed and aligned", bad == 0, f"{bad} bad of 200")
        res = Path(cfg["results_dir"])
        Rd = (res if res.is_absolute() else HERE / res) / list(cfg["models"])[0]
        check("4a. result files exist", all((Rd / f).exists() for f in ("behave.json", "source.json", "compare.json", "rating.json")))
        norms = [float(np.linalg.norm(np.load(Rd / f))) for f in ("worth_direction.npy", "comparison_direction.npy")]
        check("4b. learned directions have length 1", all(abs(n - 1) < 1e-3 for n in norms), str(np.round(norms, 4)))
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
