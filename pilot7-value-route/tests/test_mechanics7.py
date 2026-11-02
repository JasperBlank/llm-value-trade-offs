"""Mechanical checks for pilot 7 (run after the pipeline, on the same config).

  python tests/test_mechanics7.py --config <config used for the run> [--models qwen]
  python tests/test_mechanics7.py --quick

1. Sites: every prompt has non-empty sites, all positions negative, the answer position is -1, the
   A sites come before B's line, and recipient and donor line up (A's line end at the same negative
   position; for B swaps, the same length).
2. Patching the whole text after A's line at the top block gives the donor's answer exactly;
   patching an empty set changes nothing; the subspace swap with the whole space equals the site swap.
3. Map: at the top layer, 'after_A' and 'last' carry 1 for every kind of swap; the A sites carry
   nothing of a B swap (they come before B's line).
4. Learned directions have length 1; fit, map, das files and the report exist.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import pilot7 as p7  # noqa: E402
from common import Hooks, forward, layer_output  # noqa: E402


def main():
    import torch
    args = p7.parse_args()
    cfg = p7.load_cfg(args)
    names = args.models.split(",") if args.models else list(cfg["models"])
    data = p7.Data(cfg)
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        print(f"{'OK ' if cond else 'BAD'} {name} {info}")
        ok = ok and bool(cond)

    for name in names:
        R = Path(cfg["res"]) / name
        if not (R / "fit.json").exists():
            print(f"(skip {name}: not run)")
            continue
        M = p7.Model(cfg, name, data)
        bad = 0
        for s, ps in M.pairs.items():
            for p in ps[:50]:
                er, ed = M.enc(p["user"]), M.enc(p["donor_user"])
                S = er["sites"]
                aligned = (len(er["ids"]) == len(ed["ids"])) if p.get("swap") == "bnumber" else \
                    (er["a_end"] - len(er["ids"]) == ed["a_end"] - len(ed["ids"]))
                bad += not (all(len(S[k]) > 0 for k in ("A_end", "A_nl", "B_line", "B_nl", "reply", "template", "between", "after_A"))
                            and all(k < 0 for v in S.values() for k in v) and S["last"] == [-1]
                            and max(S["A_end"] + S["A_nl"]) < min(S["B_line"]) and aligned)
        check(f"[{name}] 1. sites well formed and aligned", bad == 0, f"{bad} bad")
        ps = M.pairs["good_test"][:8]
        idx = list(range(len(ps)))
        top = len(M.layers)
        with torch.no_grad():
            cap = {}
            with Hooks() as hk:
                hk.fwd(M.layers[top - 1], lambda m, i, o: cap.__setitem__("top", layer_output(o)))
                hk.fwd(M.layers[M.map_layers[1] - 1], lambda m, i, o: cap.__setitem__("mid", layer_output(o)))
                idd, attd, _ = M.batch([ps[i]["donor_user"] for i in idx])
                y_d = M.ydiff(forward(M.model, idd, attd, torch))
            Td = idd.shape[1]
            ids, att, es = M.batch([ps[i]["user"] for i in idx])
            y_r = M.ydiff(forward(M.model, ids, att, torch))
            rows, ks = M.site_index(es, "after_A")
            with Hooks() as hk:
                hk.fwd(M.layers[top - 1], p7.patch_hook(rows, ks, cap["top"][rows, Td + ks]))
                y_top = M.ydiff(forward(M.model, ids, att, torch))
            e = torch.tensor([], dtype=torch.long, device=M.dev)
            with Hooks() as hk:
                hk.fwd(M.layers[top - 1], p7.patch_hook(e, e, cap["top"][e, e]))
                y_none = M.ydiff(forward(M.model, ids, att, torch))
            L = M.map_layers[1]
            rows, ks = M.site_index(es, "between")
            src = cap["mid"][rows, Td + ks]
            with Hooks() as hk:
                hk.fwd(M.layers[L - 1], p7.patch_hook(rows, ks, src))
                y_site = M.ydiff(forward(M.model, ids, att, torch))
            I = torch.eye(M.sh["d"], device=M.dev)
            with Hooks() as hk:
                hk.fwd(M.layers[L - 1], p7.patch_hook(rows, ks, src, I))
                y_full = M.ydiff(forward(M.model, ids, att, torch))
        tol = 1e-3 + 0.02 * float(np.abs(y_d).max())
        check(f"[{name}] 2a. after_A at the top block = donor's answer", np.abs(y_top - y_d).max() < tol, f"max diff {np.abs(y_top - y_d).max():.2e}")
        check(f"[{name}] 2b. empty patch = no change", np.abs(y_none - y_r).max() < tol)
        check(f"[{name}] 2c. subspace swap of the whole space = site swap", np.abs(y_full - y_site).max() < tol, f"max diff {np.abs(y_full - y_site).max():.2e}")
        mp = json.load(open(R / "map.json"))
        Lt = str(mp["layers"][-1])
        if int(Lt) == top:
            vals = [mp["swaps"][s]["transfer"][site][Lt]["value"] for s in mp["swaps"] for site in ("after_A", "last")]
            check(f"[{name}] 3. map: top layer carries 1 for after_A and last", all(abs(v - 1) < 0.02 for v in vals), str(np.round(vals, 3)))
        bt = [mp["swaps"]["bnumber"]["transfer"][site][str(L)]["value"] for site in ("A_end", "A_nl") for L in mp["layers"]]
        check(f"[{name}] 3b. map: the A sites (before B's line) carry nothing of a B swap", all(abs(v) < 1e-6 for v in bt))
        if (R / "directions.npz").exists():
            Z = np.load(R / "directions.npz")
            worst = max(abs(np.linalg.norm(Z[k]) - 1) for k in Z.files)
            check(f"[{name}] 4a. learned directions have length 1", worst < 1e-3, f"worst {worst:.1e}")
        check(f"[{name}] 4b. fit, map, das files exist", all((R / f).exists() for f in ("fit.json", "map.json", "das.json")))
        del M
    check("4c. report and verdicts exist", (Path(cfg["res"]) / "report.md").exists() and (Path(cfg["res"]) / "verdicts7.json").exists())
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
