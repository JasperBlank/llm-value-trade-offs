"""Build pilot 5's prompt list (run once, locally; prompts.jsonl is committed).

Groups:
  test   pilot 4's 6,912 strangers test prompts (pilot 3's six wordings): the value directions are
         fitted on these, and the circuit is traced on them.
  goal / twin   pilot 4's goal-module prompts and their twins without the goal line (the
         objective, read again with the last-position code).
  fresh  NEW prompts for the pre-registered steering test and the patching: the same pairs of
         real goods (within, cross, gain-wording mirrors), pilot 3's grid around each pair's balance
         point, both orders, strangers, in three NEW wordings (6, 7, 8) the probes never saw.
  donor  for each fresh prompt, the same prompt with option A's good swapped for another real
         option (same number of people, same B, same wording): the donor in the patching runs.
Labels: V = u_option + b * log(number), from pilot 3's fit of Qwen3-14B (values.json).
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
P3 = HERE.parent / "pilot3-value-gate"
P4 = HERE.parent / "pilot4-value-code"
sys.path.insert(0, str(P3))
from common import build_prompt, load_config, mirror_pairs, option_key, real_pairs, stage2_cells  # noqa: E402

NEW_WORDINGS = [
    "Which of these two outcomes should happen?",
    "Of these two outcomes, which one would you pick?",
    "Which of these two outcomes is preferable?",
]


def main():
    cfg = load_config(argparse.Namespace(config=str(P3 / "config.yaml"), dry_run=True, quick=False))
    first_new = len(cfg["prompts"]["question"])
    assert first_new == 6
    cfg["prompts"]["question"] = list(cfg["prompts"]["question"]) + NEW_WORDINGS
    W = list(range(first_new, first_new + len(NEW_WORDINGS)))
    bal = json.load(open(P3 / "results" / "real" / "balance_14b.json"))
    vals = json.load(open(HERE / "values.json"))
    b, u = vals["b"], vals["u"]

    def labels(oa, ob):
        ka, kb = option_key(oa), option_key(ob)
        import math
        return {"key_a": ka, "key_b": kb, "q_a": oa["q"], "q_b": ob["q"],
                "v_a": u[ka] + b * math.log(oa["q"]), "v_b": u[kb] + b * math.log(ob["q"])}

    out = []
    for r in (json.loads(l) for l in open(P4 / "prompts.jsonl", encoding="utf-8")):
        r.pop("index", None)
        r.pop("full", None)
        out.append(r)
    n_old = len(out)

    reals = sorted({option_key(a) for a, _, _ in real_pairs(cfg) + mirror_pairs(cfg)} |
                   {option_key(b_) for _, b_, _ in real_pairs(cfg) + mirror_pairs(cfg)})
    opt_of = {k: {"good": k.split("~")[0], **({"mirror": True} if "~" in k else {})} for k in reals}
    rng = random.Random(20261102)
    seen = set()
    for a, b_, kind in real_pairs(cfg) + mirror_pairs(cfg):
        R = bal[f"{option_key(a)}|{option_key(b_)}"]["strangers"]
        for qa, qb in stage2_cells(R, cfg):
            oa, ob = {**a, "q": int(qa)}, {**b_, "q": int(qb)}
            for opts in ([oa, ob], [ob, oa]):
                for w in W:
                    _, user = build_prompt(cfg, w, "strangers", opts, None)
                    rid = "|".join(["fresh", kind, ",".join(f"{option_key(o)}:{o['q']}" for o in opts), f"w{w}"])
                    if rid in seen:
                        continue
                    seen.add(rid)
                    out.append({"id": rid, "group": "fresh", "kind": kind, "goal_good": None, "wording": w,
                                "system": None, "user": user, **labels(*opts)})
                    # The donor: option A's good swapped for another real option (not B's good).
                    ka, kb = option_key(opts[0]), option_key(opts[1])
                    cands = [k for k in reals if k != ka and k.split("~")[0] != kb.split("~")[0]]
                    kd = rng.choice(cands)
                    od = {**opt_of[kd], "q": opts[0]["q"]}
                    _, duser = build_prompt(cfg, w, "strangers", [od, opts[1]], None)
                    out.append({"id": rid + "|donor", "group": "donor", "kind": kind, "goal_good": None, "wording": w,
                                "system": None, "user": duser, "recipient": rid, **labels(od, opts[1])})
    with open(HERE / "prompts.jsonl", "w", encoding="utf-8") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(len(out), "prompts:", dict(Counter(p["group"] for p in out)), "(from pilot 4:", n_old, ")")
    fresh = [p for p in out if p["group"] == "fresh"]
    print("fresh kinds:", dict(Counter(p["kind"] for p in fresh)))
    print("example fresh:\n" + fresh[0]["user"])
    print("its donor:\n" + out[out.index(fresh[0]) + 1]["user"])


if __name__ == "__main__":
    main()
