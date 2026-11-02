"""Build pilot 7's prompt pairs (run once, locally; pairs.jsonl is committed).

Every item is a PAIR: a recipient prompt and a donor that differ only in option A's line.
  good_train / good_test     A's good changes, the number of people stays (pilot 6's value pairs:
                             pilot 4's six wordings for training, pilot 5's three new wordings held out)
  number_train / number_test A's number of people changes, the good stays (same recipients); the
                             number changes by at least a factor 2
  bnumber_test               B's number of people changes (same number of digits, never 1), for the
                             test of whether a site holds A's value or already the comparison
  quantity / weight          pilot 6's non-value comparisons (A's number of apples / A's object changes)
Values (V = u + b log n) are pilot 3's fit for Qwen3-14B; pilot7.py also fits each model's own.
"""
import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
P3 = HERE.parent / "pilot3-value-gate"
P6 = HERE.parent / "pilot6-value-subspace"
sys.path.insert(0, str(P3))
from common import build_prompt, load_config  # noqa: E402

NUMBERS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def main():
    cfg = load_config(argparse.Namespace(config=str(P3 / "config.yaml"), dry_run=True, quick=False))
    cfg["prompts"]["question"] = list(cfg["prompts"]["question"]) + [
        "Which of these two outcomes should happen?", "Of these two outcomes, which one would you pick?",
        "Which of these two outcomes is preferable?"]
    vals = json.load(open(HERE / "values.json"))
    b, u = vals["b"], vals["u"]
    rng = random.Random(20261103)

    def opt(key, q):
        return {"good": key.split("~")[0], **({"mirror": True} if "~" in key else {}), "q": q}

    p6 = [json.loads(l) for l in open(P6 / "pairs.jsonl", encoding="utf-8")]
    rows5 = {json.loads(l)["id"]: json.loads(l) for l in open(HERE.parent / "pilot5-value-circuit/prompts.jsonl", encoding="utf-8")}
    out = []
    for p in p6:
        s = {"value_train": "good_train", "value_test": "good_test"}.get(p["set"], p["set"])
        out.append({**p, "set": s, "swap": "good" if s.startswith("good") else s})
        if not s.startswith("good"):
            continue
        r = rows5[p["id"].split("|", 1)[1]]
        _, user = build_prompt(cfg, r["wording"], "strangers", [opt(r["key_a"], r["q_a"]), opt(r["key_b"], r["q_b"])], None)
        assert user == p["user"], p["id"]
        cands = [n for n in NUMBERS if abs(math.log2(n / r["q_a"])) >= 1]
        qd = rng.choice(cands)
        _, duser = build_prompt(cfg, r["wording"], "strangers", [opt(r["key_a"], qd), opt(r["key_b"], r["q_b"])], None)
        out.append({"id": p["id"].replace("train|", "ntrain|").replace("test|", "ntest|"), "set": s.replace("good", "number"),
                    "swap": "number", "user": p["user"], "donor_user": duser, "key_a": r["key_a"], "key_b": r["key_b"],
                    "q_a": r["q_a"], "q_b": r["q_b"], "donor_q_a": qd, "wording": r["wording"],
                    "v_a": p["v_a"], "v_b": p["v_b"], "donor_v_a": u[r["key_a"]] + b * math.log(qd)})
    pools = [[2, 3, 4, 6, 8], [12, 16, 24, 32, 48, 64]]
    for p in [x for x in out if x["set"] == "good_test"]:
        qb = p["q_b"]
        if qb == 1:
            continue                                        # "1 household ... gets" has another form
        pool = pools[0] if qb < 10 else pools[1]
        cands = [n for n in pool if abs(math.log2(n / qb)) >= 1]
        if not cands:
            continue
        r = rows5[p["id"].split("|", 1)[1]]
        qd = rng.choice(cands)
        _, duser = build_prompt(cfg, r["wording"], "strangers", [opt(r["key_a"], r["q_a"]), opt(r["key_b"], qd)], None)
        out.append({"id": p["id"].replace("test|", "btest|"), "set": "bnumber_test", "swap": "bnumber",
                    "user": p["user"], "donor_user": duser, "key_a": r["key_a"], "key_b": r["key_b"],
                    "q_a": r["q_a"], "q_b": qb, "donor_q_b": qd, "wording": r["wording"],
                    "v_a": p["v_a"], "v_b": p["v_b"], "donor_v_b": u[r["key_b"]] + b * math.log(qd)})
    for p in out:
        if p["swap"] == "good":
            p["wording"] = rows5[p["id"].split("|", 1)[1]]["wording"]
            p["donor_q_a"] = p["q_a"]
    with open(HERE / "pairs.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(out), dict(Counter(r["set"] for r in out)))
    ex = next(x for x in out if x["set"] == "number_test")
    print(ex["user"], "\n-- donor A:", ex["donor_user"].splitlines()[2])


if __name__ == "__main__":
    main()
