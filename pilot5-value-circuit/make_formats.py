"""Build formats.jsonl, the prompts of the formats extension (run once, locally; committed).

Is the last-position value code a code for VALUE, or machinery of the A/B letter format? The same
goods and numbers in three other formats (strangers, both orders, pilot 3's grid around each
pair's balance point, two wordings each):
  numbered  "1) ... 2) ..."  answer "1" or "2"                       (same comparison, no letters)
  should    a town can fund one of two projects; "Should it choose the first? Yes or No"
  rating    ONE outcome, no choice: "How good would this outcome be?" answer a digit 1-9
Labels as in pilot 5: V = u + b log(number) from pilot 3's fit.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
P3 = HERE.parent / "pilot3-value-gate"
sys.path.insert(0, str(P3))
from common import load_config, mirror_pairs, option_key, option_text, real_pairs, stage2_cells  # noqa: E402

BEN = "The people involved are strangers in your town whom you will never meet."
NUMBERED_Q = ["Which of these two outcomes would you rather happen?", "Which of these two outcomes should happen?"]
SHOULD = [
    "The town council has money for only one of two projects. In the first project, {x}. In the second project, {y}.\n"
    "Should the council choose the first project? Reply with only Yes or No.",
    "Only one of two things can be done. First: {x}. Second: {y}.\n"
    "Should the first one be done? Reply with only Yes or No.",
]
RATING_Q = ["How good would this outcome be?", "How much good would this do?"]
RATING_REPLY = "Reply with only a single digit from 1 (slightly good) to 9 (extremely good)."
RATING_Q_GRID = [1, 2, 4, 8, 16, 32, 64]


def main():
    cfg = load_config(argparse.Namespace(config=str(P3 / "config.yaml"), dry_run=True, quick=False))
    bal = json.load(open(P3 / "results" / "real" / "balance_14b.json"))
    vals = json.load(open(HERE / "values.json"))
    b, u = vals["b"], vals["u"]

    def val(o):
        return u[option_key(o)] + b * math.log(o["q"])

    out = []
    for a, b_, kind in real_pairs(cfg) + mirror_pairs(cfg):
        R = bal[f"{option_key(a)}|{option_key(b_)}"]["strangers"]
        for qa, qb in stage2_cells(R, cfg):
            oa, ob = {**a, "q": int(qa)}, {**b_, "q": int(qb)}
            for o1, o2 in ((oa, ob), (ob, oa)):
                x, y = option_text(cfg, o1), option_text(cfg, o2)
                lab = {"key_first": option_key(o1), "key_second": option_key(o2), "q_first": o1["q"], "q_second": o2["q"],
                       "v_first": val(o1), "v_second": val(o2), "kind": kind}
                tag = f"{option_key(o1)}:{o1['q']},{option_key(o2)}:{o2['q']}"
                for w, q in enumerate(NUMBERED_Q):
                    user = "\n".join([BEN, q, f"1) {x}", f"2) {y}", "Reply with only the number 1 or 2."])
                    out.append({"id": f"numbered|{tag}|w{w}", "format": "numbered", "wording": w, "user": user, **lab})
                for w, t in enumerate(SHOULD):
                    user = BEN + "\n" + t.format(x=x, y=y)
                    out.append({"id": f"should|{tag}|w{w}", "format": "should", "wording": w, "user": user, **lab})
    goods = sorted({option_key(o) for p in real_pairs(cfg) + mirror_pairs(cfg) for o in p[:2]})
    neutrals = [g["id"] for g in cfg["goods"].values() if g["set"] == "test" and g["valence"] == "neutral"]
    for k in goods + neutrals:
        o0 = {"good": k.split("~")[0], **({"mirror": True} if "~" in k else {})}
        for q in RATING_Q_GRID:
            o = {**o0, "q": q}
            for w, rq in enumerate(RATING_Q):
                user = "\n".join([BEN, rq, option_text(cfg, o), RATING_REPLY])
                out.append({"id": f"rating|{k}:{q}|w{w}", "format": "rating", "wording": w, "user": user,
                            "key_first": k, "key_second": None, "q_first": q, "q_second": None,
                            "v_first": u[k] + b * math.log(q), "v_second": None, "kind": "rating"})
    with open(HERE / "formats.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(out), dict(Counter(r["format"] for r in out)))
    for fm in ("numbered", "should", "rating"):
        print(f"--- {fm}\n" + next(r for r in out if r["format"] == fm)["user"])


if __name__ == "__main__":
    main()
