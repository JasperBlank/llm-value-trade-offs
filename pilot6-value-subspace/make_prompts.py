"""Build pilot 6's prompt pairs (run once, locally; pairs.jsonl is committed).

Each line is a PAIR: a recipient prompt and a donor prompt that differ only in option A.
  value_train  pilot 4's strangers test prompts between real goods (within, cross, mirror; pilot 3's
               six wordings), each with a donor whose option A is another real option (same number
               of people, same B, same wording; never B's own good). Used to TRAIN the subspace.
  value_test   pilot 5's fresh prompts (three new wordings) and their donors. Held out.
  quantity     a non-value control: "Which of these two piles has more apples?" A) 12 apples B) 30 apples.
               The donor changes A's number so that the answer flips.
  weight       a non-value control: "Which of these two is heavier?" A) a horse B) a piano.
               The donor changes A's object so that the answer flips.
Values for the value pairs: V = u + b log(number) (values.json, pilot 3's fit of Qwen3-14B).
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
P5 = HERE.parent / "pilot5-value-circuit"
sys.path.insert(0, str(P3))
from common import build_prompt, load_config  # noqa: E402

QUANTITY_Q = [("Which of these two piles has more apples?", "apples"),
              ("Which of these two boxes contains more marbles?", "marbles"),
              ("Which of these two bags holds more coins?", "coins")]
QUANTITY_N = [2, 3, 5, 7, 10, 14, 20, 28, 40, 56]
WEIGHT_Q = ["Which of these two is heavier?", "Which of these two weighs more?",
            "Which of these two would be harder to lift?"]
OBJECTS = {"a mouse": 0.02, "an apple": 0.2, "a book": 1, "a cat": 4, "a bicycle": 15, "a dog": 25,
           "a washing machine": 70, "a motorcycle": 200, "a piano": 300, "a horse": 500, "a cow": 700,
           "a car": 1500, "an elephant": 5000, "a bus": 12000, "a blue whale": 150000}
REPLY = "Reply with only the letter A or B."


def user_of(question, a, b):
    return "\n".join([question, f"A) {a}", f"B) {b}", REPLY])


def main():
    cfg = load_config(argparse.Namespace(config=str(P3 / "config.yaml"), dry_run=True, quick=False))
    cfg["prompts"]["question"] = list(cfg["prompts"]["question"]) + [
        "Which of these two outcomes should happen?", "Of these two outcomes, which one would you pick?",
        "Which of these two outcomes is preferable?"]
    vals = json.load(open(HERE / "values.json"))
    b, u = vals["b"], vals["u"]
    rng = random.Random(20261102)
    rows = [json.loads(l) for l in open(P5 / "prompts.jsonl", encoding="utf-8")]
    reals = sorted(k for k in u if not k.startswith("N_"))

    def opt(key, q):
        return {"good": key.split("~")[0], **({"mirror": True} if "~" in key else {}), "q": q}

    out = []
    for r in rows:
        if r["group"] == "test" and r["kind"] in ("within", "cross", "mirror"):
            oa, ob = opt(r["key_a"], r["q_a"]), opt(r["key_b"], r["q_b"])
            _, user = build_prompt(cfg, r["wording"], "strangers", [oa, ob], None)
            assert user == r["user"], r["id"]
            cands = [k for k in reals if k != r["key_a"] and k.split("~")[0] != r["key_b"].split("~")[0]]
            kd = rng.choice(cands)
            _, duser = build_prompt(cfg, r["wording"], "strangers", [opt(kd, r["q_a"]), ob], None)
            out.append({"id": "train|" + r["id"], "set": "value_train", "user": r["user"], "donor_user": duser,
                        "key_a": r["key_a"], "key_b": r["key_b"], "q_a": r["q_a"], "q_b": r["q_b"], "donor_key_a": kd,
                        "v_a": u[r["key_a"]] + b * math.log(r["q_a"]), "v_b": u[r["key_b"]] + b * math.log(r["q_b"]),
                        "donor_v_a": u[kd] + b * math.log(r["q_a"])})
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        if r["group"] == "donor":
            rec = by_id[r["recipient"]]
            out.append({"id": "test|" + rec["id"], "set": "value_test", "user": rec["user"], "donor_user": r["user"],
                        "key_a": rec["key_a"], "key_b": rec["key_b"], "q_a": rec["q_a"], "q_b": rec["q_b"],
                        "donor_key_a": r["key_a"], "v_a": rec["v_a"], "v_b": rec["v_b"], "donor_v_a": r["v_a"]})
    for w, (q, noun) in enumerate(QUANTITY_Q):
        for na in QUANTITY_N:
            for nb in QUANTITY_N:
                if na == nb:
                    continue
                flip = [n for n in QUANTITY_N if (n > nb) != (na > nb) and n != nb]
                if not flip:                                   # B is the largest or smallest: no flip possible
                    continue
                nd = rng.choice(flip)
                out.append({"id": f"quantity|{na},{nb}|w{w}", "set": "quantity", "user": user_of(q, f"{na} {noun}", f"{nb} {noun}"),
                            "donor_user": user_of(q, f"{nd} {noun}", f"{nb} {noun}"), "a": na, "b": nb, "donor_a": nd})
    objs = list(OBJECTS)
    pairs = [(x, y) for x in objs for y in objs if x != y]
    for w, q in enumerate(WEIGHT_Q):
        for x, y in rng.sample(pairs, 100):
            flip = [z for z in objs if z not in (x, y) and (OBJECTS[z] > OBJECTS[y]) != (OBJECTS[x] > OBJECTS[y])]
            if not flip:
                continue
            z = rng.choice(flip)
            out.append({"id": f"weight|{x},{y}|w{w}", "set": "weight", "user": user_of(q, x, y),
                        "donor_user": user_of(q, z, y), "a": x, "b": y, "donor_a": z})
    with open(HERE / "pairs.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(out), dict(Counter(r["set"] for r in out)))
    for s in ("quantity", "weight"):
        r = next(x for x in out if x["set"] == s)
        print(f"--- {s}\n{r['user']}\n(donor A: {r['donor_user'].splitlines()[1]})")


if __name__ == "__main__":
    main()
