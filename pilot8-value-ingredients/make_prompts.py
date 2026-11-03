"""Build pilot 8's outcomes and prompts (run once, locally; the files are committed).

outcomes.json   the 48 outcomes (5 ingredients crossed) and the 32 / 16 split for the worth direction
behaviour.jsonl every unordered pair of outcomes, both orders, 3 numbers of people, 2 wordings
pairs.jsonl     recipient / donor pairs that differ only in option A's outcome:
                  ing_<ingredient>   minimal pairs (one ingredient changed), 4 option-B contexts
                  das_train / das_test   outcome swaps within the 32 training / 16 held-out outcomes
"""
import itertools
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENEFICIARY = "The people involved are strangers in your town whom you will never meet."
QUESTIONS = ["Which of these two outcomes would you rather happen?", "Which of these two outcomes do you prefer?"]
REPLY = "Reply with only the letter A or B."
DOMAINS = {"money": ("€20", "€200"), "time": ("an hour of free time", "a week of free time"),
           "health": ("a day of good health", "a month of good health")}
FRAMES = {"gain": "each gain", "loss": "are each prevented from losing"}
SIZES, WHO, NEED = ["small", "large"], ["adults", "children"], ["none", "poverty"]
NUMBERS = [(2, 8), (4, 4), (8, 2)]


def outcomes():
    out = []
    for d, f, s, w, n in itertools.product(DOMAINS, FRAMES, SIZES, WHO, NEED):
        out.append({"id": f"{d}|{f}|{s}|{w}|{n}", "domain": d, "frame": f, "size": s, "who": w, "need": n})
    return out


def text(o, n):
    who = o["who"] + (" living in poverty" if o["need"] == "poverty" else "")
    return f"{n} {who} {FRAMES[o['frame']]} {DOMAINS[o['domain']][SIZES.index(o['size'])]}"


def user(a, na, b, nb, w):
    return "\n".join([BENEFICIARY, QUESTIONS[w], f"A) {text(a, na)}", f"B) {text(b, nb)}", REPLY])


def main():
    rng = random.Random(20261103)
    O = outcomes()
    by_id = {o["id"]: o for o in O}
    ids = [o["id"] for o in O]
    test = sorted(rng.sample(ids, 16))
    train = [i for i in ids if i not in test]
    json.dump({"outcomes": O, "train": train, "test": test,
               "example": text(by_id["time|loss|large|children|poverty"], 8)}, open(HERE / "outcomes.json", "w"), indent=1)
    rows = []
    for i, j in itertools.combinations(range(len(O)), 2):
        for a, b in ((O[i], O[j]), (O[j], O[i])):
            for na, nb in NUMBERS:
                for w in range(len(QUESTIONS)):
                    rows.append({"a": a["id"], "b": b["id"], "na": na, "nb": nb, "wording": w, "user": user(a, na, b, nb, w)})
    with open(HERE / "behaviour.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # minimal pairs: recipient has the first level, donor the second; 4 option-B contexts each
    contexts = rng.sample(ids, 8)
    flips = {"frame": ("frame", "gain", "loss"), "size": ("size", "small", "large"), "who": ("who", "adults", "children"),
             "need": ("need", "none", "poverty"), "time_vs_money": ("domain", "money", "time"),
             "health_vs_money": ("domain", "money", "health")}
    pairs = []
    for name, (key, lo, hi) in flips.items():
        for o in O:
            if o[key] != lo:
                continue
            d = dict(o, **{key: hi})
            did = f"{d['domain']}|{d['frame']}|{d['size']}|{d['who']}|{d['need']}"
            cs = [c for c in contexts if c not in (o["id"], did)][:4]
            for k, c in enumerate(cs):
                w = k % 2
                pairs.append({"set": f"ing_{name}", "a": o["id"], "donor_a": did, "b": c, "wording": w,
                              "user": user(o, 4, by_id[c], 4, w), "donor_user": user(by_id[did], 4, by_id[c], 4, w)})
    for split, pool, n in (("das_train", train, 1440), ("das_test", test, 480)):
        for _ in range(n):
            a, d = rng.sample(pool, 2)
            c = rng.choice([x for x in ids if x not in (a, d)])
            nn = rng.choice([2, 4, 8])
            w = rng.randrange(2)
            pairs.append({"set": split, "a": a, "donor_a": d, "b": c, "wording": w, "n": nn,
                          "user": user(by_id[a], nn, by_id[c], nn, w), "donor_user": user(by_id[d], nn, by_id[c], nn, w)})
    with open(HERE / "pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(O), "outcomes;", len(rows), "behaviour prompts;", dict(Counter(p["set"] for p in pairs)))
    print(rows[0]["user"])
    print("example outcome:", text(by_id["time|loss|large|children|poverty"], 8))


if __name__ == "__main__":
    main()
