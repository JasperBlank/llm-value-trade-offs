"""Build pilot 8's outcomes and prompts (run once, locally; the files are committed).

Revised 3 Nov (Jasper): only domain and size. 24 outcomes: 3 domains × 8 sizes, all gains for
"people", so worth can be traced as a curve over size within each domain.

outcomes.json   the 24 outcomes (with each size in a common unit: euros or hours) and the 16 / 8 split
behaviour.jsonl every unordered pair of outcomes, both orders, 3 numbers of people, 4 wordings
pairs.jsonl     recipient / donor pairs that differ only in option A's outcome:
                  ing_size        A's size one step up, same domain (4 option-B contexts)
                  ing_time_vs_money, ing_health_vs_money   A's domain changed at the same size rank
                  das_train / das_test   outcome swaps within the 16 training / 8 held-out outcomes
"""
import itertools
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENEFICIARY = "The people involved are strangers in your town whom you will never meet."
QUESTIONS = ["Which of these two outcomes would you rather happen?", "Which of these two outcomes do you prefer?",
             "Only one of these two outcomes can happen. Which would you choose?",
             "If only one of these two outcomes could happen, which should it be?"]
REPLY = "Reply with only the letter A or B."
HOURS = [("10 minutes", 1 / 6), ("an hour", 1), ("an afternoon", 4), ("a day", 24), ("a week", 168),
         ("a month", 720), ("half a year", 4380), ("a year", 8760)]
SIZES = {"money": [(f"€{v:,}", v) for v in (2, 5, 20, 50, 200, 1000, 5000, 20000)],
         "time": [(f"{t} of free time", h) for t, h in HOURS],
         "health": [(f"{t} of good health", h) for t, h in HOURS]}
UNIT = {"money": "euro", "time": "hour", "health": "hour"}
NUMBERS = [(2, 8), (4, 4), (8, 2)]
TEST_RANKS = {"money": [1, 4, 6], "time": [2, 5], "health": [1, 3, 6]}     # held out: interpolations only


def outcomes():
    out = []
    for d, sizes in SIZES.items():
        for r, (phrase, amount) in enumerate(sizes):
            out.append({"id": f"{d}|{r}", "domain": d, "rank": r, "phrase": phrase, "amount": amount, "unit": UNIT[d]})
    return out


def text(o, n):
    return f"{n} people each gain {o['phrase']}"


def user(a, na, b, nb, w):
    return "\n".join([BENEFICIARY, QUESTIONS[w], f"A) {text(a, na)}", f"B) {text(b, nb)}", REPLY])


def main():
    rng = random.Random(20261103)
    O = outcomes()
    by_id = {o["id"]: o for o in O}
    ids = [o["id"] for o in O]
    test = [f"{d}|{r}" for d, rs in TEST_RANKS.items() for r in rs]
    train = [i for i in ids if i not in test]
    json.dump({"outcomes": O, "train": train, "test": test, "example": text(by_id["health|4"], 8)},
              open(HERE / "outcomes.json", "w"), indent=1, ensure_ascii=False)
    rows = []
    for i, j in itertools.combinations(range(len(O)), 2):
        for a, b in ((O[i], O[j]), (O[j], O[i])):
            for na, nb in NUMBERS:
                for w in range(len(QUESTIONS)):
                    rows.append({"a": a["id"], "b": b["id"], "na": na, "nb": nb, "wording": w, "user": user(a, na, b, nb, w)})
    with open(HERE / "behaviour.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    contexts = [f"{d}|{r}" for d, r in (("money", 3), ("time", 4), ("health", 2), ("money", 6), ("time", 1), ("health", 7))]
    pairs = []

    def add(setname, o, d):
        cs = [c for c in contexts if c not in (o["id"], d["id"])][:4]
        for k, c in enumerate(cs):
            w = k % len(QUESTIONS)
            pairs.append({"set": setname, "a": o["id"], "donor_a": d["id"], "b": c, "wording": w,
                          "user": user(o, 4, by_id[c], 4, w), "donor_user": user(d, 4, by_id[c], 4, w)})
    for o in O:
        if o["rank"] < 7:
            add("ing_size", o, by_id[f"{o['domain']}|{o['rank'] + 1}"])
        if o["domain"] == "money":
            add("ing_time_vs_money", o, by_id[f"time|{o['rank']}"])
            add("ing_health_vs_money", o, by_id[f"health|{o['rank']}"])
    for split, pool, n in (("das_train", train, 1440), ("das_test", test, 480)):
        for _ in range(n):
            a, d = rng.sample(pool, 2)
            c = rng.choice([x for x in ids if x not in (a, d)])
            nn = rng.choice([2, 4, 8])
            w = rng.randrange(len(QUESTIONS))
            pairs.append({"set": split, "a": a, "donor_a": d, "b": c, "wording": w, "n": nn,
                          "user": user(by_id[a], nn, by_id[c], nn, w), "donor_user": user(by_id[d], nn, by_id[c], nn, w)})
    with open(HERE / "pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(O), "outcomes;", len(rows), "behaviour prompts;", dict(Counter(p["set"] for p in pairs)))
    print(rows[0]["user"])
    print("examples:", [text(by_id[i], 4) for i in ("money|7", "time|0", "health|4")])


if __name__ == "__main__":
    main()
