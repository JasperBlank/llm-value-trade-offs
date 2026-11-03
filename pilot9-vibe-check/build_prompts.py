"""Builds ratings9.jsonl: pilot 8's 48 sentences at 2, 4 and 8 people, asked four ways (DESIGN.md section 2)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P8 = HERE.parent / "pilot8-intrinsic-values"

BENEFICIARY = "The people involved are strangers in your town whom you will never meet."
WORDINGS = [
    ("How good would it be if the following happened?", "Reply with only a number from 0 (not good at all) to 9 (extremely good)."),
    ("How much better would the world be if the following happened?", "Reply with only a number from 0 (not at all) to 9 (much better)."),
    ("How bad or good would it be if the following happened?", "Reply with only a number from 0 (very bad) to 9 (very good)."),
    ("How much does it matter to you that the following happens?", "Reply with only a number from 0 (not at all) to 9 (extremely)."),
]
SIZES = [2, 4, 8]


def main():
    vals = json.load(open(P8 / "values.json"))["values"]
    first = json.loads(open(P8 / "ratings.jsonl").readline())
    assert first["user"].split("\n")[0] == BENEFICIARY, "context line differs from pilot 8"
    assert first["user"].split("\n")[1] == WORDINGS[3][0], "pilot 8's question line differs"
    rows = []
    for w, (q, r) in enumerate(WORDINGS):
        for v in vals:
            for phr, text in enumerate(v["phrasings"]):
                for n in SIZES:
                    rows.append({"value": v["name"], "phr": phr, "n": n, "wording": w,
                                 "user": "\n".join([BENEFICIARY, q, f"{n} people {text}", r])})
    # pilot 8's own rating prompt (wording 3, 4 people) must come out identical
    p8 = {(x["value"], x["phr"]): x["user"] for x in map(json.loads, open(P8 / "ratings.jsonl")) if x["wording"] == 0}
    for x in rows:
        if x["wording"] == 3 and x["n"] == 4:
            assert x["user"] == p8[(x["value"], x["phr"])].replace(p8[(x["value"], x["phr"])].split("\n")[3], WORDINGS[3][1]), x
    with open(HERE / "ratings9.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    print(f"{len(rows)} prompts; example:\n{rows[0]['user']}")


if __name__ == "__main__":
    main()
