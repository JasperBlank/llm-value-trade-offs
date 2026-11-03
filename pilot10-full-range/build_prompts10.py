"""Builds pilot 10's material (DESIGN.md section 2).

- sentences10.json: 96 outcome sentences, 24 values x 4: a small version, pilot 8's two sentences, a large version.
- menu_keys10.json: every ordered pair of different sentences at 2:32, 4:16, 8:8, 16:4 and 32:2 people
  (9,120 pairs x 5 = 45,600 trade-offs); the prompt text is built from these on the pod (`menu_prompt`).
- ratings10.jsonl: each sentence at 2, 4, 8 and 32 people, asked the four ways of pilot 9 (1,536 prompts).
"""
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P8 = HERE.parent / "pilot8-intrinsic-values"
P9 = HERE.parent / "pilot9-vibe-check"

BENEFICIARY = "The people involved are strangers in your town whom you will never meet."
Q2 = "Which of these two outcomes would you rather happen?"
R2 = "Reply with only the letter A or B."
PAIRS_N = [(2, 32), (4, 16), (8, 8), (16, 4), (32, 2)]
RATING_SIZES = [2, 4, 8, 32]

# small and large versions of each value; the medium ones are pilot 8's two sentences
SMALL_LARGE = {
    "Achievement": ("each finish a small task they had been putting off", "each achieve the greatest ambition of their life"),
    "Beauty": ("each notice a pretty flower on their way to work",
               "each see a sight so beautiful that they remember it for the rest of their lives"),
    "Caring": ("each get to make a cup of tea for someone they care about",
               "are each able to be there for a loved one through the hardest time of their life"),
    "Defending": ("each speak up for a colleague who was interrupted in a meeting", "each save a stranger from a violent attack"),
    "Diversity": ("each try a dish from a culture they have never encountered",
                  "each help build a town where people of every background are truly welcome"),
    "Doing": ("each get ten minutes to do something they enjoy",
              "each get to spend the rest of their working life doing the work they love most"),
    "Embodying": ("each keep a small promise they made to themselves",
                  "each become, over the years, exactly the kind of person they always wanted to be"),
    "Experiencing": ("each taste a fruit they have never tried before",
                     "each go on the journey of a lifetime that they had dreamed of for decades"),
    "Fairness": ("each get back a small amount they were overcharged on a bill",
                 "each have years of unfair treatment finally set right and fully compensated"),
    "Freedom": ("each get to choose which film to watch tonight", "are each freed after years of being wrongly imprisoned"),
    "Happiness": ("each smile at a funny joke", "each become deeply happy for the rest of their lives"),
    "Justice": ("each see someone who cut in line sent to the back", "each see the person who ruined their life brought to justice"),
    "Knowing": ("each learn a fun fact", "each gain a deep understanding that transforms how they see the world"),
    "Legacy": ("each leave a kind note that someone finds years later",
               "each create a lasting work that helps people for generations after they are gone"),
    "Longevity": ("each live one day longer", "each live twenty more years in good health"),
    "Loyalty": ("each get a message from a friend checking in on them",
                "each have a friend who stands by them through the worst crisis of their life"),
    "Nurturing": ("each help a child tie their shoes", "each raise a child who grows up healthy, kind and happy"),
    "Pleasure": ("each enjoy a piece of chocolate", "each enjoy a perfect week of rest, good food and fun"),
    "Preserving": ("each see an old bench in their park repaired", "each see their hometown saved from being destroyed by a flood"),
    "Protection": ("are each warned about a slippery step before they reach it", "are each saved from being killed in a fire"),
    "Respect": ("each get a polite nod from a stranger",
                "each finally earn the lasting respect of everyone who once looked down on them"),
    "Spirituality": ("each feel a brief sense of awe while looking at the stars",
                     "each find a deep sense of meaning and peace that stays with them for the rest of their lives"),
    "Non-suffering": ("are each spared a mild headache", "are each cured of a severe pain they have suffered from for years"),
    "Truth": ("each learn the real reason a meeting was moved", "each finally learn the truth about a lie that shaped their whole life"),
}


def load_sentences():
    return json.load(open(HERE / "sentences10.json"))["sentences"]


def menu_prompt(S, key):
    a, b, na, nb = key
    return "\n".join([BENEFICIARY, Q2, f"A) {na} people {S[a]['text']}", f"B) {nb} people {S[b]['text']}", R2])


def main():
    vals = json.load(open(P8 / "values.json"))["values"]
    assert {v["name"] for v in vals} == set(SMALL_LARGE)
    S = []
    for v in vals:
        small, large = SMALL_LARGE[v["name"]]
        S.append({"value": v["name"], "level": "small", "text": small, "p8": None})
        for i, t in enumerate(v["phrasings"]):
            S.append({"value": v["name"], "level": "medium", "text": t, "p8": i})
        S.append({"value": v["name"], "level": "large", "text": large, "p8": None})
    json.dump({"sentences": S}, open(HERE / "sentences10.json", "w"), indent=1)
    keys = [[a, b, na, nb] for a, b in itertools.permutations(range(len(S)), 2) for na, nb in PAIRS_N]
    json.dump(keys, open(HERE / "menu_keys10.json", "w"))
    # the rating prompts use pilot 9's four wordings exactly
    import importlib.util
    spec = importlib.util.spec_from_file_location("bp9", P9 / "build_prompts.py")
    bp9 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp9)
    with open(HERE / "ratings10.jsonl", "w") as f:
        for w, (q, r) in enumerate(bp9.WORDINGS):
            for i, s in enumerate(S):
                for n in RATING_SIZES:
                    f.write(json.dumps({"s": i, "n": n, "wording": w,
                                        "user": "\n".join([BENEFICIARY, q, f"{n} people {s['text']}", r])}) + "\n")
    # pilot 9's prompts (4 people) must reappear unchanged
    p9 = {x["user"] for x in map(json.loads, open(P9 / "ratings9.jsonl"))}
    mine = [json.loads(l) for l in open(HERE / "ratings10.jsonl")]
    same = [x for x in mine if x["user"] in p9]
    assert len(same) == 48 * 3 * 4, len(same)          # 48 sentences x sizes 2, 4, 8 x 4 wordings
    print(f"{len(S)} sentences; {len(keys)} trade-offs; {len(mine)} rating prompts ({len(same)} identical to pilot 9's)")
    print(menu_prompt(S, keys[0]))


if __name__ == "__main__":
    main()
