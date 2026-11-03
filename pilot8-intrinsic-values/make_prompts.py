"""Build pilot 8's prompts (run once, locally; the files are committed).

24 intrinsic values (Clearer Thinking's 22 categories, plus Non-suffering and Truth), each as an
outcome for strangers in two phrasings.

values.json      the values, sentences, and the 12 / 12 split for the held-out tests
behaviour.jsonl  two-option trade-offs: every pair of values, 4 phrasing combinations, both orders,
                 3 numbers of people, 2 wordings
menus.jsonl      three-option menus and the matching two-option prompts (menu invariance)
ratings.jsonl    each sentence on its own: "how much does it matter to you", 0-9, 2 wordings
pairs.jsonl      recipient / donor pairs that differ only in option A's value (das_train: all three
                 values from the training half; das_test: all from the held-out half)
rating_pairs.jsonl  rating prompts, recipient and donor differ only in the value
"""
import itertools
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENEFICIARY = "The people involved are strangers in your town whom you will never meet."
Q2 = ["Which of these two outcomes would you rather happen?", "Which of these two outcomes do you prefer?"]
Q3 = ["Which of these three outcomes would you rather happen?", "Which of these three outcomes do you prefer?"]
QR = ["How much does it matter to you that the following happens?", "How important is it to you that the following happens?"]
R2, R3 = "Reply with only the letter A or B.", "Reply with only the letter A, B or C."
RR = "Reply with only a number from 0 (not at all) to 9 (extremely)."
NUMBERS = [(2, 8), (4, 4), (8, 2)]
VALUES = [
    ("Achievement", "each achieve a goal they have worked towards for a long time", "each finally succeed at something they have spent years trying to do"),
    ("Beauty", "each spend an afternoon somewhere breathtakingly beautiful", "each get to see something so beautiful it moves them deeply"),
    ("Caring", "each get the chance to help someone they care about", "are each able to look after a loved one who needs them"),
    ("Defending", "each successfully stand up for someone who was being mistreated", "each protect a stranger from being bullied"),
    ("Diversity", "each get to know someone very different from themselves", "each become part of a community made up of many different kinds of people"),
    ("Doing", "each spend a day doing an activity they love", "each get to spend time on a hobby they are passionate about"),
    ("Embodying", "each manage to act according to their ideals in a hard situation", "each live up to the kind of person they want to be when it is difficult"),
    ("Experiencing", "each have a new, memorable experience they have never had before", "each try something exciting they have always wanted to experience"),
    ("Fairness", "each have an unfair decision against them reversed", "each receive the fair share they had wrongly been denied"),
    ("Freedom", "each gain the freedom to decide how they spend their own time", "are each no longer prevented from making their own choices about their lives"),
    ("Happiness", "each feel happier for a week", "are each in a noticeably better mood for several days"),
    ("Justice", "each see someone who wronged them held accountable", "each see a person who harmed them face fair consequences"),
    ("Knowing", "each come to understand something important about the world", "each finally learn the answer to a question that has long puzzled them"),
    ("Legacy", "each create something that will outlast them", "each leave behind something that will be remembered after they are gone"),
    ("Longevity", "each live a year longer", "each have their life extended by a year"),
    ("Loyalty", "each have a friend stand by them through a hard time", "each find that the people close to them stay faithful when it matters"),
    ("Nurturing", "each help a young person grow and flourish", "each get to guide a child through an important stage of growing up"),
    ("Pleasure", "each enjoy a delicious meal", "each have a thoroughly enjoyable evening"),
    ("Preserving", "each see a place they love protected from destruction", "each see an old tradition they cherish kept alive"),
    ("Protection", "are each kept safe from a danger that threatened them", "are each shielded from a threat to their safety"),
    ("Respect", "are each treated with respect by people who looked down on them", "each finally receive the recognition and respect they deserve"),
    ("Spirituality", "each have a deeply meaningful spiritual experience", "each feel a profound connection to something greater than themselves"),
    ("Non-suffering", "are each spared a week of pain", "are each relieved of a painful condition for a week"),
    ("Truth", "are each told the truth instead of being lied to", "each learn the truth about something they had been deceived about"),
]


def sentence(v, phr, n):
    return f"{n} people {VALUES[v][1 + phr]}"


def two(a, pa, na, b, pb, nb, w):
    return "\n".join([BENEFICIARY, Q2[w], f"A) {sentence(a, pa, na)}", f"B) {sentence(b, pb, nb)}", R2])


def three(opts, w, n=4):
    lines = [BENEFICIARY, Q3[w]] + [f"{L}) {sentence(v, p, n)}" for L, (v, p) in zip("ABC", opts)] + [R3]
    return "\n".join(lines)


def rating(v, p, w, n=4):
    return "\n".join([BENEFICIARY, QR[w], sentence(v, p, n), RR])


def main():
    rng = random.Random(20261103)
    V = len(VALUES)
    names = [x[0] for x in VALUES]
    test = sorted(rng.sample(range(V), 12))
    train = [i for i in range(V) if i not in test]
    json.dump({"values": [{"name": x[0], "phrasings": [x[1], x[2]]} for x in VALUES], "train": [names[i] for i in train],
               "test": [names[i] for i in test], "example": two(10, 0, 4, 22, 0, 4, 0)}, open(HERE / "values.json", "w"), indent=1)
    rows = []
    for a, b in itertools.combinations(range(V), 2):
        for pa, pb in itertools.product((0, 1), (0, 1)):
            for (x, px), (y, py) in (((a, pa), (b, pb)), ((b, pb), (a, pa))):
                for na, nb in NUMBERS:
                    for w in (0, 1):
                        rows.append({"a": names[x], "pa": px, "b": names[y], "pb": py, "na": na, "nb": nb, "wording": w,
                                     "user": two(x, px, na, y, py, nb, w)})
    with open(HERE / "behaviour.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    menus = []
    triples = rng.sample(list(itertools.permutations(range(V), 3)), 300)
    for k, (x, y, z) in enumerate(triples):
        p, w = k % 2, (k // 2) % 2
        for (a, b) in ((x, y), (y, x)):
            menus.append({"kind": "three", "x": names[a], "y": names[b], "z": names[z], "phr": p, "wording": w,
                          "user": three([(a, p), (b, p), (z, p)], w)})
            menus.append({"kind": "two", "x": names[a], "y": names[b], "z": names[z], "phr": p, "wording": w,
                          "user": two(a, p, 4, b, p, 4, w)})
    with open(HERE / "menus.jsonl", "w", encoding="utf-8") as f:
        for r in menus:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(HERE / "ratings.jsonl", "w", encoding="utf-8") as f:
        for v in range(V):
            for p in (0, 1):
                for w in (0, 1):
                    f.write(json.dumps({"value": names[v], "phr": p, "wording": w, "user": rating(v, p, w)}, ensure_ascii=False) + "\n")
    with open(HERE / "rating_pairs.jsonl", "w", encoding="utf-8") as f:
        for p, w in ((0, 0), (1, 1)):
            for x, y in itertools.permutations(range(V), 2):
                f.write(json.dumps({"value": names[x], "donor_value": names[y], "phr": p, "wording": w,
                                    "user": rating(x, p, w), "donor_user": rating(y, p, w)}, ensure_ascii=False) + "\n")
    pairs = []
    for split, pool, n in (("das_train", train, 1440), ("das_test", test, 480)):
        for _ in range(n):
            x, y, z = rng.sample(pool, 3)
            px, py, pz = rng.randrange(2), rng.randrange(2), rng.randrange(2)
            nn, w = rng.choice([2, 4, 8]), rng.randrange(2)
            pairs.append({"set": split, "a": names[x], "donor_a": names[y], "b": names[z], "wording": w, "n": nn,
                          "user": two(x, px, nn, z, pz, nn, w), "donor_user": two(y, py, nn, z, pz, nn, w)})
    with open(HERE / "pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(V, "values; behaviour", len(rows), "; menus", len(menus), "; ratings", V * 4, "; rating pairs", 2 * V * (V - 1), "; pairs", len(pairs))
    print(rows[0]["user"]); print(menus[0]["user"]); print(rating(22, 0, 0))
    print("held out:", [names[i] for i in test])


if __name__ == "__main__":
    main()
