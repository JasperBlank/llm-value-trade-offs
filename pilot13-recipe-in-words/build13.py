"""Pilot 13's material (DESIGN.md section 2): the ten feature questions, the 24 new sentences, and the trade-offs that
price the new sentences against pilot 10's 96. Writes sentences13.json."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P10 = HERE.parent / "pilot10-full-range"

BENEFICIARY = "The people involved are strangers in your town whom you will never meet."
N_PEOPLE = 8
NEW_PAIRS_N = [(2, 32), (8, 8), (32, 2)]

# name, question, label at 0, label at 9
FEATURES = [
    ("size", "How big a difference would this make to each person's life?", "almost none", "life-changing"),
    ("severity", "How serious is the problem or need that this outcome deals with?", "there is no problem", "extremely serious"),
    ("duration", "How long would the effect of this outcome last?", "a few minutes", "the rest of their lives"),
    ("health_safety", "How much does this outcome concern people's physical health or safety?", "not at all", "entirely"),
    ("moral", "How much does this outcome involve putting right something that was morally wrong?", "not at all", "entirely"),
    ("relationships", "How much is this outcome about relationships and caring for other people?", "not at all", "entirely"),
    ("meaning", "How much does this outcome add meaning or purpose to people's lives?", "not at all", "a great deal"),
    ("pleasure", "How much is this outcome about pleasure or fun?", "not at all", "entirely"),
    ("legacy", "How much would this outcome leave something lasting for other people or future generations?", "nothing", "a great deal"),
    ("concreteness", "How concrete and certain is this outcome?", "vague", "completely concrete"),
]
WORDS = [f[0] for f in FEATURES[1:]]          # the recipe in words: F2-F10

# 24 new sentences, one per value, of mixed size; not paraphrases of pilot 10's 96
NEW = [
    ("Achievement", "small", "each solve a tricky puzzle they had been stuck on"),
    ("Beauty", "medium", "each watch a spectacular sunset over the sea"),
    ("Caring", "large", "are each able to give a seriously ill family member the care they need to recover"),
    ("Defending", "small", "each speak up when a waiter is treated rudely"),
    ("Diversity", "medium", "each spend a month living with a family from another culture"),
    ("Doing", "large", "each turn the activity they love into their full-time job"),
    ("Embodying", "small", "each stick to a small resolution they made"),
    ("Experiencing", "medium", "each spend a week travelling somewhere they have never been"),
    ("Fairness", "large", "each have a decade of unequal pay made up to them in full"),
    ("Freedom", "small", "each get to decide where to go for lunch"),
    ("Happiness", "medium", "each have a month in which they feel truly content"),
    ("Justice", "large", "each see the company that poisoned their town held fully responsible"),
    ("Knowing", "small", "each find out how a magic trick works"),
    ("Legacy", "medium", "each plant a tree that will stand for a hundred years"),
    ("Longevity", "large", "each gain ten more years of healthy life"),
    ("Loyalty", "small", "each get a birthday call from an old friend"),
    ("Nurturing", "medium", "each help a teenager get through a difficult school year"),
    ("Pleasure", "large", "each enjoy a month-long holiday doing everything they love"),
    ("Preserving", "small", "each see a favourite old shop stay open another year"),
    ("Protection", "medium", "are each protected from a burglary"),
    ("Respect", "large", "each be honoured publicly for a lifetime of quiet work"),
    ("Spirituality", "small", "each feel a moment of peace during a quiet prayer or meditation"),
    ("Non-suffering", "medium", "are each spared a month of back pain"),
    ("Truth", "large", "each finally learn the truth about what happened to a parent they never knew"),
]


def feature_user(text, f):
    name, q, lo, hi = f
    return "\n".join([BENEFICIARY, q, f"{N_PEOPLE} people {text}", f"Reply with only a number from 0 ({lo}) to 9 ({hi})."])


def main():
    old = json.load(open(P10 / "sentences10.json"))["sentences"]
    texts = {s["text"] for s in old}
    assert all(t not in texts for _, _, t in NEW)
    assert len({v for v, _, _ in NEW}) == 24
    json.dump({"new": [{"value": v, "level": lev, "text": t} for v, lev, t in NEW],
               "features": [{"name": n, "question": q, "low": lo, "high": hi} for n, q, lo, hi in FEATURES]},
              open(HERE / "sentences13.json", "w"), indent=1)
    print(f"{len(NEW)} new sentences; {len(FEATURES)} features; {len(NEW) * len(old) * 2 * len(NEW_PAIRS_N)} new trade-offs")
    print(feature_user(NEW[0][2], FEATURES[2]))


if __name__ == "__main__":
    main()
