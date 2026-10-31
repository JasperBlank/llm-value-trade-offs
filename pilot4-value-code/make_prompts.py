"""Build pilot 4's prompt list (run once, locally; the output prompts.jsonl is committed).

Every strangers two-option prompt from pilot 3 (the gate prompts and the goal module), plus,
for each goal-module prompt, an identical twin without the goal line. Each prompt carries the
value labels from pilot 3's fit of Qwen3-14B (strangers): V = u_option + b * log(number).
Goal-module prompts carry the goal-free values (the objective's effect is what we look for).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P3 = HERE.parent / "pilot3-value-gate"
sys.path.insert(0, str(P3))
from common import build_prompt, goal_jobs, load_config, load_jsonl, option_key, test_jobs  # noqa: E402
from fit import UtilityFit, cells  # noqa: E402

TEST_KINDS = ("within", "cross", "mirror", "rvn", "neutral")


def main():
    cfg = load_config(argparse.Namespace(config=str(P3 / "config.yaml"), dry_run=True, quick=False))
    bal = json.load(open(P3 / "results" / "real" / "balance_14b.json"))
    rows = load_jsonl(P3 / "results" / "real" / "test_14b.jsonl")
    C = [c for c in cells(rows) if c["kind"] in ("within", "cross", "mirror", "rvn") and c["beneficiary"] == "strangers"]
    fit = UtilityFit(C)
    u = {k: float(v) for k, v in zip(fit.keys, fit.u)}
    b = float(fit.b)

    def labels(options):
        ka, kb = option_key(options[0]), option_key(options[1])
        qa, qb = options[0]["q"], options[1]["q"]
        return {"key_a": ka, "key_b": kb, "q_a": qa, "q_b": qb,
                "v_a": u[ka] + b * np.log(qa), "v_b": u[kb] + b * np.log(qb)}

    out = []
    tests = [j for j in test_jobs(cfg, bal) if j["meta"]["beneficiary"] == "strangers"
             and j["meta"]["kind"] in TEST_KINDS and len(j["meta"]["options"]) == 2]
    goals = [j for j in goal_jobs(cfg) if j["meta"]["beneficiary"] == "strangers"]
    for j in tests + goals:
        m = j["meta"]
        out.append({"id": j["id"], "group": "test" if m["kind"] in TEST_KINDS else "goal", "kind": m["kind"],
                    "goal_good": m["goal_good"], "wording": m["phrasing"], "system": j["system"], "user": j["user"],
                    **labels(m["options"])})
    for j in goals:                                     # twins: the same prompt without the goal line
        m = j["meta"]
        system, user = build_prompt(cfg, m["phrasing"], m["beneficiary"], m["options"], None)
        assert user == j["user"] and system is None
        out.append({"id": j["id"] + "|twin", "group": "twin", "kind": m["kind"] + "_twin", "goal_good": m["goal_good"],
                    "wording": m["phrasing"], "system": None, "user": user, **labels(m["options"])})
    # 500 prompts recorded at every token (stratified over the test kinds, seeded).
    rng = random.Random(20261031)
    test_idx = [i for i, p in enumerate(out) if p["group"] == "test" and p["kind"] != "neutral"]
    full = set(rng.sample(test_idx, 500))
    with open(HERE / "prompts.jsonl", "w", encoding="utf-8") as f:
        for i, p in enumerate(out):
            p["index"], p["full"] = i, i in full
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    json.dump({"b": b, "u": u, "source": "pilot 3, Qwen3-14B, strangers cells (within, cross, mirror, rvn)"},
              open(HERE / "values.json", "w"), indent=2)
    from collections import Counter
    print(len(out), "prompts:", dict(Counter(p["kind"] for p in out)))


if __name__ == "__main__":
    main()
