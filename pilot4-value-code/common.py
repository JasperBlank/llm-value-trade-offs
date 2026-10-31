"""Shared code for pilot 4: settings, prompts and labels, token positions, small helpers."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
REGIONS = ["template_start", "beneficiary", "question", "a_label", "a_number", "a_text", "b_label", "b_number",
           "b_text", "reply", "template_end"]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def deep_update(base, upd):
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def parse_args(description, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--quick", action="store_true", help="small model and few prompts, to test the pipeline")
    ap.add_argument("--model-path", default=None, help="override the model (a local folder, for tests)")
    ap.add_argument("--acts-dir", default=None, help="override where activations are stored")
    ap.add_argument("--random", action="store_true",
                    help="the control: the same architecture with random weights (test prompts only)")
    if extra:
        extra(ap)
    return ap.parse_args()


def load_config(args) -> dict:
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.quick:
        cfg = deep_update(cfg, cfg.get("quick", {}))
    if getattr(args, "model_path", None):
        cfg["model"] = args.model_path
    if getattr(args, "acts_dir", None):
        cfg["acts_dir"] = args.acts_dir
    cfg["quick_mode"] = bool(args.quick)
    cfg["random"] = bool(getattr(args, "random", False))
    if cfg["random"]:
        cfg["acts_dir"] = str(Path(cfg["acts_dir"]) / "random")
        cfg["results_dir"] = cfg["results_dir"] + "_random"
        cfg["only_groups"] = ["test"]
        cfg["no_full"] = True
    Path(cfg["acts_dir"]).mkdir(parents=True, exist_ok=True)
    res = ROOT / cfg["results_dir"]
    res.mkdir(parents=True, exist_ok=True)
    cfg["res"] = str(res)
    return cfg


def load_prompts(cfg) -> list[dict]:
    rows = [json.loads(l) for l in open(ROOT / "prompts.jsonl", encoding="utf-8")]
    if cfg.get("limit"):
        # Quick mode: a few prompts of every kind, keeping goal prompts with their twins.
        rng = np.random.default_rng(0)
        by_kind = {}
        for r in rows:
            by_kind.setdefault(r["kind"], []).append(r)
        per = max(cfg["limit"] // len(by_kind), 4)
        keep_ids = set()
        for k, rs in by_kind.items():
            if k.endswith("_twin"):
                continue
            for r in rng.choice(rs, size=min(per, len(rs)), replace=False):
                keep_ids.add(r["id"])
                if r["group"] == "goal":
                    keep_ids.add(r["id"] + "|twin")
        rows = [r for r in rows if r["id"] in keep_ids]
        full_n = cfg.get("full_subset", 12)
        cands = [r for r in rows if r["group"] == "test" and r["kind"] != "neutral"]
        full = {r["id"] for r in cands[:full_n]}
        for r in rows:
            r["full"] = r["id"] in full
    if cfg.get("only_groups"):
        rows = [r for r in rows if r["group"] in cfg["only_groups"]]
    if cfg.get("no_full"):
        for r in rows:
            r["full"] = False
    for i, r in enumerate(rows):
        r["index"] = i
    return rows


def acts_path(cfg, name):
    return Path(cfg["acts_dir"]) / name


# --------------------------------------------------------------------------
# Token positions, from character offsets in the chat-formatted prompt
# --------------------------------------------------------------------------
def char_landmarks(text: str, user: str) -> dict:
    """Character spans in the full chat string for each part of the user message."""
    start = text.index(user)
    lines = user.split("\n")
    marks, pos = {"user_start": start, "user_end": start + len(user)}, start
    for line in lines:
        if line.startswith("A) ") or line.startswith("B) "):
            s = line[0].lower()
            marks[f"{s}_label"] = (pos, pos + 3)
            num_len = len(line[3:].split(" ")[0])
            marks[f"{s}_number"] = (pos + 3, pos + 3 + num_len)
            marks[f"{s}_text"] = (pos + 3 + num_len, pos + len(line))
        elif line.startswith("Reply with"):
            marks["reply"] = (pos, pos + len(line))
        elif line.startswith("The people involved"):
            marks["beneficiary"] = (pos, pos + len(line))
        else:
            marks["question"] = (pos, pos + len(line))
        pos += len(line) + 1
    return marks


def token_at(offsets, char_idx: int) -> int:
    """Index of the token whose span contains char_idx (the last such token)."""
    hit = [i for i, (a, b) in enumerate(offsets) if a <= char_idx < b]
    if not hit:
        hit = [i for i, (a, b) in enumerate(offsets) if a <= char_idx <= b and b > a]
    if not hit:
        raise ValueError(f"no token covers character {char_idx}")
    return hit[-1]


def positions_for(offsets, marks: dict, n_tokens: int, names: list[str]) -> list[int]:
    out = []
    for nm in names:
        if nm == "a_num":
            out.append(token_at(offsets, marks["a_number"][1] - 1))
        elif nm == "a_end":
            out.append(token_at(offsets, marks["a_text"][1] - 1))
        elif nm == "b_num":
            out.append(token_at(offsets, marks["b_number"][1] - 1))
        elif nm == "b_end":
            out.append(token_at(offsets, marks["b_text"][1] - 1))
        elif nm == "reply_end":
            out.append(token_at(offsets, marks["reply"][1] - 1))
        elif nm.startswith("t-"):
            out.append(n_tokens - int(nm[2:]))
        else:
            raise ValueError(nm)
    return out


def token_regions(offsets, marks: dict) -> np.ndarray:
    """Region id (index into REGIONS) for every token."""
    reg = np.zeros(len(offsets), dtype=np.int16)
    spans = [(REGIONS.index(k), marks[k]) for k in REGIONS if k in marks and isinstance(marks[k], tuple)]
    for i, (a, b) in enumerate(offsets):
        if b <= marks["user_start"]:
            reg[i] = REGIONS.index("template_start")
            continue
        if a >= marks["user_end"]:
            reg[i] = REGIONS.index("template_end")
            continue
        mid = a if b == a else (a + b - 1) / 2
        for rid, (s, e) in spans:
            if s <= mid < e:
                reg[i] = rid
                break
        else:
            reg[i] = REGIONS.index("question")      # newlines between lines
    return reg


def label_arrays(rows: list[dict]) -> dict:
    """Numeric labels for every prompt (goal-free values for goal prompts)."""
    va = np.array([r["v_a"] for r in rows], float)
    vb = np.array([r["v_b"] for r in rows], float)
    return {"v_a": va, "v_b": vb, "dv": va - vb,
            "log_qa": np.log([r["q_a"] for r in rows]), "log_qb": np.log([r["q_b"] for r in rows])}
