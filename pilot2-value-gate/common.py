"""Shared code for pilot 2: settings, goods, prompt building, jobs, and the model backends.

The model's answer is read from its FIRST answer token, with reasoning switched off: for
every prompt we record how much probability it puts on "A", "B" (and "C" in three-option
menus). That gives an exact choice probability from one forward pass, with no sampling.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
LETTERS = "ABC"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Settings and files
# --------------------------------------------------------------------------
def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def parse_args(description: str, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--quick", action="store_true", help="tiny model and few prompts, to test the pipeline on the pod")
    ap.add_argument("--dry-run", action="store_true", help="no model: a fake model with known values, to test the code")
    if extra:
        extra(ap)
    return ap.parse_args()


def load_config(args) -> dict:
    cfg = load_yaml(args.config)
    if args.quick:
        cfg = deep_update(cfg, cfg.get("quick", {}))
    cfg["dry_run"], cfg["quick_mode"] = bool(args.dry_run), bool(args.quick)
    tag = "dryrun" if args.dry_run else ("quick" if args.quick else "real")
    out = ROOT / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    cfg["out_dir"] = str(out)
    cfg["goods"] = {g["id"]: g for g in load_yaml(ROOT / "goods.yaml")["goods"]}
    cfg["prompts"] = load_yaml(ROOT / "prompts.yaml")
    return cfg


def out_path(cfg, name: str) -> Path:
    return Path(cfg["out_dir"]) / name


def load_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # half-written last line after a crash; it is simply redone
    return rows


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def load_locked(cfg, required: bool):
    p = out_path(cfg, "locked_format.json")
    if p.exists():
        return json.loads(p.read_text())
    if required and not (cfg["dry_run"] or cfg["quick_mode"]):
        raise SystemExit("No locked_format.json. Run select_format.py after the tune phase (and commit it) first.")
    return {"frame": "B", "shape": "log", "wide_pairs": [], "note": "fallback for dry-run/quick only"}


# --------------------------------------------------------------------------
# Goods and prompts
# --------------------------------------------------------------------------
def goods_in(cfg, set_name: str, valence: str | None = None) -> list[dict]:
    gs = [g for g in cfg["goods"].values() if g["set"] == set_name and (valence is None or g["valence"] == valence)]
    order = {"health": 0, "money": 1, "time": 2, "archive": 0, "office": 1, "supplies": 2}
    return sorted(gs, key=lambda g: order.get(g["domain"], 9))


def option_key(o: dict) -> str:
    """A unique name for 'this good, worded this way, for these people'."""
    return o["good"] + ("~gainword" if o.get("mirror") else "") + (f"@{o['who']}" if o.get("who") else "")


def option_text(cfg, o: dict) -> str:
    g = cfg["goods"][o["good"]]
    t = g["mirror"] if o.get("mirror") else g
    text = t["one"] if o["q"] == 1 else t["many"].format(n=o["q"])
    if o.get("who"):
        text = cfg["prompts"]["mixed_prefix"][o["who"]] + text
    return text


def build_prompt(cfg, frame: str, phrasing: int, beneficiary: str | None, options: list[dict], goal: str | None):
    P = cfg["prompts"]
    F = P["frames"][frame]
    system_parts = [F["system"][phrasing]] if F["system"][phrasing] else []
    if goal:
        system_parts.append(P["goal_line"].format(goal=goal))
    lines = [P["beneficiaries"][beneficiary]] if beneficiary else []
    lines.append(F["question"][phrasing] if len(options) == 2 else F["question3"][phrasing])
    for letter, o in zip(LETTERS, options):
        lines.append(f"{letter}) {option_text(cfg, o)}")
    lines.append(P["reply2"] if len(options) == 2 else P["reply3"])
    return (" ".join(system_parts) or None), "\n".join(lines)


# --------------------------------------------------------------------------
# Jobs: every prompt of a phase
# --------------------------------------------------------------------------
def _job(cfg, set_name, frame, kind, options, phrasing, beneficiary, goal_good=None, extra=None):
    goal = cfg["goods"][goal_good]["goal"] if goal_good else None
    system, user = build_prompt(cfg, frame, phrasing, beneficiary, options, goal)
    jid = "|".join([set_name, frame, kind, ",".join(f"{option_key(o)}:{o['q']}" for o in options),
                    f"p{phrasing}", beneficiary or "-", goal_good or "-"])
    meta = {"set": set_name, "frame": frame, "kind": kind, "options": options, "phrasing": phrasing,
            "beneficiary": beneficiary, "goal_good": goal_good, **(extra or {})}
    return {"id": jid, "system": system, "user": user, "meta": meta}


def _pair_jobs(cfg, set_name, frame, kind, a, b, grid, beneficiaries, goal_good=None, wa=None, wb=None):
    """Both orders × 3 phrasings × beneficiary groups × every quantity combination."""
    jobs = []
    for qa, qb in itertools.product(grid, grid):
        oa = {"good": a, "q": qa, **(wa or {})}
        ob = {"good": b, "q": qb, **(wb or {})}
        for opts in ([oa, ob], [ob, oa]):
            for ph in range(3):
                for ben in beneficiaries:
                    jobs.append(_job(cfg, set_name, frame, kind, opts, ph, ben, goal_good))
    return jobs


def _domain_key(valence, d1, d2):
    return "|".join([valence] + sorted([d1, d2]))


def build_jobs(cfg, phase: str, frame: str, locked: dict | None = None) -> list[dict]:
    """phase: tune | test (the gate and the extras) | goal (the goal module, test goods)."""
    set_name = "tune" if phase == "tune" else "test"
    bens = ["strangers", "close"]
    wide = set(map(tuple, (locked or {}).get("wide_pairs", [])))
    gains, harms = goods_in(cfg, set_name, "gain"), goods_in(cfg, set_name, "harm_prevented")
    neutrals, reals = goods_in(cfg, set_name, "neutral"), goods_in(cfg, set_name, "gain") + goods_in(cfg, set_name, "harm_prevented")
    red = cfg["grid_reduced"]
    jobs = []

    if phase == "goal":
        for t in reals:                                   # version 1: objective on a real good
            for partner in (gains if t["valence"] == "gain" else harms):
                if partner["id"] != t["id"]:
                    jobs += _pair_jobs(cfg, set_name, frame, "goal1", t["id"], partner["id"], red, bens, goal_good=t["id"])
        box = next(n for n in neutrals if n["domain"] == "archive")
        for r in reals:                                   # version 2: objective on a neutral good
            jobs += _pair_jobs(cfg, set_name, frame, "goal2", box["id"], r["id"], red, bens, goal_good=box["id"])
        return jobs

    # Within-valence pairs (full grid; widened for a domain pair flagged in the tune phase).
    for trio in (gains, harms):
        for a, b in itertools.combinations(trio, 2):
            grid = cfg["grid_wide"] if _domain_key(a["valence"], a["domain"], b["domain"]) in wide else cfg["grid_full"]
            jobs += _pair_jobs(cfg, set_name, frame, "within", a["id"], b["id"], grid, bens)
    for a, b in itertools.combinations(neutrals, 2):      # neutral against neutral
        jobs += _pair_jobs(cfg, set_name, frame, "neutral", a["id"], b["id"], red, bens)
    for g in gains:                                       # across valences
        for h in harms:
            jobs += _pair_jobs(cfg, set_name, frame, "cross", g["id"], h["id"], red, bens)
    for r in reals:                                       # real against neutral
        for n in neutrals:
            jobs += _pair_jobs(cfg, set_name, frame, "rvn", r["id"], n["id"], red, bens)
    for h in harms:                                       # description invariance: gain wording of a prevented harm
        for g in gains:
            jobs += _pair_jobs(cfg, set_name, frame, "mirror", h["id"], g["id"], red, bens, wa={"mirror": True})
    for r in reals:                                       # closeness: same good, family versus strangers
        jobs += _pair_jobs(cfg, set_name, frame, "mixed", r["id"], r["id"], red, [None],
                           wa={"who": "close"}, wb={"who": "strangers"})
    box = next(n for n in neutrals if n["domain"] == "archive")
    for trio in (gains, harms):                           # three-option menus (menu invariance)
        for a, b in itertools.combinations(trio, 2):
            for qa, qb in cfg["menu_quantities"]:
                for third_type in ("decoy", "neutral"):
                    third = {"good": a["id"], "q": max(1, qa // 2)} if third_type == "decoy" else {"good": box["id"], "q": qa}
                    base = [{"good": a["id"], "q": qa}, {"good": b["id"], "q": qb}, third]
                    for rot in range(3):
                        opts = base[rot:] + base[:rot]
                        for ph in range(3):
                            for ben in bens:
                                jobs.append(_job(cfg, set_name, frame, "menu", opts, ph, ben,
                                                 extra={"focal": [a["id"], b["id"]], "focal_q": [qa, qb],
                                                        "third_type": third_type, "rotation": rot}))
    return jobs


# --------------------------------------------------------------------------
# Running jobs (resumable)
# --------------------------------------------------------------------------
def run_jobs(backend, jobs: list[dict], path, cfg, label: str):
    path = Path(path)
    done = {r["id"] for r in load_jsonl(path)}
    todo = [j for j in jobs if j["id"] not in done]
    if not todo:
        log(f"{label}: all {len(jobs)} already done ({path.name})")
        return
    log(f"{label}: {len(todo)} to do ({len(done)} already in {path.name})")
    bs, t0, n = int(cfg["batch_size"]), time.time(), 0
    with open(path, "a", encoding="utf-8") as f:
        for k in range(0, len(todo), bs):
            chunk = todo[k:k + bs]
            n_opts = len(chunk[0]["meta"]["options"])
            same = [j for j in chunk if len(j["meta"]["options"]) == n_opts]
            rest = [j for j in chunk if len(j["meta"]["options"]) != n_opts]
            for group in (same, rest):
                if not group:
                    continue
                m = len(group[0]["meta"]["options"])
                probs = backend.letter_probs([backend.chat(j["system"], j["user"]) for j in group], m,
                                             metas=[j["meta"] for j in group])
                for j, p in zip(group, probs):
                    f.write(json.dumps({"id": j["id"], **j["meta"], "probs": [float(x) for x in p],
                                        "mass": float(sum(p))}, ensure_ascii=False) + "\n")
            f.flush()
            n += len(chunk)
            if (k // bs) % 20 == 0 or n == len(todo):
                rate = n / max(time.time() - t0, 1e-6)
                log(f"  {label}: {n}/{len(todo)}  ({rate:.0f}/s, about {(len(todo) - n) / max(rate, 1e-6) / 60:.0f} min left)")


# --------------------------------------------------------------------------
# Real model
# --------------------------------------------------------------------------
class HFBackend:
    def __init__(self, cfg: dict, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.cfg = torch, cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Loading {model_id} ...")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = getattr(torch, cfg.get("dtype", "bfloat16")) if self.device == "cuda" else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model = model.to(self.device).eval()
        # Token ids for each answer letter, with and without a leading space.
        self.letter_ids = []
        for L in LETTERS:
            ids = set()
            for v in (L, " " + L):
                enc = self.tok.encode(v, add_special_tokens=False)
                if len(enc) == 1:
                    ids.add(enc[0])
            if not ids:
                raise RuntimeError(f"No single token for the letter {L}")
            self.letter_ids.append(sorted(ids))
        log(f"Loaded. Letter tokens: {self.letter_ids}")

    def chat(self, system, user) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def letter_probs(self, prompts, n_letters: int, metas=None):
        torch = self.torch
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[:, -1, :].float()
        p = torch.softmax(logits, dim=-1)
        out = torch.stack([p[:, ids].sum(-1) for ids in self.letter_ids[:n_letters]], dim=-1)
        return out.cpu().numpy()


# --------------------------------------------------------------------------
# Fake model for --dry-run, with known values (the analysis must find them again)
# --------------------------------------------------------------------------
class FakeBackend:
    """Built-in truth (per-unit values; choice by log value, slope 2, a small letter bias):
      gains 1.0 / 1.5 / 2.0, harms prevented 1.2 / 2.4 / 0.8 (by domain health / money / time);
      neutral goods have no value; the gain wording of a prevented harm is valued 0.8x;
      the user's family is valued 1.3x (uniformly); no decoy effect.
      Goal module: an objective on a real good multiplies its value by 3 (extra value);
      an objective on a neutral good makes it win regardless (a rule).
      Model '4b' is inconsistent: every pair of goods gets its own random distortion.
      Frame A puts 3% of the probability outside the letters and is noisier than frame B."""
    BASE = {("gain", "health"): 1.0, ("gain", "money"): 1.5, ("gain", "time"): 2.0,
            ("harm_prevented", "health"): 1.2, ("harm_prevented", "money"): 2.4, ("harm_prevented", "time"): 0.8}

    def __init__(self, cfg, model_key: str):
        self.cfg, self.model_key = cfg, model_key

    def chat(self, system, user):
        return (system or "") + "\n" + user

    def _value(self, o, meta):
        g = self.cfg["goods"][o["good"]]
        u = self.BASE.get((g["valence"], g["domain"]), 0.0)
        if o.get("mirror"):
            u *= 0.8
        if o.get("who") == "close":
            u *= 1.3
        if meta.get("goal_good") == o["good"]:
            u = u * 3 if g["valence"] != "neutral" else 1e6
        return u * o["q"] + 0.05

    def letter_probs(self, prompts, n_letters, metas=None):
        out = []
        for meta in metas:
            rng = np.random.default_rng(int(hashlib.md5(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:8], 16))
            vals = np.array([self._value(o, meta) for o in meta["options"]])
            z = 2.0 * np.log(vals)
            if self.model_key == "4b":
                for i, o in enumerate(meta["options"]):
                    others = sorted(x["good"] for x in meta["options"] if x is not o)
                    seed = int(hashlib.md5((o["good"] + "|" + ",".join(others)).encode()).hexdigest()[:8], 16)
                    z[i] += np.random.default_rng(seed).normal(0, 1.5)
            z += np.array([0.2, 0.0, -0.1][:n_letters])
            z += rng.normal(0, 0.4 if meta["frame"] == "A" else 0.15, size=n_letters)
            p = np.exp(z - z.max())
            p /= p.sum()
            mass = 0.97 if meta["frame"] == "A" else 0.995
            out.append(p * mass)
        return np.array(out)


def get_backend(cfg, model_key: str):
    return FakeBackend(cfg, model_key) if cfg["dry_run"] else HFBackend(cfg, cfg["models"][model_key])
