"""Shared code for pilot 3: settings, goods, prompts, the two-stage job lists, and the backends.

The answer is read from the FIRST answer token, with reasoning switched off: for every prompt
we record the probability on "A", "B" (and "C" in three-option menus). Every comparison is
shown in both option orders and six wordings; the analysis averages over them.
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
GROUPS = ("strangers", "close")


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
    ap.add_argument("--quick", action="store_true", help="small model and few prompts, to test the pipeline on the pod")
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


# --------------------------------------------------------------------------
# Goods and prompts
# --------------------------------------------------------------------------
def goods_in(cfg, valence: str | None = None) -> list[dict]:
    gs = [g for g in cfg["goods"].values() if g["set"] == "test" and (valence is None or g["valence"] == valence)]
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


def build_prompt(cfg, wording: int, beneficiary: str | None, options: list[dict], goal: str | None):
    P = cfg["prompts"]
    system = P["goal_line"].format(goal=goal) if goal else None
    lines = [P["beneficiaries"][beneficiary]] if beneficiary else []
    lines.append(P["question"][wording] if len(options) == 2 else P["question3"][wording])
    for letter, o in zip(LETTERS, options):
        lines.append(f"{letter}) {option_text(cfg, o)}")
    lines.append(P["reply2"] if len(options) == 2 else P["reply3"])
    return system, "\n".join(lines)


# --------------------------------------------------------------------------
# Pairs of goods
# --------------------------------------------------------------------------
def real_pairs(cfg) -> list[tuple[dict, dict, str]]:
    """The 15 real pairs: 3 among gains, 3 among prevented harms, 9 across. (first, second, kind)"""
    gains, harms = goods_in(cfg, "gain"), goods_in(cfg, "harm_prevented")
    out = []
    for trio in (gains, harms):
        for a, b in itertools.combinations(trio, 2):
            out.append(({"good": a["id"]}, {"good": b["id"]}, "within"))
    for g in gains:
        for h in harms:
            out.append(({"good": g["id"]}, {"good": h["id"]}, "cross"))
    return out


def mirror_pairs(cfg) -> list[tuple[dict, dict, str]]:
    """The gain wording of each prevented harm against each gain (description invariance)."""
    return [({"good": h["id"], "mirror": True}, {"good": g["id"]}, "mirror")
            for h in goods_in(cfg, "harm_prevented") for g in goods_in(cfg, "gain")]


def pair_name(a: dict, b: dict) -> str:
    return f"{option_key(a)}|{option_key(b)}"


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------
def _job(cfg, kind, options, wording, beneficiary, goal_good=None, extra=None):
    goal = cfg["goods"][goal_good]["goal"] if goal_good else None
    system, user = build_prompt(cfg, wording, beneficiary, options, goal)
    jid = "|".join([kind, ",".join(f"{option_key(o)}:{o['q']}" for o in options),
                    f"w{wording}", beneficiary or "-", goal_good or "-"])
    meta = {"kind": kind, "options": options, "phrasing": wording, "beneficiary": beneficiary,
            "goal_good": goal_good, **(extra or {})}
    return {"id": jid, "system": system, "user": user, "meta": meta}


def _cell_jobs(cfg, kind, a, b, cells, beneficiaries, wordings, goal_good=None, extra=None):
    """Every (qa, qb) cell in both orders, every wording, every beneficiary group."""
    jobs = []
    for qa, qb in cells:
        oa, ob = {**a, "q": int(qa)}, {**b, "q": int(qb)}
        for opts in ([oa, ob], [ob, oa]):
            for w in wordings:
                for ben in beneficiaries:
                    jobs.append(_job(cfg, kind, opts, w, ben, goal_good, extra))
    return jobs


def _unique(jobs):
    seen, out = set(), []
    for j in jobs:
        if j["id"] not in seen:
            seen.add(j["id"])
            out.append(j)
    return out


def scout_jobs(cfg) -> list[dict]:
    """Stage 1: a coarse sweep of every real and mirror pair, to find its balance point."""
    S = cfg["scout"]
    jobs = []
    for a, b, kind in real_pairs(cfg) + mirror_pairs(cfg):
        jobs += _cell_jobs(cfg, "scout", a, b, S["cells"], GROUPS, S["wordings"], extra={"pair_kind": kind})
    return _unique(jobs)


def stage2_cells(R: float, cfg) -> list[tuple[int, int]]:
    """Quantities around the balance ratio R (units of the second good per unit of the first)."""
    T = cfg["test"]
    cells = []
    for base in T["bases"]:
        for f in T["factors"]:
            rho = R * f
            if rho >= 1:
                qa, qb = base, int(round(base * rho))
            else:
                qa, qb = int(round(base / rho)), base
            qa, qb = min(max(qa, 1), T["q_max"]), min(max(qb, 1), T["q_max"])
            if (qa, qb) not in cells:
                cells.append((qa, qb))
    return cells


def balance_cell(R: float, base: int, cfg) -> tuple[int, int]:
    """The stage-2 cell at the balance point (factor 1) for one size."""
    if R >= 1:
        qa, qb = base, int(round(base * R))
    else:
        qa, qb = int(round(base / R)), base
    q_max = cfg["test"]["q_max"]
    return min(max(qa, 1), q_max), min(max(qb, 1), q_max)


def test_jobs(cfg, balance: dict) -> list[dict]:
    """Stage 2: the gate and the extras. `balance` maps pair name -> group -> R."""
    W = list(range(cfg["wordings"]))
    fixed = list(itertools.product(cfg["grid_fixed"], cfg["grid_fixed"]))
    neutrals = goods_in(cfg, "neutral")
    reals = goods_in(cfg, "gain") + goods_in(cfg, "harm_prevented")
    jobs = []
    # Real and mirror pairs, centred on each pair's balance point, per beneficiary group.
    for a, b, kind in real_pairs(cfg) + mirror_pairs(cfg):
        for ben in GROUPS:
            R = balance[pair_name(a, b)][ben]
            jobs += _cell_jobs(cfg, kind, a, b, stage2_cells(R, cfg), [ben], W, extra={"balance": R})
    for r in reals:                                       # real against neutral (criterion 1b)
        for n in neutrals:
            jobs += _cell_jobs(cfg, "rvn", {"good": r["id"]}, {"good": n["id"]}, fixed, GROUPS, W)
    for a, b in itertools.combinations(neutrals, 2):      # neutral against neutral (reported)
        jobs += _cell_jobs(cfg, "neutral", {"good": a["id"]}, {"good": b["id"]}, fixed, GROUPS, W)
    for r in reals:                                       # closeness: same good, family versus strangers
        jobs += _cell_jobs(cfg, "mixed", {"good": r["id"], "who": "close"}, {"good": r["id"], "who": "strangers"},
                           fixed, [None], W)
    archive = next(n for n in neutrals if n["domain"] == "archive")
    for a, b, kind in real_pairs(cfg):                    # three-option menus at the balance point
        if kind != "within":
            continue
        for ben in GROUPS:
            R = balance[pair_name(a, b)][ben]
            for base in cfg["test"]["bases"]:
                qa, qb = balance_cell(R, base, cfg)
                for third_type in ("decoy", "neutral"):
                    third = ({"good": a["good"], "q": max(1, qa // 2)} if third_type == "decoy"
                             else {"good": archive["id"], "q": qa})
                    trio = [{**a, "q": qa}, {**b, "q": qb}, third]
                    for rot in range(3):
                        opts = trio[rot:] + trio[:rot]
                        for w in W:
                            jobs.append(_job(cfg, "menu", opts, w, ben,
                                             extra={"focal": [a["good"], b["good"]], "focal_q": [qa, qb],
                                                    "third_type": third_type, "rotation": rot, "base": base}))
    return _unique(jobs)


def goal_jobs(cfg) -> list[dict]:
    """The goal module: version 1 (objective on a real good), version 2 (on the archive good)."""
    G, W = cfg["goal"], list(range(cfg["wordings"]))
    gains, harms = goods_in(cfg, "gain"), goods_in(cfg, "harm_prevented")
    jobs = []
    for t in gains + harms:
        for p in (gains if t["valence"] == "gain" else harms):
            if p["id"] != t["id"]:
                cells = list(itertools.product(G["v1_target"], G["v1_partner"]))
                jobs += _cell_jobs(cfg, "goal1", {"good": t["id"]}, {"good": p["id"]}, cells, GROUPS, W, goal_good=t["id"])
    archive = next(n for n in goods_in(cfg, "neutral") if n["domain"] == "archive")
    for r in gains + harms:
        cells = list(itertools.product(G["v2_neutral"], G["v2_real"]))
        jobs += _cell_jobs(cfg, "goal2", {"good": archive["id"]}, {"good": r["id"]}, cells, GROUPS, W,
                           goal_good=archive["id"])
    return _unique(jobs)


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
    todo.sort(key=lambda j: len(j["meta"]["options"]))      # keep two- and three-option prompts apart
    bs, t0, n = int(cfg["batch_size"]), time.time(), 0
    with open(path, "a", encoding="utf-8") as f:
        for k in range(0, len(todo), bs):
            chunk = todo[k:k + bs]
            for m in sorted({len(j["meta"]["options"]) for j in chunk}):
                group = [j for j in chunk if len(j["meta"]["options"]) == m]
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
    """Built-in truth. Per-unit values by (valence, domain):
         gains 1.0 / 1.5 / 2.0, prevented harms 1.4 / 2.4 / 0.8 (health / money / time);
         neutral goods 0.02 per unit (low, not zero); the gain wording of a prevented harm 0.9x;
         the user's family 1.5x for every good; no decoy effect.
       Choice: log-odds = 2.3 x (log value A - log value B) + a letter bias of +0.8 for "A".
       Goal module: an objective on a real good makes it win regardless (a rule); an objective
       on the archive good gives it 3 per unit (extra value: 1 file is worth 1.25-3.75 people).
       Model '8b' (the negative control): a letter bias of -3 for "A" and a random distortion for
       every pair of goods, so it is NOT consistent.
       Two of the six wordings are a little noisier; none falls below 95% on the letters."""
    BASE = {("gain", "health"): 1.0, ("gain", "money"): 1.5, ("gain", "time"): 2.0,
            ("harm_prevented", "health"): 1.4, ("harm_prevented", "money"): 2.4, ("harm_prevented", "time"): 0.8,
            ("neutral", "archive"): 0.02, ("neutral", "office"): 0.02, ("neutral", "supplies"): 0.02}

    def __init__(self, cfg, model_key: str):
        self.cfg, self.model_key = cfg, model_key

    def chat(self, system, user):
        return (system or "") + "\n" + user

    def _value(self, o, meta):
        g = self.cfg["goods"][o["good"]]
        u = self.BASE[(g["valence"], g["domain"])]
        if o.get("mirror"):
            u *= 0.9
        if o.get("who") == "close" or (meta.get("beneficiary") == "close" and g["valence"] != "neutral"):
            u *= 1.5
        if meta.get("goal_good") == o["good"]:
            u = 1e6 if g["valence"] != "neutral" else 3.0
        return u * o["q"]

    def letter_probs(self, prompts, n_letters, metas=None):
        out = []
        for meta in metas:
            rng = np.random.default_rng(int(hashlib.md5(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:8], 16))
            z = 2.3 * np.log(np.array([self._value(o, meta) for o in meta["options"]]))
            if self.model_key == "8b":
                for i, o in enumerate(meta["options"]):
                    others = sorted(x["good"] for x in meta["options"] if x is not o)
                    seed = int(hashlib.md5((o["good"] + "|" + ",".join(others)).encode()).hexdigest()[:8], 16)
                    z[i] += np.random.default_rng(seed).normal(0, 1.5)
                z += np.array([-3.0, 0.0, 0.5][:n_letters])
            else:
                z += np.array([0.8, 0.0, 0.3][:n_letters])
            z += rng.normal(0, 0.3 if meta["phrasing"] in (2, 5) else 0.15, size=n_letters)
            p = np.exp(z - z.max())
            p /= p.sum()
            out.append(p * 0.995)
        return np.array(out)


def get_backend(cfg, model_key: str):
    return FakeBackend(cfg, model_key) if cfg["dry_run"] else HFBackend(cfg, cfg["models"][model_key])
