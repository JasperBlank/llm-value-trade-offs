"""Shared code for pilot 5: settings, prompts, token regions, the model, batching and hooks."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
# Parts of the prompt. "separator" = the newline tokens between the lines of the user message
# (pilot 4 counted them as "question"); "last" = the final token, where the answer is read.
REGIONS = ["template_start", "beneficiary", "question", "a_label", "a_number", "a_text", "b_label", "b_number",
           "b_text", "reply", "separator", "template_end", "last"]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def deep_update(base, upd):
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def parse_args(description, extra=None):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--quick", action="store_true", help="small model and few prompts, to test the pipeline")
    ap.add_argument("--model-path", default=None, help="override the model (a local folder, for tests)")
    ap.add_argument("--acts-dir", default=None, help="override where recordings are stored")
    ap.add_argument("--results-dir", default=None, help="override where results are written")
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
    if getattr(args, "results_dir", None):
        cfg["results_dir"] = args.results_dir
    cfg["quick_mode"] = bool(args.quick)
    Path(cfg["acts_dir"]).mkdir(parents=True, exist_ok=True)
    res = Path(cfg["results_dir"])
    res = res if res.is_absolute() else ROOT / res
    res.mkdir(parents=True, exist_ok=True)
    cfg["res"] = str(res)
    return cfg


def acts_path(cfg, name):
    return Path(cfg["acts_dir"]) / name


def load_prompts(cfg) -> list[dict]:
    """All prompts (prompts.jsonl). Quick mode keeps a seeded handful of each group, with goal
    prompts kept together with their twins and fresh prompts with their donors."""
    rows = [json.loads(l) for l in open(ROOT / "prompts.jsonl", encoding="utf-8")]
    lim = cfg.get("limit")
    if lim:
        rng = np.random.default_rng(0)
        keep = set()
        for grp, n in lim.items():
            cands = [r for r in rows if r["group"] == grp]
            if grp == "test":                         # stratified over kinds
                kinds = sorted({r["kind"] for r in cands})
                for k in kinds:
                    ck = [r for r in cands if r["kind"] == k]
                    for r in rng.choice(ck, size=min(max(n // len(kinds), 2), len(ck)), replace=False):
                        keep.add(r["id"])
            else:
                for r in rng.choice(cands, size=min(n, len(cands)), replace=False):
                    keep.add(r["id"])
        for r in rows:                                # partners
            if r["group"] == "twin" and r["id"][:-len("|twin")] in keep:
                keep.add(r["id"])
            if r["group"] == "donor" and r["recipient"] in keep:
                keep.add(r["id"])
        rows = [r for r in rows if r["id"] in keep]
    for i, r in enumerate(rows):
        r["index"] = i
    return rows


def values():
    v = json.load(open(ROOT / "values.json"))
    return float(v["b"]), {k: float(x) for k, x in v["u"].items()}


def base_good(key: str) -> str:
    return key.split("~")[0]


# --------------------------------------------------------------------------
# Token positions and regions, from character offsets in the chat-formatted prompt
# --------------------------------------------------------------------------
def char_landmarks(text: str, user: str) -> dict:
    start = text.index(user)
    marks, pos = {"user_start": start, "user_end": start + len(user)}, start
    for line in user.split("\n"):
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


def token_regions(text: str, offsets, marks: dict) -> np.ndarray:
    """Region id (index into REGIONS) for every token."""
    R = REGIONS.index
    reg = np.zeros(len(offsets), dtype=np.int16)
    spans = [(R(k), marks[k]) for k in REGIONS if isinstance(marks.get(k), tuple)]
    for i, (a, b) in enumerate(offsets):
        if b <= marks["user_start"]:
            reg[i] = R("template_start")
            continue
        if a >= marks["user_end"]:
            reg[i] = R("template_end")
            continue
        if b > a and text[a:b].strip() == "":
            reg[i] = R("separator")
            continue
        mid = a if b == a else (a + b - 1) / 2
        for rid, (s, e) in spans:
            if s <= mid < e:
                reg[i] = rid
                break
        else:
            reg[i] = R("separator")
    reg[-1] = R("last")
    return reg


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
def load_model(cfg):
    """Tokenizer and model, eager attention (so attention weights can be read) throughout pilot 5."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading {cfg['model']} on {dev} ...")
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    dtype = getattr(torch, cfg["dtype"]) if dev == "cuda" else torch.float32
    kw = {"attn_implementation": "eager"}
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=dtype, **kw)
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model, dev


def model_shape(model) -> dict:
    c = model.config
    H, KV = c.num_attention_heads, c.num_key_value_heads
    hd = getattr(c, "head_dim", None) or c.hidden_size // H
    return {"n_layers": c.num_hidden_layers, "d": c.hidden_size, "H": H, "KV": KV, "hd": hd, "rep": H // KV}


def letter_ids(tok) -> list[list[int]]:
    """Token ids for "A" and "B" (with and without a leading space)."""
    out = []
    for letter in "AB":
        ids = {e[0] for v in (letter, " " + letter) if len(e := tok.encode(v, add_special_tokens=False)) == 1}
        out.append(sorted(ids))
    return out


def main_letter_ids(tok) -> tuple[int, int]:
    """The single tokens "A" and "B" (the answer as the model writes it), for the logit direction."""
    a, b = tok.encode("A", add_special_tokens=False), tok.encode("B", add_special_tokens=False)
    assert len(a) == 1 and len(b) == 1, (a, b)
    return a[0], b[0]


def encode_rows(tok, rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        msgs = ([{"role": "system", "content": r["system"]}] if r.get("system") else []) + \
               [{"role": "user", "content": r["user"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        marks = char_landmarks(text, r["user"])
        out.append({"ids": enc["input_ids"], "regions": token_regions(text, enc["offset_mapping"], marks)})
    return out


def make_batch(encs, idx, pad_id, torch, dev):
    """Left-padded batch: the last real token is at position -1 for every prompt."""
    T = max(len(encs[i]["ids"]) for i in idx)
    ids = torch.full((len(idx), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(idx), T), dtype=torch.long)
    regs = np.full((len(idx), T), -1, dtype=np.int16)
    for j, i in enumerate(idx):
        n = len(encs[i]["ids"])
        ids[j, T - n:] = torch.tensor(encs[i]["ids"])
        att[j, T - n:] = 1
        regs[j, T - n:] = encs[i]["regions"]
    return ids.to(dev), att.to(dev), regs


def forward(model, ids, att, torch, **kw):
    with torch.no_grad():
        try:
            return model(input_ids=ids, attention_mask=att, logits_to_keep=1, **kw)
        except TypeError:
            return model(input_ids=ids, attention_mask=att, **kw)


def logodds(logits_last, lids, torch) -> np.ndarray:
    """log P(A)/P(B), with the letter probabilities summed over spellings (as in pilots 2-4)."""
    p = torch.softmax(logits_last.float(), -1)
    a, b = p[:, lids[0]].sum(-1), p[:, lids[1]].sum(-1)
    pn = (a / (a + b)).clamp(1e-4, 1 - 1e-4)
    return torch.log(pn / (1 - pn)).cpu().numpy()


def p_letters(logits_last, lids, torch):
    p = torch.softmax(logits_last.float(), -1)
    return p[:, lids[0]].sum(-1).cpu().numpy(), p[:, lids[1]].sum(-1).cpu().numpy()


def batches(n, bs):
    for k0 in range(0, n, bs):
        yield list(range(k0, min(k0 + bs, n)))


def layer_output(output):
    return output[0] if isinstance(output, tuple) else output


def replace_layer_output(output, h):
    return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h


class Hooks:
    """Register hooks and remove them all afterwards (use with `with`)."""

    def __init__(self):
        self.handles = []

    def fwd(self, module, fn):
        self.handles.append(module.register_forward_hook(fn))

    def pre(self, module, fn):
        self.handles.append(module.register_forward_pre_hook(fn))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


def bootstrap_mean_ci(x, reps=2000, seed=0, alpha=0.05):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(reps)])
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)
