"""Pilot 18: fetch what the run compares against (DESIGN.md).

  python fetch_theirs.py model                            the model at Utility Engineering's revision; prints its path
  python fetch_theirs.py theirs --out DIR --paper DIR     their released results for llama-31-8b-instruct (compact), the
                                                          data repo's file list, and the paper's figure pages

Everything except the model's path goes to stderr, so `MODEL=$(python3 fetch_theirs.py model)` works.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"     # their models.yaml
DATA_REPO = "mmazeika/emergent-values-data"
KEY = "llama-31-8b-instruct"
PAPER = "https://arxiv.org/pdf/2502.08640"
FIGURES = ("Figure 4:", "Figure 7:", "Figure 8:", "Figure 23:")


def say(*a):
    print(*a, file=sys.stderr, flush=True)


def model():
    from huggingface_hub import snapshot_download
    pats = ["*.json", "*.safetensors", "tokenizer*"]
    try:
        path = snapshot_download(REPO_ID, revision=REVISION, allow_patterns=pats)
        say(f"model {REPO_ID} at their revision {REVISION[:7]}")
    except Exception as e:  # noqa: BLE001
        say(f"revision {REVISION[:7]} not available ({e!r}); using main")
        path = snapshot_download(REPO_ID, allow_patterns=pats)
    print(path)


def compact(data):
    """Their results file without the raw responses: outcomes, utilities, metrics, and every asked pair's P(A)."""
    g = data.get("graph_data", {})
    return {
        "options": g.get("options"),
        "utilities": data.get("utilities"),
        "metrics": data.get("metrics"),
        "holdout_metrics": data.get("holdout_metrics"),
        "edges": {k: v["probability_A"] for k, v in g.get("edges", {}).items()},
        "training_edges": g.get("training_edges"),
        "holdout_edge_indices": g.get("holdout_edge_indices"),
        "other_keys": sorted(k for k in data if k not in ("graph_data", "utilities", "metrics", "holdout_metrics")),
    }


def theirs(out: Path, paper: Path):
    from huggingface_hub import HfApi, hf_hub_download
    out.mkdir(parents=True, exist_ok=True)
    info = {"data_repo": DATA_REPO}
    api = HfApi()
    files, rtype = None, None
    for rt in ("model", "dataset"):
        try:
            files = [(f.path, getattr(f, "size", None)) for f in api.list_repo_tree(DATA_REPO, repo_type=rt, recursive=True)]
            rtype = rt
            break
        except Exception as e:  # noqa: BLE001
            say(f"listing {DATA_REPO} as {rt}: {e!r}")
    info["repo_type"], info["files"] = rtype, files
    say(f"{DATA_REPO} ({rtype}): {len(files or [])} files")
    for name, size in files or []:
        say(f"  {name}  {size}")
    wanted = [n for n, s in (files or []) if n.endswith(".zip") and (n == "options_hierarchical.zip" or "transitiv" in n.lower())
              and (s is None or s < 20e9)]
    for zname in wanted:
        try:
            local = hf_hub_download(DATA_REPO, zname, repo_type=rtype)
        except Exception as e:  # noqa: BLE001
            say(f"download {zname}: {e!r}")
            continue
        with zipfile.ZipFile(local) as z:
            names = z.namelist()
            mine = [n for n in names if f"/{KEY}/" in "/" + n and not n.endswith("/")]
            info[zname] = {"members": len(names), "llama_members": mine}
            say(f"{zname}: {len(names)} members; {KEY}: {mine}")
            for n in mine:
                base = Path(n).name
                with z.open(n) as f:
                    raw = f.read()
                if base.startswith("results_") and base.endswith(".json") and not base.startswith("results_utilities_"):
                    data = json.loads(raw)
                    c = compact(data)
                    json.dump(c, open(out / f"results_compact_{KEY}.json", "w"))
                    say(f"their results: {len(c['edges'])} pairs, {len(c['holdout_edge_indices'] or [])} held out; "
                        f"holdout metrics {c['holdout_metrics']}")
                elif base.endswith(".json"):
                    with gzip.open(out / (base + ".gz"), "wb") as g:
                        g.write(raw)
                elif base.endswith(".txt") and len(raw) < 1e6:
                    (out / base).write_bytes(raw)
    json.dump(info, open(out / "fetch_info.json", "w"), indent=1)
    try:
        fetch_paper(paper)
    except Exception as e:  # noqa: BLE001
        say(f"paper: {e!r}")


def fetch_paper(paper: Path):
    import fitz  # pymupdf
    paper.mkdir(parents=True, exist_ok=True)
    pdf = paper / "ue.pdf"
    req = urllib.request.Request(PAPER, headers={"User-Agent": "pilot18-research-script"})
    pdf.write_bytes(urllib.request.urlopen(req, timeout=180).read())
    doc = fitz.open(pdf)
    say(f"paper: {doc.page_count} pages")
    for i, page in enumerate(doc):
        text = page.get_text()
        for fig in FIGURES:
            if fig in text:
                name = paper / f"page{i + 1:03d}_{fig[:-1].replace(' ', '')}.png"
                page.get_pixmap(dpi=200).save(name)
                say(f"  {fig} on page {i + 1}: {name.name}")
    pdf.unlink()      # only the figure pages are kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["model", "theirs"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--paper", type=Path)
    a = ap.parse_args()
    if a.what == "model":
        model()
    else:
        theirs(a.out, a.paper)


if __name__ == "__main__":
    main()
