"""Pilot 5: run the stages in order, loading the model once.

  python pipeline.py                      all stages, Qwen3-14B
  python pipeline.py --quick              all stages on Qwen3-0.6B and a few prompts (a pipeline test)
  python pipeline.py --stages steer,analyze

Stages: record, directions, steer (THE PRE-REGISTERED TEST), circuit, circuit_analyze, patch,
readers, objective, formats (the extension: other answer formats), analyze. See DESIGN.md.
"""
from __future__ import annotations

import importlib
import time

from common import encode_rows, load_config, load_model, load_prompts, log, parse_args

STAGES = ["record", "directions", "steer", "circuit", "circuit_analyze", "patch", "readers", "objective", "formats", "analyze"]


class Ctx:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rows = load_prompts(cfg)
        self._m = None
        self._encs = None

    def model(self):
        if self._m is None:
            self._m = load_model(self.cfg)
        return self._m

    def encs(self):
        if self._encs is None:
            tok = self.model()[0]
            t0 = time.time()
            self._encs = encode_rows(tok, self.rows)
            log(f"Tokenised {len(self.rows)} prompts in {time.time() - t0:.0f} s")
        return self._encs


def main():
    args = parse_args(__doc__, extra=lambda ap: ap.add_argument("--stages", default=",".join(STAGES)))
    cfg = load_config(args)
    ctx = Ctx(cfg)
    log(f"Pilot 5, {cfg['model']}, {len(ctx.rows)} prompts, results in {cfg['res']}")
    for s in args.stages.split(","):
        s = s.strip()
        if s not in STAGES:
            raise SystemExit(f"unknown stage {s}")
        t0 = time.time()
        log(f"===== stage {s} =====")
        importlib.import_module(s).run(ctx)
        log(f"===== stage {s} finished in {(time.time() - t0) / 60:.1f} min =====")


if __name__ == "__main__":
    main()
