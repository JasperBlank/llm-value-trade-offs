"""Step 0: check the machine, load the model once, and show exactly what it will see.

Writes results/<run>/example_prompts.txt with one fully formatted prompt per
condition, so you can read (and later quote) the real text the model gets.
Then asks the model one decision question to prove everything works.
"""
import platform

from common import (decision_system, decision_user, get_backend, load_config, log, out_path,
                    parse_args, parse_decision, protect_from)


def main():
    args = parse_args(__doc__)
    cfg = load_config(args)
    log(f"Python {platform.python_version()}")
    if not cfg["dry_run"]:
        import torch
        import transformers
        log(f"torch {torch.__version__}, transformers {transformers.__version__}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            log(f"GPU: {p.name}, {p.total_memory / 1e9:.0f} GB")
            if p.total_memory < 22e9 and "8B" in cfg["model_id"]:
                log("WARNING: under 22 GB of GPU memory; Qwen3-8B may not fit. Use a 24 GB+ GPU.")
        else:
            log("WARNING: no GPU visible to torch.")

    backend = get_backend(cfg)
    sc = cfg["scenarios"][0]
    lines = []
    for goal in (1, 0):
        for mon in (1, 0):
            prompt = backend.chat(decision_system(cfg, sc, goal, mon), decision_user(cfg, sc, 1, 0))
            lines.append(f"===== goal={goal} monitored={mon} (scenario {sc['id']}, harm level 1, A = goal option)"
                         f" =====\n{prompt}\n")
    path = out_path(cfg, "example_prompts.txt")
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {path}")

    meta = {"kind": "decision", "scenario": sc["id"], "scenario_idx": 0, "level": 1, "goal": 1,
            "monitored": 1, "order": 0}
    prompt = backend.chat(decision_system(cfg, sc, 1, 1), decision_user(cfg, sc, 1, 0))
    texts = backend.generate([prompt, prompt], cfg["generation"], steer=None, seed=cfg["seed"], metas=[meta, meta])
    for t in texts:
        letter = parse_decision(t)
        print("\n----- model answer -----\n" + t)
        print(f"----- parsed: {letter}  (chose the {'care' if protect_from(letter, 0) else 'goal'} option)"
              if letter else "----- parsed: no DECISION line found")
    log("Setup check finished.")


if __name__ == "__main__":
    main()
