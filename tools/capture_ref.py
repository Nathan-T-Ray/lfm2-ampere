#!/usr/bin/env python3
"""Capture the oracle's ground truth for gate C1 and the llama.cpp baseline.

Writes ref/oracle_ref.json containing, per suite case:
  - greedy token IDs (the hard C1 gate)
  - top-K logprobs at the first positions (logit-level comparison)
  - decode/prefill timings (the baseline to beat)
Determinism is asserted by repeating every case and requiring identical tokens.
"""
import json
import os
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from oracle import Oracle

ROOT = pathlib.Path(__file__).resolve().parent.parent
REF = pathlib.Path(os.environ.get("LK_REF", ROOT / "ref"))
N_PREDICT = int(os.environ.get("LK_NPREDICT", "128"))
N_PROBS = 20
REPEATS = 3


def main():
    o = Oracle()
    suite = json.loads((REF / "parity_suite.json").read_text())["cases"]
    results = []
    for case in suite:
        ids = case["prompt_ids"]
        runs = [o.greedy(ids, n_predict=N_PREDICT, n_probs=N_PROBS) for _ in range(REPEATS)]
        toks = [r.get("tokens") or [] for r in runs]
        deterministic = all(t == toks[0] for t in toks)
        r0 = runs[0]
        dts = [r["timings"]["predicted_per_second"] for r in runs if r.get("timings")]
        pts = [r["timings"]["prompt_per_second"] for r in runs if r.get("timings")]
        rec = {
            "name": case["name"],
            "n_prompt": len(ids),
            "greedy_tokens": toks[0],
            "n_generated": len(toks[0]),
            "deterministic": deterministic,
            "top_logprobs": Oracle.top_logprobs(r0, N_PROBS)[:16],
            "content": r0.get("content", ""),
            "decode_tok_s": statistics.median(dts) if dts else None,
            "prefill_tok_s": statistics.median(pts) if pts else None,
        }
        results.append(rec)
        flag = "OK " if deterministic else "NONDETERMINISTIC"
        print(f"  {case['name']:10s} gen={len(toks[0]):4d} decode={rec['decode_tok_s']:7.2f} tok/s "
              f"prefill={rec['prefill_tok_s']:8.2f} tok/s  [{flag}]")

    dec = [r["decode_tok_s"] for r in results if r["decode_tok_s"]]
    out = {
        "model": os.environ.get("LK_MODEL", ""),
        "engine": "llama.cpp/llama-server (oracle)",
        "n_predict": N_PREDICT,
        "decode_tok_s_median": statistics.median(dec) if dec else None,
        "decode_tok_s_min": min(dec) if dec else None,
        "decode_tok_s_max": max(dec) if dec else None,
        "cases": results,
    }
    (REF / "oracle_ref.json").write_text(json.dumps(out, indent=1))
    print(f"\nBASELINE llama.cpp decode = {out['decode_tok_s_median']:.2f} tok/s "
          f"(min {out['decode_tok_s_min']:.2f}, max {out['decode_tok_s_max']:.2f})")
    nb = 1466751488.0
    print(f"  => %.0f GB/s effective (%.1f%% of 1008 GB/s peak)" %
          (nb * out["decode_tok_s_median"] / 1e9, 100 * nb * out["decode_tok_s_median"] / 1e9 / 1008))
    print("wrote", REF / "oracle_ref.json")
    if not all(r["deterministic"] for r in results):
        print("WARNING: non-deterministic cases detected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
