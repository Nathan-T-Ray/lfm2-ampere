#!/usr/bin/env python3
"""Plot liquid-kernel vs llama.cpp across context, from 0 to the model maximum.

Three panels, because one number is not the answer:

  1. DECODE throughput against context.  This is the serving number and it is
     what "440 then dropping under 400" was about.
  2. THE RATIO.  Absolute tok/s falls for everyone -- attention reads the whole
     KV cache once per token, so KV traffic grows linearly with context while
     weight traffic is fixed.  The question that decides whether long context is
     a problem is whether the GAP survives, not whether the speed does.
  3. PREFILL.  A 131k system prompt has to be ingested before a single token is
     served, the engine has no batched prefill, and the decode number hides that
     entirely.

usage: tools/plotctx.py --out ref/ctx_compare.png
"""
import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load(p):
    f = ROOT / p
    return json.loads(f.read_text()) if f.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "ref" / "ctx_compare.png"))
    a = ap.parse_args()

    sets = [(n, load("ref/ctx_%s.json" % n)) for n in ("regular", "code")]
    sets = [(n, d) for n, d in sets if d]
    if not sets:
        print("no data yet", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.8))
    colours = {"regular": "#1f77b4", "code": "#d62728"}

    for name, rows in sets:
        c = colours.get(name, "k")
        x = [r["ctx"] for r in rows]
        ax[0].plot(x, [r["engine_tps"] for r in rows], "o-", color=c, lw=2.2,
                   label="liquid-kernel (%s)" % name)
        ax[0].plot(x, [r["oracle_tps"] for r in rows], "s--", color=c, lw=1.6,
                   alpha=0.75, label="llama.cpp (%s)" % name)
        ax[1].plot(x, [r["ratio"] for r in rows], "o-", color=c, lw=2.2, label=name)
        ax[2].plot(x, [r["engine_prefill_s"] for r in rows], "o-", color=c, lw=2.2,
                   label="liquid-kernel (%s)" % name)
        ax[2].plot(x, [r["oracle_prefill_s"] for r in rows], "s--", color=c, lw=1.6,
                   alpha=0.75, label="llama.cpp (%s)" % name)

    for k, t in enumerate(["decode throughput (higher is better)",
                           "speed ratio: liquid-kernel / llama.cpp",
                           "prompt prefill time (lower is better)"]):
        ax[k].set_xscale("log", base=2)
        ax[k].set_xlabel("context length (tokens)")
        ax[k].set_title(t)
        ax[k].grid(True, which="both", alpha=0.25)
        ax[k].legend(fontsize=8)

    ax[0].set_ylabel("tokens / second (decode)")
    ax[1].set_ylabel("x faster than llama.cpp")
    ax[1].axhline(1.0, color="k", lw=1.0, ls=":")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("seconds to ingest the prompt")

    ticks = sorted({r["ctx"] for _, rows in sets for r in rows})
    for k in range(3):
        ax[k].set_xticks(ticks)
        ax[k].set_xticklabels([("%dk" % (t // 1024)) if t >= 1024 else str(t)
                               for t in ticks], fontsize=8)

    fig.suptitle("liquid-kernel vs llama.cpp — LFM2.5-24B-A2B-QAD-Q4_0, "
                 "one RTX 3090 Ti, identical FP16 KV caches, greedy, same 32 generated tokens",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = pathlib.Path(a.out)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    print("wrote %s and %s" % (out, out.with_suffix(".svg")))


if __name__ == "__main__":
    sys.exit(main())
