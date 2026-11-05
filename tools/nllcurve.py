#!/usr/bin/env python3
"""Per-position NLL for both engines, from ONE teacher-forced pass each.

WHY POSITION AND NOT JUST THE MEAN.  Gate C2 reports a single mean NLL and a
paired interval over the whole corpus.  That is the right instrument for "is the
arithmetic the same" and the WRONG one for "does long context work", because a
fault in the attention split does not shift the mean much -- most tokens sit at
short range.  It raises NLL for the tokens that sit far from the start, which is
exactly the population a 131k context exists to serve.  Averaged over a whole
corpus that signal is diluted away.

One pass over a long corpus therefore produces the entire context curve: bucket
the per-position NLL by position and the x-axis is "how much context was
available".  Cost is one pass, not one pass per context length.

The exact-float64 reference of C2 is deliberately NOT recomputed here -- it needs
LK_DUMP_LAYERS, which at 131072 positions is 131072 x 41 x 2048 x 4 B = 44 GB.
This compares the two ENGINES to each other; C2 remains the gate that says which
of them is closer to the mathematics.

usage:
  tools/nllcurve.py --corpus ref/corpus_regular.bin --out ref/nll_regular.json
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ORACLE = ROOT / "build" / "oracle_ppl"
GGUF = os.environ.get("LK_MODEL", str(ROOT / "models" / "LFM2.5-24B-A2B-QAD-Q4_0.gguf"))
BACKEND = os.environ.get("LK_BACKEND_DIR",
                         str(pathlib.Path.home() / "llama.cpp" / "build" / "bin"))

# Log-spaced, because decode cost and context effect are both logarithmic.
BUCKETS = [(1, 256), (256, 1024), (1024, 4096), (4096, 16384),
           (16384, 65536), (65536, 131072)]


def run(cmd, env):
    # errors="replace", NOT the default.  llama.cpp writes progress and timing
    # lines to stderr that are not valid UTF-8, and text=True decodes BOTH
    # streams unconditionally -- so a successful run died with
    # UnicodeDecodeError before its output was ever looked at.  This tool had
    # never been executed, which is the only reason that survived.
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                       env=env, timeout=28800)
    if r.returncode != 0:
        raise RuntimeError("%s failed:\n%s\n%s" % (cmd[0], r.stdout[-1500:], r.stderr[-2500:]))
    return r.stdout


def load(path):
    out = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 3:
            out[int(p[0])] = (float(p[1]), int(p[2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default=str(ROOT / "build" / "engine"))
    a = ap.parse_args()

    tokf = pathlib.Path(a.corpus).resolve()
    if not tokf.exists():
        tokf = ROOT / a.corpus
    odump = pathlib.Path("/tmp/nll.oracle.txt")
    edump = pathlib.Path("/tmp/nll.engine.txt")
    # NO CACHING.  C2 caches its oracle dump and that is a trap: re-running it on
    # a different corpus silently reports the OLD corpus's numbers.
    for p in (odump, edump):
        p.unlink(missing_ok=True)

    env = dict(os.environ)
    env.update({"LK_BACKEND_DIR": BACKEND,
                "CUDA_VISIBLE_DEVICES": os.environ.get("LK_ORACLE_GPU", "1")})
    print(run([str(ORACLE), GGUF, str(tokf), str(odump)], env).strip()[-400:])

    env2 = dict(os.environ)
    env2.update({"CUDA_VISIBLE_DEVICES": os.environ.get("LK_GPU", "0"),
                 "LK_PPL_DUMP": str(edump), "LK_NO_GRAPH": "1"})
    print(run([a.engine, "ppl", str(tokf)], env2).strip()[-400:])

    eng, orc = load(edump), load(odump)
    common = sorted(set(eng) & set(orc))
    print("\n%d scored positions in common" % len(common))
    print("%-18s %8s %10s %10s %9s %10s" %
          ("position", "n", "eng nll", "llama nll", "delta", "top1 agree"))
    print("-" * 70)
    rows = []
    for lo, hi in BUCKETS:
        sel = [i for i in common if lo <= i < hi]
        if not sel:
            continue
        en = sum(eng[i][0] for i in sel) / len(sel)
        on = sum(orc[i][0] for i in sel) / len(sel)
        ag = sum(eng[i][1] == orc[i][1] for i in sel) / len(sel)
        rows.append(dict(lo=lo, hi=hi, n=len(sel), engine_nll=en, oracle_nll=on,
                         delta=en - on, top1=ag))
        print("%-18s %8d %10.6f %10.6f %+9.6f %9.2f%%" %
              ("%d-%d" % (lo, hi), len(sel), en, on, en - on, 100 * ag))
    pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
