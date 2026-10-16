#!/usr/bin/env python3
"""Context sweep: engine decode throughput as a function of prompt length.

The engine takes a literal list of token IDs on the command line and prefills
all of them before timing, so a prompt of N tokens is a genuine decode at
context N -- the KV cache really is full.  We tile a natural-text corpus rather
than repeating the same few tokens, because repeated text flattens the router
distribution and the MoE is 60% of the step.

Usage:  tools/sweep.py [--ctx 144,1064,2064,4064,8064] [--ngen 64] [--reps 1]
"""
import argparse
import os
import pathlib
import re
import subprocess
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE = pathlib.Path(os.environ.get("LK_ENGINE", ROOT / "build" / "engine"))
RESULT = re.compile(r"RESULT ctx=(\d+) generated=(\d+) seconds=([\d.]+) "
                    r"tok_s=([\d.]+) GB_s=([\d.]+) pct_peak=([\d.]+)")


def load_corpus():
    blob = (ROOT / "ref" / "ppl_natural.bin").read_bytes()
    step = 4 if len(blob) % 4 == 0 else 2
    n = len(blob) // step
    fmt = "<%di" % n if step == 4 else "<%dh" % n
    import struct
    return list(struct.unpack(fmt, blob))


def make_prompt(corpus, ctx):
    if ctx <= len(corpus):
        return corpus[:ctx]
    reps = ctx // len(corpus) + 1
    return (corpus * reps)[:ctx]


def run_once(ids, ngen, extra_env=None):
    env = dict(os.environ)
    env["LK_BENCH"] = "1"
    env["LK_NGEN"] = str(ngen)
    env["CUDA_VISIBLE_DEVICES"] = env.get("LK_GPU", "0")
    if extra_env:
        env.update(extra_env)
    r = subprocess.run([str(ENGINE), "gen"] + [str(i) for i in ids],
                       capture_output=True, text=True, env=env, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    for line in r.stdout.splitlines():
        m = RESULT.match(line)
        if m:
            return dict(ctx=int(m.group(1)), n=int(m.group(2)),
                        sec=float(m.group(3)), tps=float(m.group(4)),
                        gbs=float(m.group(5)), pct=float(m.group(6)))
    raise RuntimeError("no RESULT line\n" + r.stdout[-2000:] + r.stderr[-2000:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="144,1064,2064,4064,8064")
    ap.add_argument("--ngen", type=int, default=64)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    corpus = load_corpus()
    label = a.label or os.environ.get("LK_SWEEP_LABEL", "")
    print("# corpus: %d natural tokens, ngen=%d, reps=%d %s"
          % (len(corpus), a.ngen, a.reps, label))
    print("%8s %10s %10s %10s %8s %9s" % ("ctx", "tok/s", "sd", "GB/s", "pct_peak", "kv_MB"))
    for ctx in [int(x) for x in a.ctx.split(",")]:
        ids = make_prompt(corpus, ctx)
        rs = [run_once(ids, a.ngen) for _ in range(a.reps)]
        tps = [r["tps"] for r in rs]
        sd = statistics.stdev(tps) if len(tps) > 1 else 0.0
        kv_mb = ctx * 20480 / 1e6
        print("%8d %10.2f %10.2f %10.1f %8.1f %9.1f"
              % (rs[0]["ctx"], statistics.mean(tps), sd,
                 statistics.mean([r["gbs"] for r in rs]),
                 statistics.mean([r["pct"] for r in rs]), kv_mb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
