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
import struct
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
    # LK_PROMPT_BIN, not argv.  A 131072-token prompt is ~0.79 MB of digits plus
    # ~1.05 MB of argv pointers against a 2 MB ARG_MAX that also has to hold the
    # environment, so execve fails with E2BIG before main() runs and it presents
    # as "the engine crashed at long context".  Passing the blob by file removes
    # the limit and costs one write per point.
    tmp = pathlib.Path("/tmp/lk_sweep_prompt.bin")
    tmp.write_bytes(struct.pack("<%di" % len(ids), *ids))
    env["LK_PROMPT_BIN"] = str(tmp)
    if extra_env:
        env.update(extra_env)
    r = subprocess.run([str(ENGINE), "gen"],
                       capture_output=True, text=True, env=env, timeout=28800)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            # Every field is key=value, so parse them all rather than depending on
            # positional groups: adding prefill_s to the engine's line must not be
            # able to silently change what this returns.
            kv = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
            out = {k: float(v) for k, v in kv.items()}
            out["ctx"] = int(kv["ctx"])
            out["n"] = int(kv["generated"])
            out["sec"] = float(kv["seconds"])
            out["tps"] = float(kv["tok_s"])
            out["gbs"] = float(kv["GB_s"])
            out["pct"] = float(kv["pct_peak"])
            return out
    raise RuntimeError("no RESULT line\n" + r.stdout[-2000:] + r.stderr[-2000:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="144,1064,2064,4064,8064")
    ap.add_argument("--ngen", type=int, default=64)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--label", default="")
    # --corpus exists because the default is 558 tokens and make_prompt TILES it.
    # At 8192 that is already 15 repeats and at 32768 it is 59, which flattens
    # the router distribution -- the MoE is ~60% of the step and its traffic
    # depends on routing, so a tiled prompt measures a different model.  Long
    # context needs a corpus that is actually long.
    ap.add_argument("--corpus", default="",
                    help="int32 token blob; default ref/ppl_natural.bin")
    a = ap.parse_args()

    if a.corpus:
        blob = (ROOT / a.corpus).read_bytes() if not os.path.isabs(a.corpus) \
            else pathlib.Path(a.corpus).read_bytes()
        corpus = list(struct.unpack("<%di" % (len(blob) // 4), blob))
    else:
        corpus = load_corpus()
    label = a.label or os.environ.get("LK_SWEEP_LABEL", "")
    print("# corpus: %d natural tokens, ngen=%d, reps=%d %s"
          % (len(corpus), a.ngen, a.reps, label))
    print("%8s %10s %10s %10s %8s %9s %10s %11s"
          % ("ctx", "tok/s", "sd", "GB/s", "pct_peak", "kv_MB", "pre_s", "pre_tok/s"))
    for ctx in [int(x) for x in a.ctx.split(",")]:
        if ctx > len(corpus):
            print("%8d  skipped: corpus has only %d tokens" % (ctx, len(corpus)))
            continue
        # No tiling: truncate.  Repeating a corpus changes the model, and above
        # ~32k there is nothing to repeat from anyway.
        ids = corpus[:ctx]
        rs = [run_once(ids, a.ngen) for _ in range(a.reps)]
        tps = [r["tps"] for r in rs]
        sd = statistics.stdev(tps) if len(tps) > 1 else 0.0
        kv_mb = ctx * 20480 / 1e6
        print("%8d %10.2f %10.2f %10.1f %8.1f %9.1f %10.2f %11.1f"
              % (rs[0]["ctx"], statistics.mean(tps), sd,
                 statistics.mean([r["gbs"] for r in rs]),
                 statistics.mean([r["pct"] for r in rs]), kv_mb,
                 statistics.mean([r.get("prefill_s", 0.0) for r in rs]),
                 statistics.mean([r.get("prefill_tok_s", 0.0) for r in rs])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
