#!/usr/bin/env python3
"""Engine-vs-engine context A/B, without the oracle.

tools/sweep.py always times llama.cpp alongside the engine, which is right when
the question is "are we faster than llama.cpp" and wrong when the question is
"did THIS change to THIS kernel make the engine faster".  At 131040 the oracle
adds 34 s of prefill per point and cannot answer the second question at all,
because its own run-to-run spread is larger than the effect being measured.

Two engine binaries, interleaved round-robin, so that a clock or thermal drift
over a 20-minute run hits both sides equally instead of whichever side ran
second.  That drift is not hypothetical: an earlier sweep in this project
reported 23.9 us on a cadence-1 point against 18.5 everywhere else, and the
number survived until it was re-measured round-robin.

  LK_AB_A=/tmp/engine_base LK_AB_B=/tmp/engine_v4 \
      python3 tools/abctx.py --corpus ref/corpus_regular.bin --ctx 32768 --ngen 32 --reps 2
"""
import argparse
import os
import pathlib
import struct
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
A = pathlib.Path(os.environ.get("LK_AB_A", ROOT / "build" / "engine"))
B = pathlib.Path(os.environ.get("LK_AB_B", ROOT / "build" / "engine"))
LABEL_A = os.environ.get("LK_AB_LA", "A")
LABEL_B = os.environ.get("LK_AB_LB", "B")


def run(engine, ids, ngen):
    env = dict(os.environ)
    env["LK_BENCH"] = "1"
    env["LK_NGEN"] = str(ngen)
    env["CUDA_VISIBLE_DEVICES"] = env.get("LK_GPU", "0")
    tmp = pathlib.Path("/tmp/lk_ab_prompt.bin")
    tmp.write_bytes(struct.pack("<%di" % len(ids), *ids))
    env["LK_PROMPT_BIN"] = str(tmp)
    r = subprocess.run([str(engine), "gen"], capture_output=True, text=True,
                       env=env, timeout=28800)
    if r.returncode != 0:
        return None, r.stderr[-1500:]
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return {k: v for k, v in
                    (p.split("=", 1) for p in line.split()[1:] if "=" in p)}, None
    return None, "no RESULT line:\n" + r.stdout[-1500:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="ref/corpus_regular.bin")
    ap.add_argument("--ctx", default="32768")
    ap.add_argument("--ngen", type=int, default=32)
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()

    p = pathlib.Path(a.corpus)
    if not p.is_absolute():
        p = ROOT / p
    blob = p.read_bytes()
    corpus = list(struct.unpack("<%di" % (len(blob) // 4), blob))

    ctxs = [int(c) for c in a.ctx.split(",")]
    print("corpus %s: %d tokens" % (a.corpus, len(corpus)))
    print("A = %s (%s)" % (A, LABEL_A))
    print("B = %s (%s)" % (B, LABEL_B))
    print()
    print("     ctx | %-18s | %-18s |  B/A" % (LABEL_A, LABEL_B))
    print("-" * 66)

    for ctx in ctxs:
        if ctx > len(corpus):
            print("ctx %d > corpus %d, skipping" % (ctx, len(corpus)))
            continue
        ids = corpus[:ctx]
        got = {LABEL_A: [], LABEL_B: []}
        pre = {LABEL_A: [], LABEL_B: []}
        for rep in range(a.reps):
            for lab, eng in ((LABEL_A, A), (LABEL_B, B)):
                kv, err = run(eng, ids, a.ngen)
                if kv is None:
                    print("ctx %d rep %d %s FAILED: %s" % (ctx, rep, lab, err))
                    sys.exit(1)
                got[lab].append(float(kv["tok_s"]))
                pre[lab].append(float(kv["prefill_s"]))
        ta, tb = got[LABEL_A], got[LABEL_B]
        pa, pb = pre[LABEL_A], pre[LABEL_B]
        ma, mb = sum(ta) / len(ta), sum(tb) / len(tb)
        print("%8d | %7.2f tok/s      | %7.2f tok/s      | %.3fx" % (ctx, ma, mb, mb / ma))
        # Prefill is attention-dominated at long context, so it is the more
        # sensitive detector of a change to the attention kernel even though
        # the serving number is decode.
        print("         | prefill %6.1fs     | prefill %6.1fs     | %.3fx" % (
            sum(pa) / len(pa), sum(pb) / len(pb),
            (sum(pb) / len(pb)) / (sum(pa) / len(pa))))
        if len(ta) > 1:
            print("         | runs %s | runs %s" % (
                " ".join("%.2f" % v for v in ta), " ".join("%.2f" % v for v in tb)))


if __name__ == "__main__":
    main()
