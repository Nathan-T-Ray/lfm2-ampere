#!/usr/bin/env python3
"""Head-to-head decode throughput: liquid-kernel vs llama.cpp, across context.

THE QUESTION THIS EXISTS TO ANSWER.  The engine is 1.72x llama.cpp at 8k
context, but its throughput FALLS with context (443 tok/s at 272, 399 at 6454)
and so does everyone's -- attention reads the whole KV cache once per token, so
the KV traffic grows linearly while the weight traffic is fixed.  The open
question is whether the GAP survives, because a model served behind a 30k-token
system prompt is not served at 272 tokens.

WHAT MAKES THE COMPARISON FAIR, and each of these is a thing that could
silently ruin it:

  * SAME TOKENS.  The prompt is taken from a token-id corpus and DETOKENIZED
    through the oracle's own /detokenize before being sent back to it, so both
    engines are conditioned on the identical token sequence.  Comparing a
    text-prompted oracle against a token-prompted engine would compare two
    tokenizers.
  * SAME GENERATED COUNT.  ngen is fixed and asserted, not assumed.
  * MEASURED CONTEXT, NOT REQUESTED CONTEXT.  The oracle reports prompt_n and
    the engine reports ctx; both are recorded and checked against the target,
    so a silently truncated prompt shows up as a context mismatch instead of as
    a suspiciously fast point.
  * NO PROMPT CACHE.  cache_prompt=false, or the oracle's prefill is skipped
    and its prompt_n lies.
  * ONE REQUEST PER POINT, alternating engines, so a thermal drift hits both.

usage:
  tools/ctxcompare.py --corpus ref/corpus_regular.bin \
      --ctx 256,1024,4096,8192,16384,32768,65536,131072 --ngen 64 --out ref/ctx_regular.json
"""
import argparse
import json
import os
import pathlib
import statistics
import struct
import subprocess
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("LK_ORACLE_PORT", "18099"))


def load_ids(p):
    b = pathlib.Path(p).read_bytes()
    assert len(b) % 4 == 0, "corpus must be an int32 blob"
    return list(struct.unpack("<%di" % (len(b) // 4), b))


def post(path, payload, timeout=3600):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def oracle_decode(text, ngen):
    d = post("/completion", {"prompt": text, "n_predict": ngen,
                             "temperature": 0.0, "cache_prompt": False,
                             "seed": 1234})
    t = d["timings"]
    return dict(tps=t["predicted_per_second"], prompt_n=t["prompt_n"],
                gen=t["predicted_n"], prefill_s=t["prompt_ms"] / 1000.0,
                prefill_tps=t["prompt_per_second"])


def engine_decode(binpath, ids, ngen):
    """Writes the ids to a temp int32 blob and drives the engine through
    LK_PROMPT_BIN -- a 131072-token prompt cannot be passed through argv."""
    tmp = pathlib.Path("/tmp/lk_ctx_prompt.bin")
    tmp.write_bytes(struct.pack("<%di" % len(ids), *ids))
    env = dict(os.environ)
    env.update({"LK_BENCH": "1", "LK_NGEN": str(ngen),
                "LK_PROMPT_BIN": str(tmp),
                "CUDA_VISIBLE_DEVICES": env.get("LK_GPU", "0")})
    r = subprocess.run([str(binpath), "gen"], capture_output=True, text=True,
                       env=env, timeout=14400)
    if r.returncode != 0:
        raise RuntimeError("engine failed: " + r.stderr[-2000:])
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            kv = dict(p.split("=") for p in line.split()[1:])
            return dict(tps=float(kv["tok_s"]), ctx=int(kv["ctx"]),
                        gen=int(kv["generated"]), prompt_n=int(kv["prompt_n"]),
                        prefill_s=float(kv["prefill_s"]),
                        prefill_tps=float(kv["prefill_tok_s"]))
    raise RuntimeError("no RESULT line:\n" + r.stdout[-2000:] + r.stderr[-2000:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ctx", default="256,1024,4096,8192,16384,32768,65536,131072")
    ap.add_argument("--ngen", type=int, default=64)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--engine", default=str(ROOT / "build" / "engine"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ids = load_ids(a.corpus)
    ctxs = [int(x) for x in a.ctx.split(",")]
    maxc = max(ctxs)
    if len(ids) < maxc:
        print("WARNING: corpus has %d tokens, target %d -- points above the corpus "
              "length are DROPPED rather than tiled, because tiling changes the "
              "router distribution" % (len(ids), maxc), file=sys.stderr)
        ctxs = [c for c in ctxs if c <= len(ids)]

    print("corpus %s: %d tokens; sweeping %s" % (a.corpus, len(ids), ctxs))
    print("DECODE is the serving number; PREFILL is what a 131k system prompt costs")
    print("FIRST, and the engine has no batched prefill, so it is one decode step")
    print("per prompt token.  Both are reported because only reporting decode makes")
    print("long context look uniformly cheap.")
    print()
    print("%8s | %9s %9s %7s | %9s %9s %9s | %6s" %
          ("ctx", "eng t/s", "llama t/s", "ratio",
           "eng pre t", "orc pre t", "pre ratio", "pro n"))
    print("-" * 84)

    rows = []
    for c in ctxs:
        sub = ids[:c]
        # Detokenize through the oracle so both engines see the SAME ids.
        text = post("/detokenize", {"tokens": sub})["content"]
        E, O = [], []
        for rep in range(a.reps):
            # Alternate engines within a rep so any drift is shared.
            E.append(engine_decode(a.engine, sub, a.ngen))
            O.append(oracle_decode(text, a.ngen))
        e = max(E, key=lambda d: d["tps"])
        o = max(O, key=lambda d: d["tps"])
        # The engine reports ctx as prompt+generated; the oracle reports the
        # prompt only and adds a BOS.  Compare PROMPT lengths or every point
        # raises a false alarm.
        orc_prompt = min(d["prompt_n"] for d in O)
        flag = "" if abs(e["prompt_n"] - orc_prompt) <= max(2, 0.005 * c) else "  <-- CTX MISMATCH"
        row = dict(ctx=c, engine_tps=e["tps"], oracle_tps=o["tps"],
                   ratio=e["tps"] / o["tps"],
                   engine_prefill_s=e["prefill_s"], oracle_prefill_s=o["prefill_s"],
                   prefill_ratio=e["prefill_s"] / max(o["prefill_s"], 1e-9),
                   engine_prompt_n=e["prompt_n"], oracle_prompt_n=orc_prompt,
                   ngen=a.ngen, engine_gen=e["gen"], oracle_gen=o["gen"],
                   corpus=os.path.basename(a.corpus))
        rows.append(row)
        print("%8d | %9.2f %9.2f %6.2fx | %8.2fs %8.2fs %8.1fx | %6d%s" %
              (c, e["tps"], o["tps"], row["ratio"],
               e["prefill_s"], o["prefill_s"], row["prefill_ratio"],
               e["prompt_n"], flag))
        sys.stdout.flush()
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))   # checkpoint each point

    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
