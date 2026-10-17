#!/usr/bin/env python3
"""Needle in a haystack: can the model still FIND something at long context?

WHY THIS AND NOT A VIBE CHECK.  A long-context implementation can be wrong in a
way that no short-context test and no throughput benchmark reveals: if the
attention split silently stops covering the whole sequence -- a chunk that is
too small for the slice rule, a slice bound computed from the wrong count, an
off-by-one at a slice edge -- then every token still decodes at full speed and
the model still produces fluent text.  It simply cannot see the beginning of the
document any more.  The ONLY cheap way to catch that is to put a fact at a known
depth and ask for it back.

WHAT IT CONTROLS FOR:
  * the needle is inserted at a TOKEN index, so "depth 0.5" means 50% of the
    tokens, not 50% of the characters, and is identical for both engines;
  * the haystack tokens are untouched -- only the needle is spliced in -- so the
    two engines see the same context and the same retrieval problem;
  * the passcode is a random nonce that cannot be guessed or paraphrased, so a
    hit is retrieval and not luck;
  * both engines generate greedily, and the check is substring containment of
    the nonce, so it does not depend on either one's phrasing.

usage:
  tools/needle.py --corpus ref/corpus_code.bin --ctx 2048,8192,32768,131072 \
                  --depth 0.5 --out ref/needle.json
"""
import argparse
import json
import os
import pathlib
import random
import struct
import subprocess
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("LK_ORACLE_PORT", "18099"))


def post(path, payload, timeout=7200):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def tok(text):
    return post("/tokenize", {"content": text})["tokens"]


def detok(ids):
    return post("/detokenize", {"tokens": ids})["content"]


def load_ids(p):
    b = pathlib.Path(p).read_bytes()
    return list(struct.unpack("<%di" % (len(b) // 4), b))


def make_nonce(rng):
    # Letters and digits only: a tokenizer can split punctuation in ways that
    # make substring matching unreliable, and a hyphenated nonce can come back
    # reformatted.
    return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(10))


def build(hay, needle_ids, question_ids, ctx, depth):
    """Splice the needle at `depth` of the CONTEXT, not of the haystack."""
    room = ctx - len(needle_ids) - len(question_ids)
    if room < 16:
        raise ValueError("context %d too small for needle+question" % ctx)
    h = hay[:room]
    k = int(len(h) * depth)
    return h[:k] + needle_ids + h[k:] + question_ids


def engine_run(binpath, ids, ngen):
    tmp = pathlib.Path("/tmp/lk_needle.bin")
    tmp.write_bytes(struct.pack("<%di" % len(ids), *ids))
    env = dict(os.environ)
    env.update({"LK_NGEN": str(ngen), "LK_PROMPT_BIN": str(tmp),
                "CUDA_VISIBLE_DEVICES": env.get("LK_GPU", "0")})
    env.pop("LK_BENCH", None)          # bench mode suppresses the token output
    r = subprocess.run([str(binpath), "gen"], capture_output=True, text=True,
                       env=env, timeout=28800)
    if r.returncode != 0:
        raise RuntimeError("engine failed:\n" + r.stderr[-2000:])
    out = None
    for line in r.stdout.splitlines():
        if line.startswith("out:"):
            out = [int(x) for x in line[4:].split()]
    if out is None:
        raise RuntimeError("no out: line\n" + r.stdout[-1500:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ctx", default="2048,8192,32768,131072")
    ap.add_argument("--depth", default="0.5")
    ap.add_argument("--ngen", type=int, default=24)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--engine", default=str(ROOT / "build" / "engine"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    hay = load_ids(a.corpus)
    rng = random.Random(a.seed)
    ctxs = [int(x) for x in a.ctx.split(",")]
    depths = [float(x) for x in a.depth.split(",")]

    print("%9s %7s | %-28s | %-28s" % ("ctx", "depth", "liquid-kernel", "llama.cpp"))
    print("-" * 82)
    rows = []
    for c in ctxs:
        for d in depths:
            nonce = make_nonce(rng)
            needle_ids = tok("\n\nIMPORTANT: the secret passcode for this document "
                             "is %s. Remember it.\n\n" % nonce)
            question_ids = tok("\n\nQuestion: what is the secret passcode for this "
                               "document? Answer with the passcode only.\n"
                               "Answer: the secret passcode is")
            ids = build(hay, needle_ids, question_ids, c, d)
            prompt = detok(ids)

            eo = engine_run(a.engine, ids, a.ngen)
            etxt = detok(eo)
            od = post("/completion", {"prompt": prompt, "n_predict": a.ngen,
                                      "temperature": 0.0, "cache_prompt": False,
                                      "seed": 1234})
            otxt = od["content"]

            ehit = nonce in etxt
            ohit = nonce in otxt.replace(" ", "")
            rows.append(dict(ctx=c, depth=d, nonce=nonce, prompt_tokens=len(ids),
                             engine_hit=ehit, oracle_hit=ohit,
                             engine_text=etxt[:200], oracle_text=otxt[:200],
                             oracle_prompt_n=od["timings"]["prompt_n"]))
            print("%9d %7.2f | %-28s | %-28s" %
                  (c, d, ("HIT " if ehit else "MISS") + repr(etxt[:20]),
                   ("HIT " if ohit else "MISS") + repr(otxt[:20])))
            sys.stdout.flush()
            pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))

    eh = sum(r["engine_hit"] for r in rows)
    oh = sum(r["oracle_hit"] for r in rows)
    print("\nTOTAL: liquid-kernel %d/%d   llama.cpp %d/%d" % (eh, len(rows), oh, len(rows)))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
