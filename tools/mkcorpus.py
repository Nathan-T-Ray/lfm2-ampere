#!/usr/bin/env python3
"""Build int32 token-id corpora for the long-context sweep.

WHY A FILE AND NOT A LIST.  tools/sweep.py passes token IDs on the command line.
That is fine to 8k and fatal at 131k: 131072 tokens is ~0.79 MB of digits plus
~1.05 MB of argv pointers against a 2 MB ARG_MAX that ALSO has to hold the
environment, so execve fails with E2BIG before main() runs.  The engine reads
these blobs through LK_PROMPT_BIN instead, so there is no argument list at all.

WHY TOKENISE THROUGH llama-server.  Both engines must be fed the IDENTICAL
token sequence or the comparison is measuring the tokenizer rather than the
kernels.  /tokenize on the running oracle is by construction the same tokenizer
llama.cpp will use, so a disagreement is impossible rather than merely unlikely.

Chunks are cut on blank lines, so as few BPE merges as possible are split
across a boundary.  A split merge changes a handful of tokens out of 131072 --
irrelevant to throughput, but this corpus is also used for the NLL curve, so it
is worth not being sloppy about.

usage:
  tools/mkcorpus.py --in FILE --out ref/corpus_regular.bin --target 140000
"""
import argparse
import json
import pathlib
import struct
import sys
import urllib.request

PORT = 18099


def tokenize(text, port):
    req = urllib.request.Request(
        "http://127.0.0.1:%d/tokenize" % port,
        data=json.dumps({"content": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["tokens"]


def chunks(text, approx_chars):
    """Split on blank lines into pieces of roughly approx_chars."""
    paras = text.split("\n\n")
    buf, n = [], 0
    for p in paras:
        buf.append(p)
        n += len(p) + 2
        if n >= approx_chars:
            yield "\n\n".join(buf)
            buf, n = [], 0
    if buf:
        yield "\n\n".join(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--target", type=int, default=140000,
                    help="stop once at least this many tokens have been produced")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--chunk-chars", type=int, default=40000)
    a = ap.parse_args()

    raw = pathlib.Path(a.src).read_text(encoding="utf-8", errors="replace")
    ids = []
    for i, ch in enumerate(chunks(raw, a.chunk_chars)):
        ids.extend(tokenize(ch, a.port))
        if len(ids) >= a.target:
            break
        if (i + 1) % 10 == 0:
            print("  %d tokens ..." % len(ids), file=sys.stderr, flush=True)

    out = pathlib.Path(a.dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    # int32 little-endian.  The engine's loader and tools/sweep.py both derive
    # the element width from len(blob) % 4, so an int32 blob is the only width
    # that round-trips unambiguously.
    out.write_bytes(struct.pack("<%di" % len(ids), *ids))
    print("%s: %d tokens, %d bytes (from %s)" %
          (out, len(ids), out.stat().st_size, a.src))


if __name__ == "__main__":
    main()
