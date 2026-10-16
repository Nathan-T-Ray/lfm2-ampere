#!/usr/bin/env python3
"""Re-measure the llama.cpp oracle's decode rate at chosen prompt lengths.

llama-server logs its timings and the HTTP face of this model is chat-templated,
so /completion answers 200 but the response body is not the thing to parse --
the log line is.  We feed filler text of roughly the right length and read back
the ACTUAL prompt token count from `prompt eval time = ... / N tokens`, so the
reported context is measured rather than assumed.

Decode speed for this model is bandwidth-bound and near-uniform in routing, so
filler content is fine; the point is to match the CONTEXT LENGTH the engine was
benchmarked at.

usage: tools/oracle_bench.py [--ngen 64] [--tok 100,1000,2000,4000,8000]
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.request

PORT = os.environ.get("LK_ORACLE_PORT", "18099")
LOG = pathlib.Path(os.environ.get(
    "LK_SCRATCH", pathlib.Path.home() / "local-agent-harnesses/model-scratch/lfm25")) / "oracle_verbose.log"
FILLER = ("The history of the Roman Empire spans more than a thousand years and "
          "its influence can still be seen in the languages, laws and borders of "
          "the modern world. ")

PROMPT_EVAL = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
DECODE = re.compile(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) tokens \(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)")


def log_size():
    return LOG.stat().st_size if LOG.exists() else 0


def tail_from(offset):
    with open(LOG, "rb") as f:
        f.seek(offset)
        return f.read().decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngen", type=int, default=64)
    ap.add_argument("--tok", default="100,1000,2000,4000,8000")
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()

    print("%8s %10s %10s %8s" % ("ctx", "tok/s", "prompttok", "reps"))
    for target in [int(x) for x in a.tok.split(",")]:
        rates = []
        real_tok = None
        for _ in range(a.reps):
            text = FILLER * max(1, int(target * 4.0 / len(FILLER)) + 1)
            off = log_size()
            body = json.dumps({"prompt": text, "n_predict": a.ngen,
                               "temperature": 0.0, "cache_prompt": False,
                               "top_k": 1}).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:%s/completion" % PORT, data=body,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    r.read()
            except Exception as e:                       # the body is templated garbage
                print("  (request error, using log: %s)" % e, file=sys.stderr)
            new = tail_from(off)
            pm = PROMPT_EVAL.findall(new)
            dm = DECODE.findall(new)
            if not pm or not dm:
                print("  no timing lines for target %d; got %d/%d" % (target, len(pm), len(dm)))
                continue
            real_tok = int(pm[-1][1])
            rates.append(float(dm[-1][2]))
        if rates:
            print("%8d %10.1f %10s %8d"
                  % (real_tok + a.ngen, sum(rates) / len(rates), real_tok, len(rates)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
