#!/usr/bin/env python3
"""Bit-exactness A/B: run two engine binaries on the same prompts and compare
the greedy token trajectory.  If a kernel change is bit-identical, the
trajectories must agree exactly; if they agree exactly the change provably did
not perturb a single logit enough to flip an argmax.

Usage: tools/ab.py BASE_BINARY NEW_BINARY [--ngen 48] [--cases factual,prose] [--prefill seq|batch]

PREFILL.  The batched prefill (src/prefill.cu) is not bit-identical to the
sequential one -- it sums GEMM blocks in a different order and runs attention on
fp16 tensor cores -- so a gate that let it run would report every decode change
as DIVERGED.  This gate therefore pins LK_PREFILL=seq on BOTH binaries by
default: it is, as it always was, a proof about the decode path.  A binary older
than the batched path ignores the variable, which is exactly the old behaviour.
--prefill batch pins the batched path on both instead, which is the gate for a
change INSIDE the batched path that claims to be bit-identical (a tile shape, a
launch config, a fused epilogue that preserves the arithmetic).
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REF = pathlib.Path(os.environ.get("LK_REF", ROOT / "ref"))


def run(binary, ids, ngen, extra_env=None, prefill="seq"):
    env = dict(os.environ)
    env["LK_NGEN"] = str(ngen)
    env["LK_PREFILL"] = prefill
    env["CUDA_VISIBLE_DEVICES"] = env.get("LK_GPU", "0")
    # TURN THE END-OF-TURN STOP OFF FOR THIS GATE.  bench mode is what normally
    # suppresses it, but bench mode also suppresses the `out:` line that this
    # gate compares, so it cannot be used here.  Without this the gate would
    # still pass -- both sides stop on the same token -- but it would silently
    # start comparing twenty tokens instead of forty-eight on any prompt whose
    # trajectory happens to end early, and a gate that quietly shortens itself
    # is worse than one that fails.
    env["LK_EOS"] = ""
    if extra_env:
        env.update(extra_env)
    r = subprocess.run([str(binary), "gen"] + [str(i) for i in ids],
                       capture_output=True, text=True, env=env, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError("%s failed:\n%s" % (binary, r.stderr[-3000:]))
    for line in r.stdout.splitlines():
        if line.startswith("out:"):
            return [int(x) for x in line[4:].split()]
    raise RuntimeError("no out: line from %s\n%s" % (binary, r.stdout[-2000:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--ngen", type=int, default=48)
    ap.add_argument("--cases", default="")
    ap.add_argument("--prefill", default="seq", choices=["seq", "batch"])
    a = ap.parse_args()

    ref = json.loads((REF / "oracle_ref.json").read_text())
    suite = {c["name"]: c for c in
             json.loads((REF / "parity_suite.json").read_text())["cases"]}
    cases = ref["cases"]
    if a.cases:
        want = set(a.cases.split(","))
        cases = [c for c in cases if c["name"] in want]

    allsame = True
    for c in cases:
        ids = suite[c["name"]]["prompt_ids"]
        b = run(pathlib.Path(a.base), ids, a.ngen, prefill=a.prefill)
        n = run(pathlib.Path(a.new), ids, a.ngen, prefill=a.prefill)
        same = (b == n)
        allsame &= same
        first = next((i for i, (x, y) in enumerate(zip(b, n)) if x != y), None)
        print("%-12s %s  base=%d new=%d tokens%s"
              % (c["name"], "IDENTICAL" if same else "DIVERGED ",
                 len(b), len(n), "" if same else "  first diff at %d" % first))
        if not same:
            print("   base: %s" % " ".join(map(str, b[:24])))
            print("   new : %s" % " ".join(map(str, n[:24])))
    print("\n%s" % ("ALL CASES BIT-IDENTICAL" if allsame else "SOME CASES DIVERGED"))
    return 0 if allsame else 1


if __name__ == "__main__":
    raise SystemExit(main())
