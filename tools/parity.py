#!/usr/bin/env python3
"""Gate C1: compare liquid-kernel's greedy trajectory against the oracle.

The oracle reference (ref/oracle_ref.json) holds llama.cpp's pure-argmax token
IDs for every suite case.  We re-run our own engine on the same frozen prompt
token IDs and report the first divergence, plus the matched-prefix length.
Token IDs are used throughout, so the tokenizer cannot influence the result.
"""
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REF = pathlib.Path(os.environ.get("LK_REF", ROOT / "ref"))
ENGINE = ROOT / "build" / "engine"


def run_engine(ids, ngen, timeout=3600):
    env = dict(os.environ)
    env["LK_NGEN"] = str(ngen)
    r = subprocess.run([str(ENGINE), "gen"] + [str(i) for i in ids],
                       capture_output=True, text=True, env=env, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    for line in r.stdout.splitlines():
        if line.startswith("out:"):
            return [int(x) for x in line[4:].split()]
    raise RuntimeError("no out: line\n" + r.stdout[-1000:])


def main():
    ref = json.loads((REF / "oracle_ref.json").read_text())
    only = sys.argv[1] if len(sys.argv) > 1 else None
    total_match = total = 0
    rows = []
    for case in ref["cases"]:
        if only and case["name"] != only:
            continue
        want = case["greedy_tokens"]
        got = run_engine(json.loads((REF / "parity_suite.json").read_text())
                         ["cases"][[c["name"] for c in json.loads(
                             (REF / "parity_suite.json").read_text())["cases"]].index(case["name"])]
                         ["prompt_ids"], len(want))
        n = min(len(want), len(got))
        first_bad = next((i for i in range(n) if want[i] != got[i]), None)
        m = first_bad if first_bad is not None else n
        total_match += m
        total += n
        rows.append((case["name"], m, n, first_bad))
        status = "EXACT" if m == n else f"match {m}/{n}"
        print(f"  {case['name']:10s} {status:14s}" +
              ("" if first_bad is None else
               f"  first divergence @{first_bad}: oracle={want[first_bad]} ours={got[first_bad]}"))
    print(f"\nGATE C1: {total_match}/{total} tokens agree "
          f"({100.0*total_match/max(total,1):.2f}%)")
    exact = sum(1 for _, m, n, _ in rows if m == n)
    print(f"  cases matching the full oracle trajectory: {exact}/{len(rows)}")
    return 0 if total_match == total else 1


if __name__ == "__main__":
    sys.exit(main())
