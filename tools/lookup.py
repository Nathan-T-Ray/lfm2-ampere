#!/usr/bin/env python3
"""Prompt-lookup (n-gram) drafting: would it actually pay, and on what text?

The single-GPU batch-1 ceiling is measured at 488.9 tok/s (Phase 4z.2), so a
fifth card is not available and a second one is out of scope.  The remaining
single-stream lever is speculative decoding, and this project already rejected
the usual form of it: a draft MODEL costs more than it saves.  Prompt-lookup
drafting is not that.  It searches text that is ALREADY in the context for a
match of the last few tokens and copies the continuation as the draft.  It costs
a host-side search, needs no second model, and can be TRIGGERED -- when there is
no match you run the plain M=1 graph and lose nothing.

Greedy speculative decoding with exact verification is LOSSLESS: the emitted
tokens are identical to what the plain engine would have produced.  So its
acceptance rate can be simulated exactly, offline, from trajectories the engine
has already generated.  That is what this does -- no GPU time, no new kernels,
and it decides whether the M=4 build is worth starting.

Measured cost of an M=4 verification pass (Phase 4z.3), relative to one plain
token at 2.266 ms:

    bytes  4 x 887.6 MB = 3550.4 MB  ->  4.30 ms at the measured 825 GB/s
    + launch toll 0.268 ms (paid once for all four)
    + small kernels 0.221 ms (latency-bound) .. 0.884 ms (work-bound)
    ratio  2.11x .. 2.41x

Expected tokens emitted from one M=4 pass, if each of the 3 drafts survives
independently with probability p, is 1 + 3p.  Break-even is therefore

    (1 + 3p) / ratio = 1   ->   p = 0.37 .. 0.47

Usage:  tools/lookup.py [--ngen 384] [--minmatch 2] [--drafts 3]
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REF = pathlib.Path(os.environ.get("LK_REF", ROOT / "ref"))


def greedy(binary, ids, ngen):
    env = dict(os.environ)
    env["LK_NGEN"] = str(ngen)
    env["CUDA_VISIBLE_DEVICES"] = env.get("LK_GPU", "0")
    r = subprocess.run([str(binary), "gen"] + [str(i) for i in ids],
                       capture_output=True, text=True, env=env, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    for line in r.stdout.splitlines():
        if line.startswith("out:"):
            return [int(x) for x in line[4:].split()]
    raise RuntimeError("no out: line")


def find_match(ctx, suffix):
    """Longest match of `suffix` ending at the last generated position.

    Returns the number of tokens that follow the match in the context, i.e.
    how many draft tokens are available, plus the draft itself.  Only text that
    would exist at generation time is searched, so this is what a real
    implementation can do.
    """
    n = len(ctx)
    L = len(suffix)
    for start in range(n - L - 1, -1, -1):          # most recent match wins
        if ctx[start:start + L] == suffix:
            nxt = ctx[start + L:]
            if nxt:
                return nxt
    return []


def simulate(traj, prompt_len, minmatch, drafts):
    """Walk the trajectory, drafting from context before each position."""
    ctx = list(traj)
    per_pass = []          # tokens correctly emitted per verification pass
    triggered = 0
    total = 0
    for t in range(prompt_len, len(traj) - 1):
        total += 1
        suffix = traj[max(0, t - minmatch + 1):t + 1]
        if len(suffix) < minmatch:
            per_pass.append(1); continue
        cand = find_match(traj[:t + 1], suffix)
        if not cand:
            per_pass.append(1); continue
        triggered += 1
        draft = cand[:drafts]
        # how many leading drafts match what the model actually did
        acc = 0
        for i, d in enumerate(draft):
            if t + 1 + i < len(traj) and traj[t + 1 + i] == d:
                acc += 1
            else:
                break
        per_pass.append(1 + acc)
    return per_pass, triggered, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/engine")
    ap.add_argument("--ngen", type=int, default=384)
    ap.add_argument("--drafts", type=int, default=3)
    ap.add_argument("--matches", default="2,3,4")
    a = ap.parse_args()

    ref = json.loads((REF / "oracle_ref.json").read_text())
    cases = ref["cases"]
    suite = {c["name"]: c for c in
             json.loads((REF / "parity_suite.json").read_text())["cases"]}

    # A verification pass costs this many plain tokens (Phase 4z.3):
    #   4 x 887.6 MB = 3550.4 MB of wire, one launch toll, and a small-kernel
    #   term that is 0.221 ms if latency-bound or 0.884 ms if work-bound.
    ratio_opt, ratio_pes = 2.11, 2.41
    base = 441.4                       # plain single-token throughput

    print(f"binary {a.binary}  ngen={a.ngen}  drafts={a.drafts}")
    print(f"M=4 verification costs {ratio_opt:.2f}x .. {ratio_pes:.2f}x a plain "
          f"token, so break-even is E[tok/pass] = {ratio_opt:.2f} .. {ratio_pes:.2f}")
    print("greedy speculation with exact verification is lossless, so this "
          "simulation is exact\n")

    trajs = {}
    for c in cases:
        name = c.get("name", "?")
        entry = suite.get(name)
        ids = entry.get("prompt_ids") if entry else None
        if not ids:
            continue
        try:
            trajs[name] = (greedy(a.binary, ids, a.ngen), len(ids))
        except Exception as e:
            print(f"{name}: FAILED {e}")

    for mm in [int(x) for x in a.matches.split(",")]:
        print(f"--- minmatch = {mm} tokens ---")
        print(f"{'case':10s} {'trig%':>7s} {'E[tok/pass]':>12s} "
              f"{'worst tok/s':>12s} {'best tok/s':>11s}")
        agg_e, agg_n = 0.0, 0
        for name, (out, plen) in trajs.items():
            if len(out) <= plen + 1:
                print(f"{name:10s}   (trajectory shorter than the prompt)")
                continue
            per_pass, trig, total = simulate(out, plen, mm, a.drafts)
            if not per_pass:
                continue
            et = sum(per_pass) / len(per_pass)
            print(f"{name:10s} {100.0*trig/total:6.1f}% {et:12.2f} "
                  f"{base*et/ratio_pes:12.0f} {base*et/ratio_opt:11.0f}")
            agg_e += et * len(per_pass); agg_n += len(per_pass)
        if agg_n:
            et = agg_e / agg_n
            v = ("PAYS" if et > ratio_pes else
                 "borderline" if et > ratio_opt else "DOES NOT PAY")
            print(f"{'ALL':10s} {'':7s} {et:12.2f}   -> {v}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
