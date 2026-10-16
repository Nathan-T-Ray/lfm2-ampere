#!/usr/bin/env python3
"""Prompt-lookup (n-gram) drafting: would it actually pay, and on what text?

The single-GPU batch-1 ceiling is measured at 488.9 tok/s (Phase 4z.2), a second
card is out of scope, and a draft MODEL costs more than it saves.  Prompt-lookup
drafting is not that: it searches text ALREADY in the context for a match of the
last few tokens, copies the continuation as the draft, costs a host-side search,
and can be TRIGGERED -- when there is no match the plain M=1 graph runs and
nothing is lost.

Greedy speculative decoding with exact verification is LOSSLESS, so the
acceptance rate can be simulated exactly, offline, from trajectories the engine
has already generated.  No GPU time and no new kernels.  This decides whether an
M-pass build is worth starting.

WHAT CHANGED, AND WHY THIS TOOL WAS WRONG BEFORE
------------------------------------------------
The previous revision hardcoded

    ratio_opt, ratio_pes = 2.11, 2.41

and never varied it.  Three things were wrong with that:

  1. It is the M=**4** cost, applied to every `--drafts` setting.  The bar is a
     function of M, and M=4 is the WORST point on the curve (the bar rises
     ~0.3x per extra draft while the marginal acceptance decays).
  2. It charged `lm_head` once per position.  The first logit vector of a pass
     is the one the previous step already computed, so a K-position pass costs
     K-1 lm_head reads, not K.  See tools/lkbytes.py.
  3. `FIXED_MB` had no router term, 19.92 MB, missing from every prior model.

The tool therefore measured ONE point in a two-dimensional space
(`minmatch` x `drafts`) and reported the verdict for all of it.

`E/bar` is the objective, not `E`.  Maximising E alone always pushes toward
larger M, which is exactly the direction the bar grows.

Usage:
  tools/lookup.py                       # full sweep
  tools/lookup.py --ngen 384 --quick    # fewer match lengths
  tools/lookup.py --refit               # ignore the trajectory cache
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lkbytes as L

ROOT = pathlib.Path(__file__).resolve().parent.parent
REF = pathlib.Path(os.environ.get("LK_REF", ROOT / "ref"))
CACHE = pathlib.Path(os.environ.get("LK_TRAJ_CACHE", "/tmp/lk_traj"))

# the current measured single-token headline (ARCHITECTURE.md section 0)
BASE_TOKS = 443.04


def greedy(binary, ids, ngen, use_cache=True):
    """Run the engine, or read/write a cached trajectory.

    Generating 7 x ngen=384 trajectories is the expensive part of this tool and
    it does not depend on (minmatch, drafts), so it is cached.  The cache key
    includes the binary's size and mtime and ngen, so a rebuild invalidates it.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    key = None
    if use_cache:
        try:
            st = os.stat(binary)
            key = CACHE / ("%s_n%d_%d_%d_%s.json"
                           % (pathlib.Path(binary).name, ngen, st.st_size,
                              int(st.st_mtime), "+".join(map(str, ids))))
            if key.exists():
                return json.loads(key.read_text())
        except OSError:
            key = None

    env = dict(os.environ)
    env["LK_NGEN"] = str(ngen)
    env["CUDA_VISIBLE_DEVICES"] = env.get("LK_GPU", "0")
    r = subprocess.run([str(binary), "gen"] + [str(i) for i in ids],
                       capture_output=True, text=True, env=env, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    for line in r.stdout.splitlines():
        if line.startswith("out:"):
            out = [int(x) for x in line[4:].split()]
            if key:
                key.write_text(json.dumps(out))
            return out
    raise RuntimeError("no out: line")


def find_match(ctx, suffix):
    """Longest match of `suffix` ending at the last generated position.

    Returns the tokens that follow the match in the context, i.e. how many draft
    tokens are available.  Only text that would exist at generation time is
    searched, so this is what a real implementation can do.
    """
    n = len(ctx)
    Ls = len(suffix)
    for start in range(n - Ls - 1, -1, -1):          # most recent match wins
        if ctx[start:start + Ls] == suffix:
            nxt = ctx[start + Ls:]
            if nxt:
                return nxt
    return []


def simulate(traj, prompt_len, minmatch, drafts):
    """Walk the trajectory, drafting from context before each position."""
    per_pass = []          # tokens correctly emitted per verification pass
    triggered = 0
    total = 0
    for t in range(prompt_len, len(traj) - 1):
        total += 1
        suffix = traj[max(0, t - minmatch + 1):t + 1]
        if len(suffix) < minmatch:
            per_pass.append(1)
            continue
        cand = find_match(traj[:t + 1], suffix)
        if not cand:
            per_pass.append(1)
            continue
        triggered += 1
        draft = cand[:drafts]
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
    ap.add_argument("--drafts", default="1,2,3",
                    help="draft counts to sweep; M = drafts + 1")
    ap.add_argument("--matches", default="2,3,4,5,6")
    ap.add_argument("--u", default="",
                    help="comma-separated U_K override, else lkbytes table")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refit", action="store_true",
                    help="ignore the trajectory cache")
    a = ap.parse_args()

    drafts_list = [int(x) for x in a.drafts.split(",")]
    matches = [int(x) for x in a.matches.split(",")]
    if a.quick:
        matches = [2, 4]

    U = dict(L.U_CONSECUTIVE)
    if a.u:
        U = {i + 1: float(x) for i, x in enumerate(a.u.split(","))}

    ref = json.loads((REF / "oracle_ref.json").read_text())
    cases = ref["cases"]
    suite = {c["name"]: c for c in
             json.loads((REF / "parity_suite.json").read_text())["cases"]}

    print(f"binary {a.binary}   ngen={a.ngen}   base {BASE_TOKS:.2f} tok/s")
    print("greedy speculation with exact verification is lossless, so this "
          "simulation is exact\n")
    print("break-even bars, corrected for the lm_head redundancy and the "
          "missing router term:")
    print(f"  {'M':>2} {'drafts':>6} {'bar':>16}  {'legacy bar':>16}")
    for d in sorted(set(drafts_list)):
        K = d + 1
        if K not in U:
            continue
        c = L.ratio_spec(U[K], K)
        l = L.ratio_spec(U[K], K, legacy=True)
        print(f"  {K:2d} {d:6d} {c[0]:7.3f} -{c[1]:7.3f}  "
              f"{l[0]:7.3f} -{l[1]:7.3f}")
    print(f"\n  (legacy = what this tool hardcoded before: 2.11-2.41 for M=4, "
          f"applied to every M)\n")

    trajs = {}
    for c in cases:
        name = c.get("name", "?")
        entry = suite.get(name)
        ids = entry.get("prompt_ids") if entry else None
        if not ids:
            continue
        try:
            trajs[name] = (greedy(a.binary, ids, a.ngen,
                                  use_cache=not a.refit), len(ids))
        except Exception as e:
            print(f"{name}: FAILED {e}")

    results = {}
    for d in drafts_list:
        K = d + 1
        if K not in U:
            print(f"(no U for M={K}; skipping)")
            continue
        bar = L.ratio_spec(U[K], K)
        print(f"=== drafts={d}  (M={K}, lm_head runs {L.lm_head_evals_spec(K)}x, "
              f"bar {bar[0]:.3f}-{bar[1]:.3f}) ===")
        hdr = (f"{'case':10s} {'minm':>4s} {'trig%':>7s} {'E':>6s} {'E/bar':>13s} "
               f"{'tok/s':>13s}  verdict")
        print(hdr)
        print("-" * len(hdr))
        for mm in matches:
            agg_e, agg_n = 0.0, 0
            rows = []
            for name, (out, plen) in trajs.items():
                if len(out) <= plen + 1:
                    continue
                per_pass, trig, total = simulate(out, plen, mm, d)
                if not per_pass:
                    continue
                et = sum(per_pass) / len(per_pass)
                rows.append((name, et, 100.0 * trig / total))
                agg_e += et * len(per_pass)
                agg_n += len(per_pass)
            if not agg_n:
                continue
            et = agg_e / agg_n
            results[(mm, d)] = et
            lo_t = BASE_TOKS * et / bar[1]
            hi_t = BASE_TOKS * et / bar[0]
            v = L.verdict(et, bar)
            print(f"{'ALL':10s} {mm:4d} {'':7s} {et:6.2f} "
                  f"{et/bar[1]:6.2f}-{et/bar[0]:5.2f} "
                  f"{lo_t:6.0f}-{hi_t:5.0f}  {v}")
            for name, et_c, trig in sorted(rows, key=lambda r: -r[1]):
                vc = L.verdict(et_c, bar)
                print(f"{name:10s} {mm:4d} {trig:6.1f}% {et_c:6.2f} "
                      f"{et_c/bar[1]:6.2f}-{et_c/bar[0]:5.2f} "
                      f"{BASE_TOKS*et_c/bar[1]:6.0f}-{BASE_TOKS*et_c/bar[0]:5.0f}  {vc}")
        print()

    # --- the monotone cross-check -----------------------------------------
    if (2, 3) in results:
        e4 = results[(2, 3)]
        bound = L.monotone_lower_bound(e4, 4, 2)
        print("monotone cross-check (per-draft survival is non-increasing, so the")
        print("K=4 measurement bounds the K=2 one from below):")
        print(f"  E(K=4, minmatch=2) = {e4:.3f}  ->  E(K=2) >= {bound:.3f}")
        for d in (1, 2):
            if (2, d) in results:
                got = results[(2, d)]
                ok = "OK" if got >= bound - 1e-9 else "*** SIMULATION BELOW THE BOUND ***"
                print(f"  simulated E(K={d+1}, minmatch=2) = {got:.3f}   {ok}")
                if got < bound - 1e-9:
                    print("  the bound is arithmetic, not an estimate: the "
                          "simulation is wrong")
        print()

    # --- the recommendation ------------------------------------------------
    if results:
        print("best (minmatch, drafts) by E/bar, sweeping only configs that PAY:")
        best = []
        for (mm, d), et in results.items():
            K = d + 1
            bar = L.ratio_spec(U[K], K)
            best.append((et / bar[1], mm, d, et, bar[1], L.verdict(et, bar)))
        best.sort(reverse=True)
        for r, mm, d, et, blo, v in best[:8]:
            print(f"  minmatch={mm} drafts={d}  E={et:.2f} bar>={blo:.3f}  "
                  f"E/bar={r:.3f}  {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
