#!/usr/bin/env python3
"""Paired per-token loss comparison -- the gate that can actually see a
residual-stream regression.

The problem this solves
-----------------------
`ppl` mode reports one number: mean nll over the corpus.  On the 557-token
natural corpus the per-token losses have sd ~ 2.9 nats, so the standard error
of that mean is 2.9/sqrt(557) ~ 0.12 nats, i.e. ~11% of the perplexity.  It
cannot resolve the ~1% regression a kernel change produces, which is the
acknowledged blind spot: C2 recomputes lm_head in float64 from the residual
stream the engine dumped, so it bounds lm_head's error GIVEN a residual but
not the residual's own error.

Why pairing works here
----------------------
`ppl` mode is TEACHER-FORCED: both builds are fed the identical token stream
from the corpus file, so the loss at position t is evaluated for the same
target under the same prefix in both runs.  Differencing per position cancels
token difficulty exactly:

    se(delta) = sd(nll_b - nll_a) / sqrt(N)

Because the two runs are highly correlated, sd(delta) is one to two orders of
magnitude below sd(nll), and the same 557 tokens resolve ~0.1% instead of ~11%.

Usage
-----
    LK_PPL_DUMP=a.txt ./build/engine_a ppl ref/ppl_natural.bin
    LK_PPL_DUMP=b.txt ./build/engine_b ppl ref/ppl_natural.bin
    tools/paired.py a.txt b.txt

A regression is real only if |mean delta| exceeds ~2 * se.  The tool prints
that verdict rather than leaving it to the reader.
"""
import sys
import math


def load(path):
    nll, arg, tgt = [], [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) != 4:
                continue
            nll.append(float(p[1]))
            arg.append(int(p[2]))
            tgt.append(int(p[3]))
    return nll, arg, tgt


def mean(v):
    return sum(v) / len(v)


def sd(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    an, aa, at = load(sys.argv[1])
    bn, ba, bt = load(sys.argv[2])
    if not an or not bn:
        print("one of the dumps is empty")
        return 2
    if at != bt:
        print(f"NOT PAIRED: the two runs scored different targets "
              f"({len(at)} vs {len(bt)} positions)")
        return 2
    if len(an) != len(bn):
        print(f"NOT PAIRED: {len(an)} vs {len(bn)} positions")
        return 2
    N = len(an)

    na, nb = mean(an), mean(bn)
    d = [y - x for x, y in zip(an, bn)]
    md = mean(d)
    sdd = sd(d)
    se = sdd / math.sqrt(N)

    # what the un-paired comparison would have given, for contrast
    se_unpaired = math.sqrt(sd(an) ** 2 / N + sd(bn) ** 2 / N)

    t1a = sum(1 for x, y in zip(aa, at) if x == y)
    t1b = sum(1 for x, y in zip(ba, bt) if x == y)
    flip = sum(1 for x, y in zip(aa, ba) if x != y)

    print(f"positions          {N}")
    print(f"nll   A            {na:.6f}   ppl {math.exp(na):.6f}")
    print(f"nll   B            {nb:.6f}   ppl {math.exp(nb):.6f}")
    print(f"perplexity ratio   {math.exp(nb - na):.6f}   "
          f"({(math.exp(nb - na) - 1) * 100:+.3f}%)")
    print()
    print(f"mean delta nll     {md:+.6f} nats")
    print(f"sd(delta)          {sdd:.6f}   (vs sd(nll) ~ {sd(an):.4f})")
    print(f"se(paired)         {se:.6f}")
    print(f"se(unpaired)       {se_unpaired:.6f}   "
          f"-> pairing sharpens by {se_unpaired / se:.1f}x" if se > 0 else "")
    if se > 0:
        t = md / se
        lo, hi = md - 1.96 * se, md + 1.96 * se
        print(f"t = mean/se         {t:+.2f}")
        print(f"95% CI on delta    [{lo:+.6f}, {hi:+.6f}] nats")
        print(f"95% CI on ppl      [{math.exp(lo):.6f}, {math.exp(hi):.6f}]")

        # The paired delta has a heavy tail: at the positions where the top-1
        # prediction FLIPS the loss can move by whole nats, and those positions
        # dominate sd(delta).  Two robust readings are reported alongside the
        # mean, because a mean that is dragged by a dozen chaotic positions is
        # the wrong instrument for a ~1% regression.
        ds = sorted(d)
        med = ds[N // 2] if N % 2 else 0.5 * (ds[N // 2 - 1] + ds[N // 2])
        npos = sum(1 for x in d if x > 0)
        nneg = sum(1 for x in d if x < 0)
        print(f"median delta       {med:+.6f} nats")
        if npos + nneg:
            # two-sided sign test, normal approximation
            z = (npos - nneg) / math.sqrt(npos + nneg)
            print(f"sign test          {npos} up / {nneg} down   z = {z:+.2f}"
                  f"   {'SYMMETRIC' if abs(z) < 2 else 'SHIFTED'}")

        # restricted to positions where the two builds still predict the same
        # token: this is the smooth part of the distribution shift.
        same = [y - x for x, y, u, v in zip(an, bn, aa, ba) if u == v]
        if len(same) > 2:
            sm, ss = mean(same), sd(same)
            sse = ss / math.sqrt(len(same))
            print(f"non-flip subset    n={len(same)}  mean {sm:+.6f}  "
                  f"se {sse:.6f}  t = {sm / sse:+.2f}"
                  if sse > 0 else f"non-flip subset    n={len(same)}  mean {sm:+.6f}")
        print()
        if abs(t) < 2.0:
            print(f"VERDICT  indistinguishable at 2 se "
                  f"(|delta| = {abs(md):.6f} < {2 * se:.6f}); "
                  f"this run bounds a regression to {2 * se:.6f} nats "
                  f"= {(math.exp(2 * se) - 1) * 100:.3f}% of perplexity")
        else:
            print(f"VERDICT  REGRESSION: mean delta exceeds 2 se "
                  f"({md:+.6f} vs {2 * se:.6f})")
    print()
    print(f"top-1 agreement A  {t1a}/{N} = {t1a / N:.4f}")
    print(f"top-1 agreement B  {t1b}/{N} = {t1b / N:.4f}")
    print(f"prediction flips   {flip}/{N} = {flip / N:.4f}  "
          f"(a chaos amplifier; reported, never gated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
