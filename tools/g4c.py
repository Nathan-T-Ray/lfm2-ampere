#!/usr/bin/env python3
"""Gate G4c: expert reuse at batch B, for BATCHING rather than speculation.

G4b answered "how many distinct experts do K *consecutive* tokens need", which
is the speculative-decoding question.  Continuous batching asks a different one:
B tokens that come from B DIFFERENT sequences, i.e. positions that are not
temporally adjacent and whose only shared structure is the static marginal
expert skew.

The distinction matters and is easy to get wrong in the optimistic direction:
temporal adjacency adds real reuse on top of marginal skew, so the speculative
number is an UPPER bound on what batching can get.

It is also the distinction that decides the byte model.  Batching has no
acceptance rate -- every one of the B tokens is real -- but every one of them
needs its OWN logits, so lm_head runs B times.  Speculation runs lm_head K-1
times for a K-position pass.  Conflating the two produces numbers that look
like independent confirmations of each other and are not.

Three estimators, all from the same router trace:
  consecutive  len(set(experts[s : s+B]))            temporal + marginal
  independent  len(set(experts[B random positions])) marginal only
  uniform      64*(1-(60/64)^B)                      analytic floor (lowest
                                                     possible U, i.e. the most
                                                     optimistic case there is)

Usage:  tools/g4c.py [route.bin]
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lkbytes as L


def union_counts(col, B, mode, rng, ndraw=400):
    """col: [steps, 4] int.  Return list of |union| over B tokens."""
    n = col.shape[0]
    out = []
    if mode == "consecutive":
        for s in range(0, n - B + 1):
            out.append(len(set(col[s:s + B].ravel().tolist())))
    else:
        for _ in range(ndraw):
            idx = rng.choice(n, size=B, replace=False)
            out.append(len(set(col[idx].ravel().tolist())))
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/route.bin"
    a = np.fromfile(path, dtype=np.int32)
    nsteps = a.size // (L.LK_N_LAYERS * L.LK_TOP_K)
    a = a[: nsteps * L.LK_N_LAYERS * L.LK_TOP_K].reshape(
        nsteps, L.LK_N_LAYERS, L.LK_TOP_K)
    R = a[:, L.MOE_LAYERS, :]
    print("trace: %d positions x %d MoE layers" % (nsteps, len(L.MOE_LAYERS)))
    base = L.pass_mb_batch(4.0, 1)
    print("non-expert weights (itemised) %.1f MB | router %.2f MB | lm_head %.2f MB"
          % (L.FIXED_MB - L.ROUTER_MB, L.ROUTER_MB, L.LM_HEAD_MB))
    print("4-expert MoE %.1f MB | model base %.2f MB vs measured %.2f (%+.3f%%)"
          % (len(L.MOE_LAYERS) * 4 * L.EXPERT_MB, base, L.MEASURED_BASE_MB,
             100.0 * (base - L.MEASURED_BASE_MB) / L.MEASURED_BASE_MB))
    print("")
    hdr = (" B |  U_indep  reuse   MB/tok  speedup(lower-upper) |  U_cons   reuse   "
           "MB/tok  speedup |  U_unif  speedup")
    print(hdr)
    print("-" * len(hdr))
    rng = np.random.default_rng(12345)
    for B in range(1, 13):
        ind = []
        for li in range(R.shape[1]):
            ind += union_counts(R[:, li, :], B, "independent", rng)
        Ui = float(np.mean(ind))
        Uu = L.LK_N_EXPERTS * (1 - ((L.LK_N_EXPERTS - L.LK_TOP_K) / L.LK_N_EXPERTS) ** B)
        mi = L.pass_mb_batch(Ui, B)
        gi = L.ratio_batch(Ui, B)
        # consecutive, for contrast only -- never for a batching claim
        cs = []
        for li in range(R.shape[1]):
            cs += union_counts(R[:, li, :], B, "consecutive", rng)
        Uc = float(np.mean(cs))
        mc = L.pass_mb_batch(Uc, B)
        gc = L.ratio_batch(Uc, B)
        gu = L.ratio_batch(Uu, B)
        print("%2d | %7.2f %5.2fx %8.1f  %.3f-%.3f x      | %7.2f %5.2fx %8.1f  "
              "%.3f-%.3f x   | %7.2f %.3f x"
              % (B, Ui, L.LK_TOP_K * B / Ui, mi / B, gi[0], gi[1],
                 Uc, L.LK_TOP_K * B / Uc, mc / B, gc[0], gc[1],
                 Uu, gu[0]))
    print("")
    print("speedup = tokens per unit time vs batch 1, from the measured step")
    print("decomposition, bracketing whether RAMP and SMALL scale with the batch.")
    print("The INDEPENDENT column is the one to quote: a batch holds B tokens from")
    print("B different sequences.  The CONSECUTIVE column is what speculation")
    print("would get, and using it for batching is optimistic by construction.")


if __name__ == "__main__":
    main()
