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

Three estimators, all from the same router trace:
  consecutive  len(set(experts[s : s+B]))            temporal + marginal
  independent  len(set(experts[B random positions])) marginal only
  uniform      64*(1-(60/64)^B)                      analytic floor (lowest
                                                     possible U, i.e. the most
                                                     optimistic case there is)
"""
import sys
import numpy as np

LK_N_LAYERS, LK_TOP_K = 40, 4
MOE_LAYERS = [i for i in range(LK_N_LAYERS) if i >= 2]
EXPERT_MB = 3 * (1536 * 2048 / 32 * 18) / 1e6
LM_HEAD_MB = 215.04
FIXED_MB = 283.0 + 59.0 + 81.0 + 3.0
BASE_MB = 1466.75
NON_MOE_MB = BASE_MB - len(MOE_LAYERS) * 4 * EXPERT_MB


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


def mbps(U, B):
    """bytes per token for batch B with U distinct experts per MoE layer."""
    return (B * LM_HEAD_MB + len(MOE_LAYERS) * U * EXPERT_MB + FIXED_MB) / B


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/route.bin"
    a = np.fromfile(path, dtype=np.int32)
    nsteps = a.size // (LK_N_LAYERS * LK_TOP_K)
    a = a[: nsteps * LK_N_LAYERS * LK_TOP_K].reshape(nsteps, LK_N_LAYERS, LK_TOP_K)
    R = a[:, MOE_LAYERS, :]
    print("trace: %d positions x %d MoE layers" % (nsteps, len(MOE_LAYERS)))
    print("non-MoE %.1f MB/tok | 4-expert MoE %.1f MB | total %.2f MB"
          % (NON_MOE_MB, len(MOE_LAYERS) * 4 * EXPERT_MB, BASE_MB))
    print("")
    hdr = (" B |  U_cons  reuse   MB/tok  spd |  U_indep  reuse   MB/tok  spd |"
           "  U_unif  spd")
    print(hdr)
    print("-" * len(hdr))
    rng = np.random.default_rng(12345)
    for B in range(1, 13):
        cs, ind = [], []
        for li in range(R.shape[1]):
            cs += union_counts(R[:, li, :], B, "consecutive", rng)
            ind += union_counts(R[:, li, :], B, "independent", rng)
        Uc, Ui = float(np.mean(cs)), float(np.mean(ind))
        Uu = 64 * (1 - (60 / 64) ** B)
        mc, mi, mu = mbps(Uc, B), mbps(Ui, B), mbps(Uu, B)
        print("%2d | %7.2f %5.2fx %8.1f %5.2fx | %7.2f %5.2fx %8.1f %5.2fx |"
              " %7.2f %5.2fx"
              % (B, Uc, 4 * B / Uc, mc, BASE_MB / mc,
                 Ui, 4 * B / Ui, mi, BASE_MB / mi,
                 Uu, BASE_MB / mu))
    print("")
    print("spd = byte-floor speed-up vs batch 1, ASSUMING unchanged bandwidth")
    print("efficiency.  It is an upper bound on the real gain, not a prediction.")


if __name__ == "__main__":
    main()
