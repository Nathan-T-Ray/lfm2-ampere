#!/usr/bin/env python3
"""Gate G4b: how skewed is the MoE routing across consecutive tokens?

Phase 5 (speculative decoding) only pays if verifying K tokens in one pass
touches far fewer than 4K experts per layer: the entire win is reading each
expert's weights once and applying them to several tokens.  That depends on the
routing distribution, which cannot be argued from the architecture.

U_K = number of DISTINCT experts selected by K consecutive tokens at one layer
(union of their top-4 sets).  Baseline is 4K (one read per token per expert).

Also prices the full pass, because the byte model has a term that hurts:
`lm_head` must produce logits at every DRAFT position.  It does NOT have to
produce one at the seed position -- those logits are the ones the previous step
already computed to emit the seeding token -- so a K-position pass runs lm_head
K-1 times, not K.  See tools/lkbytes.py for the whole model and for the bracket.

The `break-even A` column is the number of tokens a pass must emit to beat
plain decoding.  Compare it against the measured E from tools/lookup.py.

Usage:  tools/g4b.py [route.bin] [nsteps]
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lkbytes as L


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/route.bin"
    nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else None
    a = np.fromfile(path, dtype=np.int32)
    if nsteps is None:
        nsteps = a.size // (L.LK_N_LAYERS * L.LK_TOP_K)
    a = a[: nsteps * L.LK_N_LAYERS * L.LK_TOP_K].reshape(
        nsteps, L.LK_N_LAYERS, L.LK_TOP_K)
    print(f"route log: {nsteps} positions x {L.LK_N_LAYERS} layers x "
          f"{L.LK_TOP_K} experts")
    print(f"expert bytes: {L.EXPERT_MB:.3f} MB each, {len(L.MOE_LAYERS)} MoE "
          f"layers\n")

    R = a[:, L.MOE_LAYERS, :]                       # [steps][38][4]
    print("expert-selection frequency skew (per layer, averaged):")
    counts = np.zeros(L.LK_N_EXPERTS)
    for li in range(R.shape[1]):
        for s in range(R.shape[0]):
            counts[R[s, li]] += 1
    counts /= counts.sum()
    top = np.sort(counts)[::-1]
    print(f"  uniform would be {1/L.LK_N_EXPERTS:.5f} each")
    print(f"  most-used expert  {top[0]:.5f}  ({top[0]*L.LK_N_EXPERTS:.1f}x uniform)")
    print(f"  top-8 share       {top[:8].sum()*100:.1f}%")
    print(f"  top-16 share      {top[:16].sum()*100:.1f}%")
    print(f"  experts never used: {int((counts == 0).sum())}/{L.LK_N_EXPERTS}")
    ent = -(counts[counts > 0] * np.log(counts[counts > 0])).sum()
    print(f"  entropy           {ent:.3f} nats (max {np.log(L.LK_N_EXPERTS):.3f})\n")

    print(f"baseline (no speculation): {L.MEASURED_BASE_MB:.2f} MB/token, "
          f"{L.STEP_MS:.3f} ms\n")

    hdr = (" K  E[U_K]  4K  reuse  lmhd   pass MB   bar(corrected)  "
           "bar(legacy)  break-even A")
    print(hdr)
    print("-" * len(hdr))
    for K in range(1, 9):
        Us = []
        for li in range(R.shape[1]):
            col = R[:, li, :]
            for s in range(0, R.shape[0] - K + 1):
                Us.append(len(set(col[s:s + K].ravel().tolist())))
        U = float(np.mean(Us))
        pm = L.pass_mb_spec(U, K)
        c = L.ratio_spec(U, K)
        l = L.ratio_spec(U, K, legacy=True)
        print(f"{K:2d}  {U:6.2f}  {4*K:3d}  {4*K/U:5.2f}x  "
              f"{L.lm_head_evals_spec(K):4d}  {pm:9.1f}   "
              f"{c[0]:.3f}-{c[1]:.3f}    {l[0]:.3f}-{l[1]:.3f}    "
              f"{c[0]:.2f}-{c[1]:.2f}")

    print("\n'bar' = the pass time as a multiple of one plain step, so it IS the")
    print("break-even E[tokens/pass]: a pass must emit more than this to win.")
    print("The bracket is RAMP and SMALL scaling; see tools/lkbytes.py.  The")
    print("LEGACY column is what this tool computed before the lm_head fix and")
    print("is kept so the change is auditable rather than silent.")
    print("\nFor comparison against a measured E: the document's own M=4 run gave")
    print("E=1.87 averaged over 7 cases, and E=3.61 on `factual`.")


if __name__ == "__main__":
    main()
