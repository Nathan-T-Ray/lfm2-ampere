#!/usr/bin/env python3
"""Gate G4b: how skewed is the MoE routing across consecutive tokens?

Phase 5 (speculative decoding) only pays if verifying K tokens in one pass
touches far fewer than 4K experts per layer: the entire win is reading each
expert's weights once and applying them to several tokens.  That depends on the
routing distribution, which cannot be argued from the architecture.

U_K = number of DISTINCT experts selected by K consecutive tokens at one layer
(union of their top-4 sets).  Baseline is 4K (one read per token per expert).

Also prices the full pass, because the byte model has a term that hurts:
lm_head must produce logits at every verified position, so it costs K x 215 MB
per pass no matter how good the expert reuse is.
"""
import sys
import numpy as np

LK_N_LAYERS, LK_TOP_K = 40, 4
MOE_LAYERS = [i for i in range(LK_N_LAYERS) if i >= 2]
# bytes per expert per MoE layer: gate 1536x2048 + up 1536x2048 + down 2048x1536, Q4_0
EXPERT_MB = 3 * (1536 * 2048 / 32 * 18) / 1e6
LM_HEAD_MB = 215.04
FIXED_MB = 283.0 + 59.0 + 81.0 + 3.0      # shortconv + attention + dense + misc
BASE_MB = 1466.75


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/route.bin"
    nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else None
    a = np.fromfile(path, dtype=np.int32)
    if nsteps is None:
        nsteps = a.size // (LK_N_LAYERS * LK_TOP_K)
    a = a[: nsteps * LK_N_LAYERS * LK_TOP_K].reshape(nsteps, LK_N_LAYERS, LK_TOP_K)
    print(f"route log: {nsteps} positions x {LK_N_LAYERS} layers x {LK_TOP_K} experts")
    print(f"expert bytes: {EXPERT_MB:.3f} MB each, {len(MOE_LAYERS)} MoE layers\n")

    R = a[:, MOE_LAYERS, :]                       # [steps][38][4]
    print("expert-selection frequency skew (per layer, averaged):")
    counts = np.zeros(64)
    for li in range(R.shape[1]):
        for s in range(R.shape[0]):
            counts[R[s, li]] += 1
    counts /= counts.sum()
    top = np.sort(counts)[::-1]
    print(f"  uniform would be {1/64:.5f} each")
    print(f"  most-used expert  {top[0]:.5f}  ({top[0]*64:.1f}x uniform)")
    print(f"  top-8 share       {top[:8].sum()*100:.1f}%")
    print(f"  top-16 share      {top[:16].sum()*100:.1f}%")
    print(f"  experts never used: {int((counts == 0).sum())}/64")
    ent = -(counts[counts > 0] * np.log(counts[counts > 0])).sum()
    print(f"  entropy           {ent:.3f} nats (max {np.log(64):.3f})\n")

    print(" K   E[U_K]   4K   reuse   pass MB   MB/token @100%   break-even A")
    for K in range(1, 9):
        # union over K consecutive positions, averaged over layers and offsets
        Us = []
        for li in range(R.shape[1]):
            col = R[:, li, :]
            for s in range(0, R.shape[0] - K + 1):
                Us.append(len(set(col[s:s + K].ravel().tolist())))
        U = float(np.mean(Us))
        per_pass = K * LM_HEAD_MB + len(MOE_LAYERS) * U * EXPERT_MB + FIXED_MB
        mb_per_tok = per_pass / K
        # acceptance A at which this pass costs the same as A baseline steps
        be = per_pass / BASE_MB
        print(f"{K:2d}  {U:6.2f}  {4*K:3d}  {4*K/U:5.2f}x  {per_pass:7.1f}  "
              f"{mb_per_tok:8.1f}  ({BASE_MB/mb_per_tok:4.2f}x)   {be:.2f}")
    print(f"\nbaseline (no speculation): {BASE_MB:.1f} MB/token")
    print("'break-even A' = accepted tokens per pass needed to match the baseline.")
    print("A pass must beat it for speculation to be a net win.")


if __name__ == "__main__":
    main()
