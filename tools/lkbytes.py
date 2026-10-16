#!/usr/bin/env python3
"""The byte model for a multi-token pass, and the break-even it implies.

Why this file exists
--------------------
Four separate analyses (this project's `tools/g4b.py`, `tools/g4c.py`, the
Phase 4z.3 notes, and an external reviewer) all charged **one `lm_head` read per
position in a verification pass**.  That is wrong by exactly one read: the first
logit vector of a pass is the one the PREVIOUS step already computed to emit the
token that seeds the pass.  It is in hand, so it is not recomputed.

The overcharge is a **fixed 215.04 MB per pass**, independent of how many
positions the pass holds.  That is why it is invisible at M=1 -- which is where
every headline number in this project comes from -- and why it matters most at
small M.

A second, unrelated omission: `FIXED_MB` never had a router term, while
`lk_prof_bytes` in `src/engine.cu` charges `router_dot` 524288 B per layer.  Over
38 MoE layers that is 19.92 MB, which is the whole of the unreconciled "known
inconsistency #5" (18.85 MB) to within 1.2 MB.

Batching and speculation are NOT the same formula
-------------------------------------------------
They differ in who needs logits:

  speculation  K positions in the pass = 1 seed + (K-1) drafts.
               Only the (K-1) draft positions need logits; the seed's are held.
               The pass yields UP TO K tokens but only E[tokens] on average,
               where E depends on how predictable the text is.

  batching     B independent sequences.  Every one of them needs its own logits
               to continue, so lm_head runs B times.  The pass ALWAYS yields
               exactly B tokens.  There is no acceptance rate to argue about.

Getting these two confused is how a 1.84x batching figure and a 2.01x
verification-cost figure ended up in one paragraph as if each confirmed the
other.

The bracket, and why it is a bracket
------------------------------------
T(1) = 2.266 ms decomposes (Phase 4z.2, all terms measured):

    1.557 ms  wire, 1466.75 MB at the 941.8 GB/s marginal rate
    0.220 ms  streaming size inefficiency -- small kernels do not amortise ramp
    0.268 ms  launch toll, 343 nodes x 0.781 us, from an empty-kernel graph
    0.221 ms  the 158 small kernels: execution plus the stalls they impose

Three of the four scale differently with the number of tokens in a pass, and
two of them are genuinely unknown.  Rather than pick the flattering end:

  TOLL   is per NODE, and the node count barely moves, so it is CONSTANT and
         amortises across the pass.  This is the only term batching gets free.
  WIRE   is per BYTE, so it scales with R = pass_MB / base_MB.  Not in doubt.
  RAMP   is per KERNEL per pass.  The weight bytes per kernel are shared across
         the pass, so the kernels do not get bigger and ramp should NOT scale
         with R -- but the grids do grow, which argues the other way.  Bracketed.
  SMALL  is per TOKEN of activation work, so it scales with the token count --
         but it also contains dependency stalls, which do not.  Bracketed.

Hence, with R = pass_MB/base_MB and n = tokens flowing through the pass:

    lower = R*WIRE + TOLL + RAMP + SMALL              (nothing but bytes scales)
    upper = R*(WIRE+RAMP) + TOLL + SMALL*n            (everything scales)

At the two ends of that bracket the LEGACY model reproduces BOTH of the figures
this project has already published for M=4 -- 1.98x and 2.41x.  They were never
in conflict; they were the two ends of a bracket that nobody wrote down.  That
is now `_selftest`.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# geometry (mirrors src/kernels.cuh)
# --------------------------------------------------------------------------
LK_N_LAYERS = 40
LK_HIDDEN = 2048
LK_N_EXPERTS = 64
LK_TOP_K = 4
LK_MOE_INTER = 1536
MOE_LAYERS = [i for i in range(LK_N_LAYERS) if i >= 2]      # 38 MoE layers

# bytes per expert per MoE layer: gate 1536x2048 + up 1536x2048 + down 2048x1536,
# Q4_0 = 18 bytes per 32 elements.
EXPERT_MB = 3 * (LK_MOE_INTER * LK_HIDDEN / 32 * 18) / 1e6   # 5.308416

LM_HEAD_MB = 215.04                        # 128000 x 2048, Q6_K

# --------------------------------------------------------------------------
# the non-expert, non-lm_head weights.  Every row name is a key in
# `lk_prof_bytes` (src/engine.cu) and every multiplier is that kernel's real
# call count per token.  ROUTER_MB was MISSING from every previous revision.
# --------------------------------------------------------------------------
SHORTCONV_MB = 30 * (7077888 + 24576 + 2359296) / 1e6   # in + conv + out
ATTN_MB      = 10 * (3538944 + 2359296) / 1e6           # attn_qkv + attn_out
DENSE_MB     =  2 * (3 * 13565952) / 1e6                # dense_gate + up + down
ROUTER_MB    = 38 * 524288.0 / 1e6                      # <-- was missing

# The measured, ctx-independent traffic constant the engine reports as
# BYTES_PER_TOKEN.  This is the authority; the model must reconcile to it.
MEASURED_BASE_MB = 1466.75

# Everything small that is not worth itemising: 40 norm_attn + 38 norm_ffn +
# norm_final + embed + argmax_part.  Kept explicit rather than folded into the
# other terms so the residual below is a real check instead of a tautology.
MISC_MB = 1.81

FIXED_MB_LEGACY = 283.0 + 59.0 + 81.0 + 3.0             # what every prior model used
FIXED_MB = SHORTCONV_MB + ATTN_MB + DENSE_MB + ROUTER_MB + MISC_MB

# --------------------------------------------------------------------------
# the measured single-token step decomposition at ctx 272 (Phase 4z.2), in ms
# --------------------------------------------------------------------------
WIRE_MS  = 1.557
RAMP_MS  = 0.220
TOLL_MS  = 0.268
SMALL_MS = 0.221
STEP_MS  = 2.266
BASE_BYTES = MEASURED_BASE_MB

# --------------------------------------------------------------------------
# measured distinct-expert counts, from a 519-position route trace
# (`tools/g4b.py /tmp/route.bin`).  Present so the tools work without the trace
# and so these numbers are citable instead of folklore.
#
#   consecutive  = K tokens of ONE sequence   -> SPECULATION
#   independent  = B tokens of B sequences    -> BATCHING
#
# Batching must use `independent`.  Using `consecutive` is optimistic by exactly
# the amount that is easiest to get wrong: temporal adjacency adds real reuse
# on top of the static marginal skew.
# --------------------------------------------------------------------------
U_CONSECUTIVE = {1: 4.00, 2: 6.73, 3: 9.12, 4: 11.22, 5: 13.12,
                 6: 14.85, 7: 16.44, 8: 17.91}
U_INDEPENDENT = {1: 4.00, 2: 7.35, 3: 10.31, 4: 12.89, 5: 15.23,
                 6: 17.39, 7: 19.34, 8: 21.07}


def lm_head_evals_spec(K: int) -> int:
    """Logit vectors a K-position speculation pass must actually compute.

    K = positions in the pass = 1 seed + (K-1) drafts.  The seed's logits were
    computed by the previous step, so the pass needs K-1 of them.  K=1 is a
    plain decode step, which computes its own single logit vector.
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    return max(1, K - 1)


def lm_head_evals_batch(B: int) -> int:
    """B independent sequences all need their own logits.  No redundancy."""
    if B < 1:
        raise ValueError("B must be >= 1")
    return B


def pass_mb_spec(U: float, K: int, legacy: bool = False) -> float:
    """Bytes for a K-position SPECULATION pass with U distinct experts/layer."""
    lm = K if legacy else lm_head_evals_spec(K)
    fixed = FIXED_MB_LEGACY if legacy else FIXED_MB
    return lm * LM_HEAD_MB + len(MOE_LAYERS) * U * EXPERT_MB + fixed


def pass_mb_batch(U: float, B: int, legacy: bool = False) -> float:
    """Bytes for a B-sequence BATCHING pass with U distinct experts/layer."""
    fixed = FIXED_MB_LEGACY if legacy else FIXED_MB
    return lm_head_evals_batch(B) * LM_HEAD_MB + len(MOE_LAYERS) * U * EXPERT_MB + fixed


def _bracket(pass_mb: float, ntok: int) -> tuple[float, float]:
    """Pass time as a multiple of one plain step.  (lower, upper)."""
    R = pass_mb / BASE_BYTES
    lo = R * WIRE_MS + TOLL_MS + RAMP_MS + SMALL_MS
    hi = R * (WIRE_MS + RAMP_MS) + TOLL_MS + SMALL_MS * ntok
    return lo / STEP_MS, hi / STEP_MS


def ratio_spec(U: float, K: int, legacy: bool = False) -> tuple[float, float]:
    """Break-even E[tokens/pass] for a K-position speculation pass.

    `ratio(1) == 1.0` is the invariant that catches drift in any term above, so
    the self-test asserts it.
    """
    return _bracket(pass_mb_spec(U, K, legacy), max(1, K))


def ratio_batch(U: float, B: int, legacy: bool = False) -> tuple[float, float]:
    """Speed-up (tokens per unit time) for a B-sequence batching pass."""
    lo, hi = _bracket(pass_mb_batch(U, B, legacy), B)
    return B / hi, B / lo          # inverted: a cheaper pass is a bigger gain


def verdict(E: float, bar: tuple[float, float]) -> str:
    """PAYS if E clears the pessimistic bar; DOES NOT PAY if it misses the
    optimistic one; BREAK-EVEN in between -- which is a real answer, not a
    hedge, because the bracket is a measurable quantity, not ignorance."""
    lo, hi = bar
    if E > hi:
        return "PAYS"
    if E < lo:
        return "DOES NOT PAY"
    return "BREAK-EVEN"


def monotone_lower_bound(E_at_K: float, K: int, want_K: int) -> float:
    """E[want_K] >= this, from per-draft survival being non-increasing.

    With E(K) = 1 + sum(p_i), i = 1..K-1, and p_1 >= p_2 >= ... , the mean
    per-draft survival (E(K)-1)/(K-1) is a LOWER bound on p_1, so
    E(want_K) >= 1 + (want_K-1) * (E(K)-1)/(K-1).
    """
    if want_K <= 1:
        return 1.0
    return 1.0 + (want_K - 1) * (E_at_K - 1.0) / (K - 1)


def _selftest() -> None:
    # --- the invariant -----------------------------------------------------
    for f, n in ((ratio_spec, 1), (ratio_batch, 1)):
        lo, hi = f(4.0, n)
        assert 0.999 <= lo <= 1.002 and 0.999 <= hi <= 1.002, (f, lo, hi)

    # --- the base reconciles to the measured constant within 0.1% ----------
    base = pass_mb_spec(4.0, 1)
    res = (base - MEASURED_BASE_MB) / MEASURED_BASE_MB
    assert abs(res) < 1e-3, (base, res)

    # --- the LEGACY bracket must CONTAIN both published M=4 figures --------
    # The document says 2.11-2.41 in one place and 1.98-2.56 in another.  They
    # are the two ends of the same bracket.
    llo, lhi = ratio_spec(U_CONSECUTIVE[4], 4, legacy=True)
    assert llo <= 1.99 and lhi >= 2.40, (llo, lhi)

    # --- the corrected bars this rebrief publishes -------------------------
    for K, want in ((2, (1.258, 1.392)), (3, (1.585, 1.863)), (4, (1.884, 2.302))):
        got = ratio_spec(U_CONSECUTIVE[K], K)
        assert abs(got[0] - want[0]) < 0.005 and abs(got[1] - want[1]) < 0.005, (K, got, want)

    # --- lm_head accounting ------------------------------------------------
    assert lm_head_evals_spec(1) == 1
    assert lm_head_evals_spec(2) == 1
    assert lm_head_evals_spec(4) == 3
    assert lm_head_evals_batch(4) == 4

    # --- the monotone bound reproduces the documented M=2 lower bounds -----
    assert abs(monotone_lower_bound(3.61, 4, 2) - 1.870) < 0.002
    assert abs(monotone_lower_bound(1.87, 4, 2) - 1.290) < 0.002


if __name__ == "__main__":
    _selftest()
    print("lkbytes self-test OK\n")
    print("  expert        %8.4f MB" % EXPERT_MB)
    print("  shortconv     %8.4f MB" % SHORTCONV_MB)
    print("  attention     %8.4f MB" % ATTN_MB)
    print("  dense         %8.4f MB" % DENSE_MB)
    print("  router        %8.4f MB   <- was missing from every prior model" % ROUTER_MB)
    print("  misc          %8.4f MB" % MISC_MB)
    print("  FIXED         %8.4f MB   (legacy was %.2f)" % (FIXED_MB, FIXED_MB_LEGACY))
    base = pass_mb_spec(4.0, 1)
    print("  model base    %8.4f MB vs measured %.2f -> residual %+.3f%%"
          % (base, MEASURED_BASE_MB, 100.0 * (base - MEASURED_BASE_MB) / MEASURED_BASE_MB))
    print()
    print("SPECULATION  (consecutive tokens; the seed's logits are already held)")
    print("  K  lmhead   U   pass MB   bar corrected      bar legacy     delta")
    for K in range(1, 9):
        U = U_CONSECUTIVE[K]
        c = ratio_spec(U, K)
        l = ratio_spec(U, K, legacy=True)
        dl = 100.0 * (c[0] / l[0] - 1.0)
        print("  %d    %d   %5.2f %9.2f   %.3f - %.3f   %.3f - %.3f   %+.1f%%"
              % (K, lm_head_evals_spec(K), U, pass_mb_spec(U, K),
                 c[0], c[1], l[0], l[1], dl))
    print()
    print("BATCHING     (independent sequences; every one needs its own logits)")
    print("  B  lmhead   U   pass MB   speedup (corrected)   MB/token")
    for B in range(1, 9):
        U = U_INDEPENDENT[B]
        g = ratio_batch(U, B)
        print("  %d    %d   %5.2f %9.2f   %.3f - %.3f x      %8.1f"
              % (B, lm_head_evals_batch(B), U, pass_mb_batch(U, B),
                 g[0], g[1], pass_mb_batch(U, B) / B))
