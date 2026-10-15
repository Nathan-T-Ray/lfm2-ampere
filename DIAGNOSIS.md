# C1 parity diagnosis (state after Phase 3)

## Measured

`tools/parity.py` over the 7-case suite: **242/825 tokens agree (29.33%)**.
Per case: factual EXACT (66/66), prose 89/128, code 62/128, longctx 20/128,
math 5/128, reasoning 0/128, agentic 0/119.

The engine is *fundamentally* correct: `factual` matches llama.cpp bit-for-bit
for its entire 66-token trajectory, and for every case the top-10 logit token
SET matches the oracle, usually with only adjacent rank swaps. Divergences are
argmax flips among tokens close in probability.

## Where the error is NOT

Ruled out by direct measurement, not assumption:

| Suspect | Evidence |
|---|---|
| Hyperparameters | llama.cpp reports n_rot=64, f_norm_rms_eps=1e-5, rope type 2 (NEOX), freq_base=1e7, n_ff=11776, n_ff_exp=1536, expert_gating_func=sigmoid -- all match |
| Q6_K embedding lookup | **bit-exact** (max abs diff 0.0) vs the reference dequantiser, 6 tokens x 2048 |
| Q4_0 / Q6_K x Q8_1 dots | verified vs exact integer reference and vs true fp32 dot |
| RMSNorm / NEOX RoPE / ShortConv / router | verified vs double-precision host references |
| Attention | a 1-token prompt has trivial attention (softmax over one element = 1, output = v), yet we still differ |
| Conv state | at token 0 the state is zero, so conv = bx*w[2]; we still differ |

## Where it IS

Log-probability deltas vs the oracle, 1-token prompt [597]:

    token   ours      oracle    delta
     25    -3.4099   -3.4338   +0.024
     27    -4.2345   -4.2665   +0.032
     26    -4.3895   -4.3035   -0.086
    597    -4.6484   -4.2872   -0.361
    785    -4.7223   -5.0601   +0.338

For a 5-token prompt the deltas are +0.034 / -0.072 / +0.048 / -0.274 / -0.166
/ +0.245. So the error is present at a SINGLE token, in the per-layer math, and
is ~0.02-0.36 nats -- far too large for fp32 accumulation order (~1e-5) but too
small to be a structural logic error.

Estimate: a 0.36 nat logit shift needs a hidden-state perturbation of order 0.1
in a stream whose RMS is order 1. Something in the layer computation is
systematically off in some component, not merely rounded differently.

## Hypotheses, cheapest first

1. **Residual/activation dtype.** llama.cpp may keep the residual stream in f16
   for part of this graph, or quantise the activation differently (Q8_1 vs f16)
   for some matmuls. Cheap test: round `x` to f16 after each layer and see
   whether parity improves.
2. **Activation quantisation convention on the mmvq path.** We match ggml's CUDA
   `quantize_q8_1` (ds = (amax/127, RAW sum)) and the Q4_0 correction
   `d4*(d8*sumi - 8*sum_raw)`. Worth confirming which kernel this build actually
   selects at runtime for batch-1 (mmvq vs mmq vs a dequant path).
3. **Per-layer bisect against an independent reference.** Write a NumPy forward
   pass over the SAME pack data and compare layer by layer. This is the rigorous
   route: it separates "our CUDA does not do what we think" from "we
   misunderstand llama.cpp semantics".

## Working tools

    ./build/engine topk <ids...>     # normalised log-probs for the top 20
    ./build/engine gen <ids...>      # greedy tokens (LK_NGEN=n)
    python3 tools/parity.py [case]   # C1 comparison vs ref/oracle_ref.json
    LK_DEBUG_STEPS=1 ./build/engine ...   # synchronise + name every faulting stage

Note: `topk` prints ranks 11-20 unsorted -- `std::partial_sort` only orders the
first 10. Only ranks 1-10 are meaningful as printed.
