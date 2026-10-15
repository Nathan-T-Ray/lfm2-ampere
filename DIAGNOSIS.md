# C1 parity diagnosis (Phase 3)

## Measured

`tools/parity.py` over the 7-case suite: **242/825 tokens agree (29.33%)**.
Per case: factual EXACT (66/66), prose 89/128, code 62/128, longctx 20/128,
math 5/128, reasoning 0/128, agentic 0/119.

Log-probability deltas vs the oracle are **0.02-0.9 nats** and are **flat in
prompt length** (len 1..5: max |delta| 0.36 / 0.88 / 0.71 / 0.63 / 0.27), so the
error does NOT accumulate through attention or the recurrent state. It is a
per-layer numerical difference.

## Ruled out by direct measurement

| Suspect | Evidence |
|---|---|
| Hyperparameters | llama.cpp reports n_rot=64, f_norm_rms_eps=1e-5, rope type 2 (NEOX), freq_base=1e7, n_ff=11776, n_ff_exp=1536, expert_gating_func=sigmoid -- all match |
| KV cache | llama.cpp uses **K/V f16** (80 MiB each) -- matches ours |
| Recurrent state | llama.cpp RS buffer 1.88 MiB = 4 seqs x 30 layers x 4096 f32 -- matches our layout |
| Q6_K embedding lookup | **bit-exact** (max abs diff 0.0) |
| Q4_0 / Q6_K x Q8_1 dots | verified vs exact integer reference and vs true fp32 dot |
| RMSNorm / NEOX RoPE / ShortConv / router | verified vs double-precision host references |
| Attention, conv state | a 1-token prompt has trivial attention and zero state, yet we still differ |

## The key structural finding: the model AMPLIFIES perturbations

`tools/ref_forward.py` is an independent NumPy forward pass over the same pack.
Comparing it against the CUDA engine's residual stream (dumped with
`LK_DUMP_LAYERS`), layer by layer, for a 1-token prompt:

    layer   mean|d|    max|d|    rms(cuda)  rms(ref)
      1     1.0e-04    7.2e-04     0.0072     0.0073
      9     7.1e-04    3.1e-03     0.0163     0.0164
     10     1.0e-03    6.9e-02     0.0638     0.0664   <- divergence starts
     21     1.2e-02    5.2e-01     0.0713     0.0729
     22     9.4e-02    2.2e+01     1.4387     0.6208   <- explosion
     40     9.7e-02    3.7e+00     0.3608     0.2612

Two conclusions:

1. **The residual stream amplifies perturbations by roughly 20% per layer.**
   Over 40 layers that is ~1000x. This is why a per-layer difference of only
   ~1e-4 becomes a ~0.3 nat logit shift. It also means **exact C1 parity
   requires matching llama.cpp's arithmetic to ~1e-6 per layer** -- matching
   the algorithm is not sufficient.
2. **CUDA and an independent implementation of the same stated semantics
   already diverge from layer 10.** So there is a genuine implementation
   difference to find, not merely an unavoidable rounding gap. This is now
   bisectable: layer 0-9 agree to 1e-4, layer 10 is the first disagreement.

Note layer 10 is the 3rd attention layer (attention at il%4==2: 2,6,10,...).
Layers 2 and 6 agree well, so whatever differs is not "attention is broken"
but something that first bites at layer 10.

## Next step (precise)

Instrument layer 10: dump the normalised activation `h`, the Q8_1 blocks, the
raw QKV GEMV outputs, and the post-QK-norm/RoPE Q and K from the CUDA engine,
and compare each against `tools/ref_forward.py`. That isolates whether the
divergence is in the norm, the quantiser, the GEMV, or the attention inputs.

## Working tools

    ./build/engine topk <ids...>       # normalised log-probs, top 20
    ./build/engine gen <ids...>        # greedy tokens (LK_NGEN=n)
    python3 tools/parity.py [case]     # C1 vs ref/oracle_ref.json
    python3 tools/ref_forward.py 597   # NumPy forward + per-layer diff table
    LK_DUMP_LAYERS=/tmp/x.bin ./build/engine topk <ids...>   # residual per layer
    LK_DEBUG_STEPS=1 ./build/engine ...   # sync + name every faulting stage

Caveats:
 * `topk` prints ranks 11-20 unsorted (`partial_sort` orders only the first 10).
 * `tools/ref_forward.py` scores attention as V, valid only for a 1-token prompt.

---

# UPDATE (round 4): bisected to layer 0's out_proj + residual step

## What now matches EXACTLY (1-token prompt, token 597, layer 0)

Dumped via `LK_DUMP_L0` / `LK_DUMP_L0B` and compared against `tools/ref_forward.py`:

| tensor | max abs diff | note |
|---|---|---|
| `h` = rmsnorm(x, attn_norm) | 5.96e-08 | fp32 roundoff |
| Q8_1 codes of `h` | **0 / 2048 differ** | quantiser is exact |
| `ds.x` (d) and `ds.y` (sum) | **0.0** | fp16 rounding matches |
| `bcx` = in_proj(h)  [6144] | 2.38e-07 (rel 7.1e-08) | GEMV faithful |
| conv output `y` [2048] | 1.49e-08 | ShortConv faithful |
| Q8_1 codes of `y` | **0 / 2048 differ** | |

So the Q8_1 quantiser, the Q4_0 GEMV, the norm and the ShortConv are all
correct. A previous apparent 1.3e-4 error in `bcx` was a bug in the *reference*
(ggml stores `ds` as a **half2**, so both `d` and the raw sum are fp16-rounded;
using fp32 in the reference manufactured the discrepancy).

## Where it breaks

The residual stream AFTER layer 0 (dumped as idx 1) disagrees badly:

    rms(cuda) = 0.00723   rms(ref) = 0.01342   max|d| = 7.6e-02

Inverting the residual add (`proj_cuda = cuda_x_after - x_before`) gives a
projection whose rms is 0.02111 vs the reference 0.01575 -- **34% larger** --
correlating only 0.9075 with the reference.

Crucially, `y` (that GEMV's input) matches to 1.5e-8 and its Q8_1 codes are
identical, so the dot product *should* agree to ~1e-7.

The ONLY structural difference between this GEMV and the `in_proj` GEMV (which
matched to 7e-8) is:

    out_proj : out_rows=2048, bpr=64, add_residual=1, reads+writes e.x
    in_proj  : out_rows=6144, bpr=64, add_residual=0, writes e.y

=> **the fault is in the residual path of `k_gemv_q4_0`**, i.e.

    if (lane == 0) y[warp] = add_residual ? (y[warp] + acc) : acc;

which is the last unverified line in the decode step. Suspects, in order:
  1. `y` is declared `__restrict__` while being both read and written.
  2. Some other kernel in layer 0 touching `e.x` (none should).
  3. Grid/row mapping differing between the two launches (both use
     `grid_for(rows)` = (rows+7)/8 blocks of 256, 8 warps each).

## Also established this round

The model **amplifies perturbations ~20% per layer** (1e-4 at layer 1 grows to
9.4e-2 / max 22 by layer 22, ~1000x over 40 layers). So exact C1 parity requires
matching llama.cpp to ~1e-6 per layer; getting the algorithm right is not
sufficient. This is why a single unfixed residual bug destroys parity
everywhere.
