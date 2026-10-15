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

---

# UPDATE (round 5): everything out_proj READS is verified; the output is still wrong

I forced `add_residual=0` for layer 0's out_proj (`LK_NO_RESIDUAL=1`) so the raw
projection could be compared in isolation against the NumPy reference.

    rms(cuda) = 0.015783   rms(ref) = 0.015748   ratio = 1.0022   <- magnitude OK
    max|d|    = 5.54e-02   mean|d|  = 6.0e-03    corr   = 0.8719   <- elements WRONG
    rows with |d| > 1e-4 : 2026 / 2048           <- essentially ALL rows

So the magnitude is right but the elements are not, uniformly across all rows.

Per-row mean error is ~6e-3 against an output rms of 1.57e-2, i.e. ~38%
relative. Yet the SAME kernel function is exact to 7.1e-08 for `in_proj` in the
same layer, with the same bpr=64.

## Everything the out_proj GEMV consumes is verified identical

| input | check | result |
|---|---|---|
| weight planes | manifest geometry + generated tensor table | qs 2,097,152 B and d 262,144 B at contiguous offsets; bpr=64, rows=2048 correct |
| Q8_1 codes of the conv output | CUDA `LK_DUMP_L0Q` vs NumPy | **0 / 2048 differ** |
| `ds.x` (d) | same | **0.0** |
| `ds.y` (raw sum) | same | 3.8e-06; **1 of 64 blocks**, one fp16 ulp (butterfly vs np.sum order) |
| conv output `y` (GEMV input) | `LK_DUMP_L0B` | 1.49e-08 |

## Where that leaves it

Given identical operands, an identical dot function, and an exact result for
`in_proj`, the difference must be in the *launch* or the *output buffer*, not the
arithmetic. Remaining suspects, narrowly:

1. `out_rows=2048` vs `6144`: both give an exact warp count
   (grid_for(r) = (r+7)/8 blocks x 8 warps), so this should not matter -- but it
   is the only other difference.
2. The output buffer `e.x` specifically (in_proj writes `e.y`).
3. Some other kernel in layer 0 touching `e.x` that I have not accounted for.

## Next test (concrete, cheap)

Launch the GEMV twice in layer 0 -- once with the out_proj weights into a fresh
scratch buffer, once into `e.x` -- and compare the two. If they differ, it is a
buffer/aliasing problem. If they agree, compare that buffer against NumPy, which
would instead implicate the weight plane pointers. Also worth dumping
`PQS(e,l.sc_out)[0..3]` and `PD(e,l.sc_out)[0..3]` from the device and checking
them against the manifest offsets.

---

# UPDATE (round 6): the GEMV is PROVEN exact; round 5's conclusion was a harness artifact

## Correction to round 5

Round 5 concluded that `k_gemv_q4_0`'s out_proj path was broken. **That was
wrong, and the error was mine.** `LK_NO_RESIDUAL` disabled only the *ShortConv*
residual, not the FFN's; layer 0 then still ran its dense FFN, so the dumped
`e.x` was `proj + ffn_out`, and I compared it against a reference `proj`. The
"38% per-row error with correct rms and 0.87 correlation" is exactly what
`proj + ffn_out` vs `proj` looks like.

Lesson applied from here on: verify a value at the point it is produced, with
the mutation flags off, before drawing conclusions from a diff.

## What is now PROVEN (not inferred)

`tests/test_gemv.cu` is a standalone isolation harness: it loads the real
`blk.0.shortconv.out_proj.weight` planes from the pack, quantises a fixed
activation, runs `k_gemv_q4_0`, and dumps the output. No engine, no layer loop.

    standalone vs NumPy : max|d| = 1.49e-08   corr = 1.00000000
    row 1572 = 0.0163759  (reference 0.0163759)

and inside the engine, dumped *immediately* after the same GEMV with a stream
sync (`LK_DUMP_L0X`):

    engine vs standalone : max|d| = 3.73e-07   corr = 1.00000000
    row 1572 = 0.0163761

So `k_gemv_q4_0`, the Q8_1 quantiser, the planar weight planes and the
ShortConv path are all correct **end to end within layer 0**. That eliminates a
large part of the search space and is a permanent regression tool.

## Where the divergence actually is

The clean per-layer table (no debug flags) is unchanged from round 3, so the
real divergence remains:

    idx 1  (after layer 0)  ~1e-4
    idx 21                 1.5e-02   max 5.8e-01
    idx 22                 1.0e-01   max 2.3e+01   rms 1.4387 vs ref 0.5824
    idx 40 (final)         1.1e-01   max 4.2e+00   rms 0.3608 vs ref 0.2505

The CUDA residual *explodes* to rms 1.44 at layer 22 (60x the embedding rms)
where the reference reaches only 0.58. Since layer 0's ShortConv path is now
proven exact, the fault is in the **channel mixer (dense FFN / MoE)** or the
**attention** path -- layer 22 is an attention layer (il%4==2).

## Next

Bisect layer 0's *dense FFN* the same way the ShortConv was bisected: dump
`ffn_norm` output, its Q8_1 codes, gate/up, the SwiGLU intermediate, its Q8_1
codes, and the down output, each compared against `ref_forward.py`. Layer 0 is
dense-FFN and comes right after the now-verified ShortConv, so it is the next
unverified stage in the chain.
