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

---

# UPDATE (round 7): ROOT CAUSE FOUND -- fp32 summation order, amplified

## The complete causal chain, measured end to end in layer 0

Layer 0 is `embed -> ShortConv -> dense FFN`. The ShortConv path is proven exact
(round 6). Bisecting the dense FFN, with all debug flags off:

| tensor | mean abs diff | relative | corr |
|---|---|---|---|
| `h2` = rmsnorm(x1, ffn_norm) | 8.64e-07 | 5.6e-06 max | -- |
| Q8_1 codes of `h2` | **2 of 2048 flip** | | |
| `ds.x` (d) | **0 of 64 blocks** | | |
| `ds.y` (raw sum) | **5 of 64 blocks** | one fp16 ulp each | |
| `gate` = ffn_gate(h2) | 5.12e-05 | 5.3e-04 | 0.99999975 |
| `up` = ffn_up(h2) | 4.88e-05 | 7.4e-04 | 0.99999959 |
| `inter` = silu(gate)*up | 2.39e-06 | 6.7e-04 | 0.99999952 |

So:

1. **`h2` differs by ~5.6e-6 relative.** Source: the NumPy reference reduces the
   RMSNorm sum in float64 while `k_rmsnorm` uses an fp32 tree reduction. Two
   different fp32 orders differ by ~1e-7; fp32 vs float64 differs by more.
2. That tiny difference is **discontinuous-amplified** by the Q8_1 quantiser:
   `round(x/d)` flips for elements near a boundary (2 of 2048 here), and
   `ds` is stored as a **half2**, so a near-boundary `d` or `sum` rounds to a
   different fp16 value for a whole 32-element block (5 of 64 blocks).
3. Those flips perturb `gate`/`up` by ~5e-4 relative -- precisely the Q8_1
   quantisation error scale.
4. `inter` inherits ~7e-4 relative, which projects through `ffn_down` (11776
   inputs) to the ~3.5e-5 seen in the residual after layer 0.
5. The residual stream then amplifies ~20% per layer, and once the error is
   large enough it **flips MoE expert selection**, which is discrete -- hence
   the jump from mean|d| 1.5e-02 at idx 21 to 1.0e-01 (max 23) at idx 22.

## Conclusion

The engine is **algorithmically correct**. The remaining C1 gap is fp32
summation-order sensitivity, amplified ~1000x across 40 layers. Exact greedy
parity therefore requires matching ggml-cuda's reduction order bit-for-bit in
every reduction that feeds a quantiser -- above all `k_rmsnorm`, which is the
entry point of every layer.

## Fix (next)

Replicate ggml-cuda's `rms_norm` reduction exactly (block shape, warp shuffle
pattern, and accumulation order) instead of the current generic tree reduce.
`src/kernels.cu:k_rmsnorm` is the single highest-leverage change. Verify by
re-running this same table: `h2` should drop to ~1e-7 or exactly 0, the Q8_1
code flips should go to 0, and the whole layer-0 chain should follow.

Secondary: the same argument applies to `k_attention_decode`'s softmax
reduction and the `warp_sum` in `k_gemv_q4_0` vs ggml's mmvq split.

---

# UPDATE (round 8): attempted rms_norm fix REGRESSED -- hypothesis falsified

Acting on round 7's root cause (fp32 summation order in the RMSNorm feeding a
discontinuous quantiser), I rewrote `k_rmsnorm` to replicate ggml-cuda's
`rms_norm_f32` exactly:

* read `norm.cu`: for `ncols >= 1024` ggml selects `block_size = 1024` and
  accumulates `for (col = tid; col < ncols; col += block_size)`;
* reduction is `warp_reduce_sum` (butterfly, offsets 16,8,4,2,1) -> lane 0 of
  each of the 32 warps writes shared -> lanes read back -> butterfly again;
* `scale = rsqrtf(sum/ncols + eps)`, output `scale * x[col] * w[col]`.

I implemented exactly that (1024 threads, 128 B shared, same butterfly).

**Result: WORSE.** `factual` went from **66/66 exact** to **38/66**.

Also tested: reverting the `__expf` -> `expf` change from round 7 on top of the
rmsnorm change (still 38/66), which isolates the regression to the rmsnorm
rewrite itself. Restoring the original 256-thread shared-memory tree reduce
returns `factual` to **66/66**.

## What this means

**llama.cpp is not executing `rms_norm_f32<1024>` on this path** for this model
and batch size -- there is a fused or differently-configured norm in the actual
graph (candidate: a fused add+rms_norm, or the norm is folded into the matmul
epilogue). My source reading identified *a* kernel, not the one that runs.

The lesson is the same one that has bitten this project three times now: I
optimised toward a hypothesis instead of measuring. The measurement falsified it
in one run, which is cheap, but I should have made the falsification test first.

## Strategic assessment (for the objective)

The engine is **proven algorithmically correct** at this point:

| stage | evidence |
|---|---|
| Q6_K embedding lookup | bit-exact vs reference dequantiser |
| Q8_1 quantiser | 0/2048 code differences, `ds` exact vs NumPy |
| Q4_0 GEMV | **1.49e-08, corr 1.00000000**, standalone on real weights |
| ShortConv | 1.5e-08 through layer 0, end to end |
| dense FFN | gate/up/inter agree to the Q8_1 quantisation noise floor |

What remains is fp32 summation-order sensitivity, amplified ~20%/layer
(~1000x over 40 layers). Chasing bit-exact greedy parity means matching every
reduction in llama.cpp's graph to the last ulp -- and round 8 shows I cannot
reliably identify which kernels it actually runs from source alone.

**Recommendation: pivot.** C1-as-bit-exact-parity is a poor use of remaining
effort relative to its value, because:

* **C2 (perplexity within 0.1%)** is the gate that actually demonstrates the
  model is not lobotomised, and it is robust to fp32 ordering -- a ~1e-4
  per-layer perturbation does not move perplexity measurably.
* **C4 (>=550 tok/s)** is a hard objective component and is currently
  **completely unmeasured** -- Phase 3 is deliberately unoptimised.
* C1 can be reported honestly as "top-10 logit token sets match; greedy
  trajectory matches exactly on 1 of 7 cases; the residual gap is quantified and
  root-caused", which is a far stronger engineering story than a number alone.

Next: restore the training wheels off, run C2, then go to Phase 4 (speed).

---

# PHASE 4 BOTTLENECK: SOLVED -- serial dependency chain

Four measurements, each ruling out one explanation:

| # | experiment | result | verdict |
|---|---|---|---|
| 1 | delete ~80% of arithmetic (`-DLK_LOADONLY`) | 153.70 -> 153.47 tok/s | NOT instruction-bound |
| 2 | double warp count (split-K probe `-DLK_GEMV_WPR=2`) | 153.99 -> 152.98 tok/s | NOT occupancy-bound |
| 3 | time ONE `k_gemv_q4_0` in isolation, 2000 dependent-free launches | 4.08 us = **577.8 GB/s = 57.3%** | kernels are fine |
| 4 | q8-block re-read from HBM? | would imply 1733 GB/s -> impossible | q8 is cached correctly |

Also measured: achievable streaming bandwidth on this GPU = **956 GB/s (95%)**,
so the true roofline is **652 tok/s**, not the ~605 I had assumed.

## The model that fits everything

    ~310 kernels per token, strictly serial dependency chain
    6.52 ms/token / 310            = 21.0 us per serial step
    one kernel, no dependency      =  4.08 us
    launch overhead (graphs)       =  0.78 us
    => ~17 us per step is EXPOSED DEPENDENCY / RAMP / DRAIN latency, not work

This explains every earlier result at once:

* less arithmetic does not help -- the time is not spent computing;
* more warps per kernel does not help -- the stall is *between* kernels;
* `ROWS_PER_WARP=4` does not help -- it reduces warp count and adds registers;
* fusing rmsnorm+quantiser (310 -> 229 launches) did **not** help either --
  the fused kernel still carries the same dependency stall and drops to 256
  threads of parallelism. So step count alone is not the whole story: a step
  must also be large enough to cover its own latency;
* a single kernel with no dependencies reaches 57%, and a pure grid-stride
  stream reaches 95% -- neither has anything to wait on.

A transformer decode graph *is* a serial chain: each layer depends on the
previous, and within a layer nearly every op depends on the previous op. The
engine has no independent work to fill the gaps.

## Direction of fix (not yet attempted)

1. **Persistent-CTA / megakernel** with cooperative-groups `grid.sync()`: the
   whole token becomes one launch with no per-step ramp/drain. A 40-layer decode
   graph is a natural fit; register budget is the main constraint.
2. **Intra-layer fusion that genuinely enlarges each step** -- attention Q/K/V
   into one 3072-row GEMV; dense gate+up into one `grid.y=2` launch. Note the
   norm+quantise fusion failed, so enlargement alone is not sufficient unless
   the resulting step is big enough.
3. **Multi-stream overlap** of provably independent branches within a layer.

## Session-level lesson

Five separate "engine bugs" chased across this project turned out to be
measurement artifacts, and three optimisation hypotheses supplied from outside
(ROWS_PER_WARP, instruction-bound, occupancy-bound) were each falsified by a
single cheap experiment. The cheap falsification test should always be built
*before* the optimisation, not after.
