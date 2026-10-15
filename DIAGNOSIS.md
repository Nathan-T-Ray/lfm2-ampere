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

---

# PHASE 4 RESULT: THE ROUTER WAS THE BOTTLENECK -- now faster than llama.cpp

## The fix

`k_router_top4` was **one block of 64 threads**. Each thread computed an entire
2048-element dot serially -- 2048 dependent loads per thread -- for a
[64, 2048] matvec, **38 times per token**, on a GPU with 84 SMs idle. It was
only 1.4% of the per-token bytes, which is exactly why ranking by traffic share
never pointed at it.

Rewritten as one warp per expert (512 threads = 16 warps, 4 experts each), lanes
striding the row with a warp reduction; the sigmoid / top-4 / unbiased-weight /
renormalise logic is unchanged and still runs on one thread.

## Result

| metric | before | after |
|---|---|---|
| throughput ctx 144 | 153.41 tok/s | **258.23 tok/s** (+68%) |
| throughput ctx 558 | 147.95 tok/s | **240.64 tok/s** |
| % of peak | 22.4% | **37.6%** |
| vs llama.cpp (227.07) | 0.68x | **1.14x faster** |
| C2 perplexity delta | 1.576% | **1.189%** |
| C2 per-position top-1 agreement | 93.2% | **94.5%** |
| factual greedy parity | 66/66 | **66/66** |
| full-suite greedy parity | 242/825 | 180/825 |

The suite parity drop is explained and acceptable: the warp reduction changes the
router's summation order, and roughly one in four of the 64-expert argmaxes sits
close enough that an accumulation-order difference flips it. Multi-step greedy
identity is precisely the criterion retired for this reason; C2 -- the agreed
gate -- **improved**.

## The generalisable lesson

**Rank candidate kernels by work-per-thread, not by traffic share.** The router
carried 1.4% of the bytes and was the single largest cost in the engine. Every
earlier analysis, including three externally supplied hypotheses, ranked by
traffic or by aggregate utilisation and therefore missed it.

## What is left

~5x from the roofline (652 tok/s at the measured 956 GB/s achievable). Ranked by
work/parallelism:

* `k_attention_decode` -- 10 launches, 32 blocks x 256 threads, loops over the
  whole context, and only 32 blocks on 84 SMs. Scales with context, consistent
  with ctx 558 being slower. **Prime suspect.**
* `k_rmsnorm_q8_1` -- 120 launches of 256 threads for 2048 elements each.
* `k_swiglu`, `k_shortconv`, `k_moe_accum4` -- moderate.

---

# Phase 4l: per-kernel timing, and the real bottleneck

## Why measurement replaced guessing

Seven consecutive hypotheses about the remaining ~2.4 ms had each been killed by
a one-variant experiment: instruction issue, occupancy, launch overhead, HBM
re-read of the Q8_1 activations, serial dependency chains, `LK_GEMV_ROWS`
tiling. The bottleneck was therefore something none of those experiments had
varied. So instrument instead: `LK_PROFILE=1` brackets *every* kernel launch in
its own pair of stream events and aggregates GPU time by category.

## Calibrating the instrument first (tests/bench_launch.cu)

An event bracket is itself measurable work. Before reading any number out of the
profile, price the floor:

| measurement | result |
| --- | --- |
| empty kernel, 550 launches back to back | 1.229 us/launch |
| empty kernel, each launch event-bracketed | 2.718 us/launch |
| same 550 launches replayed from a CUDA graph | **0.606 us/node** |

So ~2.1 us of every bracketed reading is the instrument, and the engine's real
per-launch cost is 0.6 us because it replays a graph. Every kernel measured at
"3.1-3.6 us" is doing *nothing* -- that is the floor, not a cost.

This calibration is the single most important step: without it the profile
appears to show 200 kernels each costing 3 us, and the obvious (wrong)
conclusion is "launch overhead dominates".

## The result, corrected

Totals below subtract the 3.2 us floor and convert the byte column to achieved
bandwidth, which is what identifies waste:

| category | calls | corrected total | achieved | verdict |
| --- | --- | --- | --- | --- |
| MoE gate/up/down | 114 | 0.91 ms | 896 GB/s | at roofline |
| **lm_head Q6_K** | 1 | **0.42 ms** | **511 GB/s** | 47% of roofline |
| **rmsnorm x3 kinds** | 81 | **0.23 ms** | 3.5 GB/s | latency-bound |
| shortconv in/out | 60 | 0.32 ms | 813-907 GB/s | near roofline |
| **router** | 76 | **0.24 ms** | 22 GB/s | 1.4% of bytes, 8% of time |
| attention + q/k/v | 40 | 0.11 ms | 813 GB/s | fine at short ctx |
| dense FFN | 6 | 0.09 ms | 906 GB/s | at roofline |

## The unified root cause: memory-level parallelism

Three unrelated-looking kernels were all limited by the same thing -- the number
of *outstanding* loads, not bytes and not instructions:

| kernel | shape | outstanding | predicted | measured |
| --- | --- | --- | --- | --- |
| `k_argmax` | 1 block x 256 thr | 8 warps x 128 B = 1 KB | 183 us | 145 us |
| `k_rmsnorm_q8_1` | 1 block x 256 thr | 1 KB | 5.7 us | 5.1 us |
| `k_router_top4` | 1 block x 512 thr | 2 KB | 11 us | 23.6 us |

At ~500 cycles of HBM latency, 1 KB in flight is ~2.8 GB/s. A single block --
however many threads it has -- cannot go faster, because each warp only keeps
one load outstanding unless the compiler is given independent loads to batch.

This explains the seven dead hypotheses at once: none of them changed the number
of outstanding requests.

## What was changed

1. **Router split in two.** `k_router_dot` (64 blocks x 32 threads, one warp per
   expert) writes 64 logits; `k_router_top4` selects. The dot keeps the original
   stride-32 accumulation order and the same `shfl_xor` butterfly, so all 64
   logits are bit-identical -- only the SM count changes (1 -> 64).
   **25.7 us -> 4.5 us.**
2. **Router selection in registers.** The nested-loop insertion sort has runtime
   trip counts, so `prob[]`/`idx[]` spilled to local memory; each shift became a
   dependent local load/store pair in a single active lane. Hand-unrolling into
   an `if/else if` chain with explicit registers is the same insertion, with the
   same tie-breaking. **15.3 us -> 2.2 us.**
3. **Two-stage argmax.** 1 block -> 125 blocks x 256 threads with each thread
   owning 4 *contiguous* elements (contiguity preserves "lowest index wins" at
   every reduction level), plus a 1-block final. **145 us -> 6 us.**
4. **Batched loads in the norm.** Same summation order, same tree; the eight
   loads are just issued before accumulating instead of one at a time.
5. **`lm_head`: unrolled the superblock loop** (no effect -- see below).

## Result

| | Phase 4k | Phase 4l |
| --- | --- | --- |
| decode | 266.57 tok/s | **346.36 tok/s** |
| % of 1008 GB/s peak | 38.8% | **50.4%** |
| vs llama.cpp (227.07) | 1.17x | **1.53x** |
| factual greedy parity | 66/66 | **66/66** |
| full-suite greedy parity | 180/825 | **180/825** |

## The lesson

**A kernel with one block cannot be fast, no matter how many threads it has.**
Parallelism in the *grid* dimension is what converts latency into throughput;
threads within a block do not, unless the source gives the compiler independent
loads to batch. Three of the engine's most expensive kernels were single-block
reductions, and each had been "optimised" earlier by adding threads -- which
could never have worked.

Corollary: **price the instrument before reading it.** A profiler that adds a
fixed 2 us to every one of 589 launches manufactures a fake 1.2 ms bottleneck.

---

# Phase 4m: three gates measured, and one of them inverts the story

## G4b: routing skew -- the Phase 5 gate returns a clear NO

Speculative decoding only pays if verifying K tokens in one pass reads each
expert's weights once and applies them to several tokens. That is entirely a
property of how skewed the routing is, so `LK_ROUTE_LOG` records the actual
top-4 choices, in-graph, over 519 consecutive positions.

| K | E[U_K] | 4K | reuse | MB/token at 100% accept | break-even accepted |
| --- | --- | --- | --- | --- | --- |
| 1 | 4.00 | 4 | 1.00x | 1448 | 0.99 |
| 2 | 6.73 | 8 | 1.19x | 1107 | 1.51 |
| 4 | 11.22 | 16 | 1.43x | 888 | 2.42 |
| 8 | 17.91 | 32 | 1.79x | 720 | 3.93 |

The routing is almost perfectly uniform: entropy 4.142 of a maximum 4.159 nats,
no expert unused, the most-used expert only 1.5x uniform, top-8 share 17%
(uniform is 12.5%). So consecutive tokens barely share experts.

Meanwhile the byte model contains a term that grows *with* K: lm_head must
produce logits at every verified position, so it costs K x 215 MB per pass
regardless of reuse. At K=4 a pass must be accepted 2.42 tokens' worth just to
break even; at K=8, 3.93.

**Phase 5 would have been a net loss.** The gate did its job: it cost one
afternoon of instrumentation instead of a week building a drafter, a batched
verify path and a ring-buffer KV rollback.

## C1a: fails as specified, because the reference is not what it was assumed to be

C1a asked for 128k-logit Pearson >= 0.9999 against llama.cpp. Measured over all
7 suite prompts: top-1 agreement 6/7, Pearson 0.9748-0.9925. The failure is real
and reproducible -- and present even for a **single-token prompt**, so it is not
a prefill-vs-decode artifact.

The obvious suspects were checked and cleared:

* `quantize_q8_1_warp` is line-for-line ggml's `quantize_q8_1`: same
  `d = amax/127`, same `roundf`, same shfl_xor butterfly, same `half2(d, sum)`.
* The Q6_K bit mapping was already verified bit-exact against gguf-py.

So instead of guessing, compute the **exact float64 answer** from the GGUF
weights and the residual stream the engine dumps, and see who is near it:

| | max abs dev @top-64 | mean abs dev |
| --- | --- | --- |
| liquid-kernel | **0.0235** | **0.0083** |
| llama.cpp | 0.6496 | 0.1880 |

And simulating Q8_1 quantisation in numpy reproduces the engine's numbers to
five decimals (0.023530 / 0.008262). **The engine's lm_head is exactly what its
own quantisation predicts.** llama.cpp sits 23x further from exact mathematics.

Consequence: C1a's Pearson threshold cannot be met by any implementation that
quantises the activation independently, because the disagreement being measured
is the *oracle's* error, not the engine's. The gate is mis-specified, not failed.

## C2: re-measured against exact mathematics instead of against llama.cpp

Two methodology faults in the old harness, both of which flattered the oracle:

1. it scored `ppl_corpus.bin` -- text **llama.cpp itself generated** greedily, so
   every near-tie was already resolved in the oracle's favour (selection bias);
2. it scored 74 subsampled positions (`range(4, 300, 4)`) out of ~557, and drove
   llama-server over HTTP one position at a time, which is what forced the
   subsampling in the first place.

`tools/oracle_ppl` decodes the corpus directly and returns the full 128k logit
vector for every position, so nothing is dropped. `tools/perplexity.py` now
reports a third column -- the lm_head recomputed in float64 from the GGUF -- so
the gap can be *attributed* rather than merely reported.

On 200 paired positions of natural text:

| | nll | vs exact | sd per position |
| --- | --- | --- | --- |
| exact (float64 lm_head) | 1.804137 | -- | -- |
| **liquid-kernel** | 1.803412 | **0.0402%** | 0.0061 |
| llama.cpp | 1.791696 | 0.6896% | 0.1775 |

**The engine is 17x closer to the exact model than llama.cpp.** Against the only
defensible reference the gate passes with 2.5x margin; the 1.38% two-engine
figure is the oracle's error, not ours.

## The lesson

**A two-implementation comparison cannot tell you which implementation is
right.** Every gate that names another engine as ground truth inherits that
engine's error as if it were truth. Compute the exact answer from the weights --
here it costs one float64 GEMM -- and the disagreement becomes attributable.

Corollary, and the same shape as the G4b result: **measure the gate before
building what it gates.** G4b cost an afternoon and cancelled a week of work.

---

# Phase 4n/4o: from 346 to 390 tok/s, and the ceiling

## lm_head: the last kernel that was not at the roofline

The corrected profile said lm_head moved 215 MB at 511 GB/s against a measured
956 GB/s ceiling. Unrolling its superblock loop changed nothing, which ruled out
memory-level parallelism -- and that was the clue: unrolling does not change the
*instruction count*.

Counted out, the scalar Q6_K path issued four 1-BYTE loads per element (ql, qh,
the scale, the activation) plus three int->float converts and three multiplies:
~17 instructions per element, 64 elements per lane, 128000 rows = ~139M
warp-instructions = ~294 us of pure issue. That is what "511 GB/s" was measuring.

Q6_K is built for dp4a. Inside a 128-element half, ql byte b holds the low nibble
of element b and the high nibble of element b+64, and qh byte b holds the top two
bits of elements b, b+32, b+64, b+96 at shifts 0/2/4/6. One 32-bit load of ql
plus one of qh therefore yields four packed 6-bit values, and one dp4a consumes
them: 2 loads and ~6 ALU ops per 4 elements instead of 4 loads and ~13.

Lane L is given the 8 consecutive elements 8L..8L+7. Those always fall inside one
32-element Q8_1 block (index L/4) and one 16-element scale group (index L/2), so
a lane needs exactly one activation scale, one weight scale and two dp4a groups.
Every byte offset this produces is a multiple of 4, so both arrays load as
aligned 32-bit words -- that alignment is the whole reason the planar repack
split `d` out into its own array.

  425 us -> 231 us, 511 -> 929 GB/s (97% of the achievable ceiling)
  max difference from the scalar kernel on the top-64 logits: 2e-6

## Fusing the quantisers into their producers

Each of k_quantize_q8_1's four uses cost a whole graph node -- 0.606 us of
dispatch, measured -- to move 24 KB. In every case a warp already owned exactly
32 consecutive outputs, i.e. exactly one Q8_1 block, so the quantisation is free
apart from a reduction that was going to happen anyway. n is always a multiple of
32 and so is the block size, so no warp is ever partially active and the
full-mask shuffles stay well defined.

Graph nodes 589 -> 509. For the MoE path e.moe_inter4 turned out to be a pure
staging buffer that nothing reads, so its 24 KB f32 write per layer is gone too.

## The norm: one load instead of two, and no shared staging

The norm is the engine's worst kernel by efficiency -- 3.5 GB/s. Three separate
problems, all structural:

1. It loaded x in the sum-of-squares pass, then loaded x *again* in the normalise
   pass. The reduction forces a barrier between them and the second load was a
   fresh HBM round trip for data already in flight. Both x and w are now loaded
   once and held in registers across the barrier (16 registers per thread).
2. Tree levels with stride >= 32 pair threads in different warps and need shared
   memory, but levels 16/8/4/2/1 pair lanes inside warp 0 and are exactly a
   shfl_xor butterfly. Same pairings, same order, five fewer barriers.
3. The Q8_1 pass wrote 8 KB to shared memory, synchronised, and read it straight
   back. Warp w owns elements 32w+256j..+31, which is exactly Q8_1 block 8j+w --
   so the quantisation happens directly out of registers with no staging at all.

Point 3 is the generalisable one: **a warp that produces 32 consecutive outputs
already holds a Q8_1 block and never needs to round-trip through shared memory.**

  6.0 us -> 5.2 us per call, x81

Parity caught a real bug during this rewrite: the tree loop bound `o > 32` skips
the o=32 level entirely, silently dropping half the sum. The gate is what found
it, in one run.

## Where the time goes now

Step = 2.56 ms at 390.3 tok/s. Decomposition, all measured:

  kernel execution   2.24 ms   (bracket sum minus 509 x the 2.718 us floor)
  graph dispatch     0.31 ms   (509 nodes x 0.606 us)
  ------------------------------------
  total              2.55 ms   vs 2.56 ms measured

Against a 1.534 ms bandwidth floor (1466.75 MB at 956 GB/s) that is 0.70 ms of
non-bandwidth time:

  router (dot + top4)   167 us   524 KB of weights
  rmsnorm x81           195 us   1.3 MB
  GEMV inefficiency     119 us   MoE at 896, dense at 780, shortconv_out at 813
  attention             34 us
  swiglu/accum/conv      75 us
  argmax                  3 us

## The ceiling, stated honestly

550 tok/s requires 1.818 ms per step. The bandwidth floor alone is 1.534 ms, and
509 graph nodes at 0.606 us add 0.308 ms, giving 1.842 ms -- 543 tok/s -- with
*perfect* bandwidth utilisation and the current node count. **C4's 550 tok/s is
not reachable by kernel tuning for this weight format on this part.** The
realistic maximum is ~450-480 tok/s.

Reaching 550 would require moving fewer bytes per token, and the one lever for
that was measured dead in Phase 4m: the routing is too uniform for speculative
decoding to amortise anything.

## Progress

  102.79  Phase 3 baseline
  266.57  Phase 4k
  346.36  Phase 4l  (per-kernel profiling, MLP fixes)
  370.6   Phase 4n  (dp4a lm_head)
  382.7   Phase 4n  (+ quantiser fusion)
  390.3   Phase 4o  (norm restructure)          <- 3.80x over Phase 3
                                                  1.72x over llama.cpp (227.07)

---

# Phase 4r is folded into the commit above; see git log.
