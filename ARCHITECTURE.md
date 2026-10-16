# liquid-kernel — architecture, profiler, decisions and journey

A hand-written sm_86 CUDA inference engine for **LFM2.5-24B-A2B-QAD-Q4_0** on one
RTX 3090 Ti. Single stream, greedy decode. Every number in this file was measured
on the machine, and this document is written to be **audited** — see §12.

Written at `f8155b9`, and every figure in §0, §3 and §6 was **re-measured live**
while writing it, not copied forward from an earlier note. Where a figure could
not be re-measured it says so.

---

## 0. THE HEADLINE, RE-MEASURED

| | |
|---|---|
| decode, ctx 272, interleaved A/B, 5 rounds | **443.04 tok/s** (min 439.86, max 443.85) |
| the pre-Phase-4z build measured in the same session | 439.37 tok/s (spread 0.05%) |
| Phase 4z total gain | **+0.84%** |
| effective bandwidth | 650 GB/s = 64.5% of the 1008 GB/s spec, 68% of the 956 achievable |
| llama.cpp, same GGUF, same card, same session | 257.9 tok/s |
| speedup | **1.72x**, and 1.61–1.72x to 3.2k context |
| vs this engine's own Phase-3 baseline | **4.31x** (102.79 tok/s) |
| traffic | 1466.75 MB/token → bandwidth floor 1.534 ms = **652 tok/s** |
| CUDA graph | **343 nodes**, replayed per token |
| **measured batch-1 ceiling** | **488.9 tok/s** |
| batch-4 on the same one card | 731–832 tok/s (estimated from the same decomposition) |
| C2 accuracy, re-measured live | **0.0140%** from exact float64 against a 0.1% gate — **PASS**; llama.cpp is 0.2199% from the same reference |

Build reproducibility: `make engine` from the committed tree reproduces
`build/engine` **md5-identical** (`e3051b18b27b1fb08e0d1f288bed68a0`).

Re-measured gates, all PASS: G2 dots 8/8, G2 kernels, G6 attention, G7 normfuse,
G8 normscale. Bit-identity of the shipped build against the T1-only build: 7/7
cases identical.

---

## 1. THE SYSTEM UNDER AUDIT

### Hardware, measured not datasheet
| | |
|---|---|
| GPU | NVIDIA RTX 3090 Ti, GA102, sm_86 |
| SMs | 84 |
| **maxThreadsPerMultiProcessor** | **1536** — not 2048. GA102, not GH100. |
| L2 | 6144 KB |
| spec bandwidth | 1008 GB/s (21 Gbps x 384-bit) |
| achievable | 956 GB/s (dedicated `bandwidth_probe`) |
| power limit | 450 W (card maximum) |
| instrumentation available | CUDA events, occupancy APIs, ablations. **No nsys, no ncu, no nvprof.** |

`bench_shapes` prints the device line itself: `NVIDIA GeForce RTX 3090 Ti  SMs=84
maxThreadsPerSM=1536`, so the occupancy limit is confirmed by the tool rather than
assumed.

### The model
LFM2.5-24B-A2B-QAD-Q4_0 — 24B total, ~2B active per token. **QAD =
Quantization-Aware Distillation**, a retrained checkpoint whose weights do *not*
match public `LiquidAI/LFM2.5-24B-A2B` (embedding correlation **-0.036**), so the
public checkpoint is not a usable reference. The only valid external oracle is
llama.cpp on this exact GGUF.

Geometry, all confirmed against `llama.cpp src/models/lfm2.cpp`:

| | |
|---|---|
| hidden | 2048 |
| layers | 40 |
| heads / head_dim | 32 / 64 |
| KV heads | 8 (GQA) |
| vocab | 128000 |
| MoE | 64 experts, top-4, intermediate 1536 |
| dense intermediate | 11776 |
| eps | 1e-5 |
| RoPE | NEOX half-split, `n_rot = 64 = head_dim` (no partial rotation), theta 1e7, freq_scale 1, no YaRN |
| attention scale | exactly 0.125 |

**The hybrid architecture is the single most important fact about the workload:**

    40 layers
      30 ShortConv (LIV convolution)   -- layers where layer % 4 != 2
      10 GQA attention                 -- layers where layer % 4 == 2
    layers 0-1   dense SwiGLU FFN (11776 intermediate)
    layers 2-39  64-expert top-4 MoE

ShortConv exactly: `in_proj -> [B|C|x] -> bx = B*x -> causal UNFLIPPED 3-tap conv
-> y = C*conv_out -> out_proj -> residual`. **No non-linearity anywhere.** State =
the last two `B*x`, F32, `[tap, channel]` with the tap axis fastest.

Router exactly: **SIGMOID** gating (`expert_gating_func == 2`); top-4 chosen by
`sigmoid + exp_bias` where the bias affects **ranking only**; the weights come
from the **unbiased** sigmoid, renormalised with `clamp(sum, 6.103515625e-5, inf)`.
Routers are F32 and consume the **post-`ffn_norm`** hidden state in F32.

Also exact: Q and K each get a per-head RMSNorm(64) **before** RoPE, V gets
neither; SwiGLU is `down(silu(gate)*up)`; no sliding-window attention.

### GGUF format facts
100% standard v3. 428 tensors. `general.alignment = 32`. Data at byte 8,250,912.
Offsets contiguous, no trailing bytes. Q4_0 x220, F32 x207, **Q6_K x1**
(`token_embd.weight`). ggml type 14 *is* Q6_K. There is no custom format.

---

## 2. ARCHITECTURE, TOP TO BOTTOM

### 2.1 The byte model
```
bytes per token            1,466,751,488   (1466.75 MB)
  of which the 4-of-64 selected experts    807 MB
  lm_head alone                            215 MB, paid per VERIFIED POSITION,
                                           so K x 215 MB for any K-token scheme
                                           regardless of expert reuse
roofline at 956 GB/s       1.534 ms  ->  652 tok/s hard ceiling at batch 1
```

**Checked, not assumed.** The profiler's own per-call byte table, summed over the
default build's actual call counts, is **1,466,603,152 B** of weight traffic;
adding the ctx-dependent attention term gives 1,472.17 MB, which is exactly the
`TOTAL` row the live profile printed. The hardcoded constant is 0.010% above that
sum and its exact origin is not documented — small enough to be immaterial, noted
so a reviewer does not have to find it.

There is **no 68 MB discrepancy** against `bench_shapes`' "1398.70 MB": that tool
prints MiB. 1398.70 MiB = 1466.64 MB. The project mixes MB (1e6) and MiB
(1,048,576) between tools; it has misled a reader once, in this document's drafting.

### 2.2 Memory layout — the planar (SoA) repack
The GGUF's interleaved blocks are repacked **losslessly** into planar planes so
each dot product walks one contiguous stream per component. Verified bit-exact by
`tools/verify_pack.py` against a reference dequantiser, all 428 tensors.

```
Q4_0 block -> qs (16 B) | d (2 B)
Q6_K block -> ql (128 B) | qh (64 B) | scales (16 B) | d (2 B)
              128 + 64 + 16 = 208 = 13 x 16 B
```
The reason is alignment: in the native interleaved layout `block_q6_K` is 210 B,
so only block 0 of each 2048-wide row is 16-byte aligned. All weights live in one
device `d_pack` buffer (~12.64 GiB); zero-copy from the mmap is not an option
because mmap is not pinned and PCIe would be ~40x too slow for the HBM roofline.

### 2.3 The central trick — Phase 4y, the scale-invariant Q8_1 norm split
**This is the most novel component in the engine, so audit it hardest (§12 A1).**

An RMSNorm is normally `y = (x/rms)*w` then quantise `y` to Q8_1 for the next
GEMV. That forces the norm into its own kernel: the GEMV cannot start until every
element of `y` is known, so you pay a full launch and a full round trip of the
hidden vector through memory. The engine ran **81 such norm nodes**.

Instead the norm's scale is **carried into the Q8_1 block header**:

```c
struct block_q8_1 { __half2 ds; int8_t qs[32]; };   // 36 bytes
    ds.x = d   = amax / 127          // the usual Q8_1 scale
    ds.y = SUM = RAW FP32 SUM of the 32 activations, before any scaling

dot_q4_0_q8_1 = d4 * ( d8 * sumi  -  8.0f * SUM )
```

Now the algebraic observation that makes it legal:

* `ds.x` and `ds.y` are both **linear** in a positive scalar multiplier;
* `qs` and `sumi` are **invariant** under a positive scalar multiplier;
* therefore `dot(inv * u) = inv * dot(u)`.

So the producer emits the quantisation of the **unnormalised** vector plus the
reciprocal RMS, and the consumer folds `inv` in with **one multiply per output
row**. There is **no barrier, no threadfence, no atomicAdd, no counter, and
nothing to reset** between tokens — which is what makes the fusion legal inside a
CUDA graph replay where you cannot synchronise.

The obvious alternative — producer emits `amax/127`, consumer rescales the 64
`ds` half2 values — rounds `d` through fp16 **twice**, for ~5e-4 relative error on
every block of every layer. This form rounds once.

Rejected alternatives, with the reasons the author recorded:
* reduce the partials in shared memory and broadcast — introduces the only
  `__syncthreads()` into kernels that have none, and **deadlocks** wherever a warp
  can leave early via `if (row >= out_rows) return`;
* `atomicAdd` the partials — non-deterministic, so replay is not reproducible;
* warp 0 reduces while the others stream — warp 0 also owns an output row, so it
  lands on the block's critical path anyway.

Shape constraint: it requires `LK_GEMV_BIGBLOCK` (1024 threads, `rows/32` blocks),
so that block `b` owns rows `32b..32b+31` = **exactly Q8_1 block `b`**, one row per
warp. Without that, a Q8_1 block spans four blocks and the epilogue would need a
cross-block barrier. There is a `#error` enforcing it.

**The price, stated plainly.** This reassociates `(x*inv)*w` into `inv*(x*w)`,
which rounds the fp16 Q8_1 scale one ulp differently. The residual stream
amplifies a perturbation ~20% per layer (§6.1), so the greedy trajectory is no
longer reproducible: **C1 fell from 191/825 to 88/825.** The claim is that this is
*inherent* — no barrier-free norm fusion can reproduce `(x*inv)*w` bit-for-bit,
because the producer cannot know `inv`. **Attack that claim**: a barrier-free
formulation that is also bit-identical would be worth more than every remaining
optimisation combined.

### 2.4 Kernel inventory
27 `__global__` definitions, all hand-written. Launch configs as actually used by
`engine.cu` at the default build:

| kernel | launch | role |
|---|---|---|
| `k_embed_q6_K` | `<<<8,256>>>` | Q6_K row lookup into `x[2048]` |
| `k_rmsnorm_q8_1` | `<<<1,256,8192>>>` | the one surviving standalone norm (layer 0) |
| `k_gemv_q4_0` | `<<<768,256>>>` / `<<<64,1024>>>` / `<<<1472,256>>>` | general Q4_0 GEMV |
| `k_gemv_q4_0_prod` | `<<<64,1024>>>` | **Phase 4y producer**: GEMV + residual + norm + Q8_1 |
| `k_gemv_q4_0_3` | `<<<384,256>>>` | q/k/v in one launch (3072 warps) |
| `k_gemv_q4_0_exp2` | `<<<dim3(192,8),256>>>` | MoE gate+up, all 4 experts, one launch |
| `k_gemv_q6_K_dp4a` | `<<<16000,256>>>` | lm_head |
| `k_swiglu` | `<<<48,256>>>` / `<<<64,256>>>` | SwiGLU + Q8_1; **hosts the L2 prefetch** |
| `k_shortconv` | `<<<8,256>>>` | the 3-tap causal conv |
| `k_router_dot` | `<<<64,32>>>` | one warp per expert, 2048 threads |
| `k_router_top4` | `<<<1,64>>>` | the selection |
| `k_attn_split` | `<<<dim3(8,128),256>>>` | split-K, GQA-aware, one block per kv head |
| `k_attn_combine` | `<<<32,64>>>` | deterministic two-stage combine |
| `k_qk_norm_rope` | `<<<40,64>>>` | per-head norm + NEOX RoPE + KV cache write |
| `k_moe_down_accum` | `<<<64,1024>>>` | **T1**: down projection + 4-way accum + residual + norm + Q8_1 |
| `k_argmax_part` | `<<<125,256>>>` | stage 1 |
| `k_argmax_final` | `<<<1,256>>>` | stage 2, writes `d_token` and bumps `d_pos` |

Retired but kept compiled: `k_router_dot_top4` (`LK_FUSE_ROUTER` is retired by an
unconditional `#error`, because it never learned about Phase 4y's `inv_rms` and
would route on unscaled logits), `k_gemv_q4_0_exp`, `k_moe_accum4`,
`k_accum_scaled`, `k_quantize_q8_1`, `k_rmsnorm`, `k_gemv_q4_0_norm`
(`FUSE_NORM=0`, measured 1.6% slower).

### 2.5 The step, as a graph
The whole forward pass is captured once into a CUDA graph and replayed. Capture
is **position-independent**: `d_token`/`d_pos` are device-resident and
`k_argmax_final` bumps `d_pos` on device, so the whole decode loop stays on the
GPU. `cudaGraphGetNodes` reports **343 nodes** — confirmed live.

    1 embed + 1 layer-0 norm
    + 30 x 3  ShortConv  (in, conv, out)
    + 10 x 5  attention  (qkv, qk_norm_rope, attn_split, attn_combine, attn_out)
    +  2 x 4  dense      (gate, up, swiglu, down)
    + 38 x 5  MoE        (router_dot, router_top4, gate_up, swiglu, down)
    + 3       tail       (lm_head, argmax_part, argmax_final)
    = 343

Per-layer dataflow, MoE:
`x -> [norm+Q8_1 fused into the producer] -> router_dot -> router_top4 (writes
d_idx, d_w) -> moe_gate_up -> swiglu -> moe_down (4 experts walked SEQUENTIALLY)
-> residual`.

There is **no separate residual-add kernel**. Residual accumulation is
`add_residual` inside the GEMVs (`xv = add_residual ? (y[row] + acc) : acc`), and
`k_moe_down_accum` does `xv = dst[row] + a`.

---

## 3. THE PROFILER, AND EVERY OTHER INSTRUMENT

Every number in this document descends from these. Audit them first.

### 3.1 The profiler
`LK_PROFILE=1` brackets each launch in its own event pair on the same stream
(`LK_PROF_MAX = 2048` brackets, 4096 events), aggregates by tag, and prints
`category | calls | total_ms | avg_us | MB | GB/s | %step` sorted by total time.
Bytes are per-call, from a `strcmp` table, multiplied by call count; `attention`
bytes are overridden with `ctx * 2048`.

Its own header comment is the honest statement of the problem it was built for:

> Every hypothesis I formed about where the remaining 2.3 ms goes was falsified by
> a variant build, which means the bottleneck is something I have not thought to
> vary. Instrument instead of guessing.

**Three separate things are wrong with it, and all three were found by pricing
the instrument rather than by reasoning.**

**(a) The bracket is not free, and the marginal cost is 1.35–1.44 µs.** Measured
three independent ways, live:

| measurement | value |
|---|---|
| `bench_launch` (1) batched empty launches | 1.230 µs/launch |
| `bench_launch` (2) the same, event-bracketed | 2.613 µs/launch |
| **increment = the bracket** | **1.383 µs** |
| `bench_ovh` part C, over real streaming kernels | 1.35 / 1.38 / 1.44 µs |

Note the README says "~2.1 us"; that is the *absolute* bracketed empty launch from
an earlier session, conflated with the increment. **The increment is 1.38 µs**,
and 333 brackets × 1.38 µs = **0.46 ms of pure instrument** in a 2.27 ms step.
This is why the small-kernel rows are unusable, and it is the arithmetic behind
the project's own rule: *price the instrument before believing it.*

**(b) The `GB/s` column is not comparable across kernel sizes.** This retracted an
earlier claim of the author's. Every streaming kernel showed a different
bandwidth, falling monotonically with size:

```
lm_head       215.04 MB/call   925.1 GB/s
moe_gate_up    14.16 MB/call   727.8 GB/s
moe_down        7.08 MB/call   619.9 GB/s
attn_qkv        3.54 MB/call   431.3 GB/s
router_dot      0.52 MB/call   110.0 GB/s
```
A least-squares fit of `time = a + bytes/BW` over all **eleven** streaming kernels
gives **BW = 941.8 GB/s, a = 4.12 µs per call, every residual inside ±1.03 µs**.
So every one of those kernels was already at peak *marginal* bandwidth and the
whole 8x spread was **one constant plus the bracket**. The "619.9 vs 727.8 GB/s
gap between `moe_down` and `moe_gate_up`" that Phase 4y reported **does not exist
and is retracted**.

`bench_ovh` confirmed it directly: all of the engine's own shapes land within 2%
of each other while the profile table showed 17% between them. **The spread was
instrument, not silicon.**

**(c) NEW FINDING — in a saturated stream the bracket quantises to ~1 µs, so the
profiler cannot resolve a small kernel at all.** The author suspected this from
the table itself: the five `calls==1` rows printed **231.42, 6.14, 5.12, 4.10,
4.10 µs** and *every one* is an exact multiple of **1.024 µs** (226, 6, 5, 4 and 4
ticks). Five for five is not chance.

It was then reproduced directly. A test that queues 333 tiny kernels, each in its
own event pair on one stream with a single sync at the end — **the profiler's
exact regime** — reports:

```
2048.00 ns  x90      <- exactly 2 x 1.024 us
3072.00 ns  x217     <- exactly 3 x 1.024 us
4096.00 ns  x4       <- exactly 4 x 1.024 us
plus a fine tail at 2080..2370 ns
```
311 of 333 land exactly on the 1.024 µs grid. In *isolation* (one sync per
bracket) the timer resolves 32 ns, so this is a property of the **saturated**
regime, which is the regime the profiler runs in.

**Consequence, and it matters:** `router_top4` printing 4.10 µs and `embed`
printing 4.10 µs are **the same tick of the same clock**, not two measurements
that agree. No row in the profile table below ~5 µs is a measurement of that
kernel. This is independent confirmation that the trustworthy exchange rate for
deleting a node is the **0.3–0.45 µs measured by fusion experiments** (where no
event bracket is involved), not the 1.5 µs Phase 4y inferred from the table and
not the 3.811 µs `bench_launch` part (7) reports for a bracketed norm node
inserted into a busy stream.

**(d) NEW FINDING — the profiler's bracket granularity is not the graph's node
granularity.** `LKP_B("attention")` wraps **both** `k_attn_split` **and**
`k_attn_combine`, so the table prints **333 rows for 343 kernels** (confirmed
live: `TOTAL 333`) and the `attention` row's 12.90 µs average is a *two-kernel
sum* whose byte count — `ctx*2048` — belongs to one of them. The `attention` row's
43.2 GB/s is therefore meaningless, and the row cannot be compared against any
single-kernel row.

### 3.2 The instruments that are not the profiler
| instrument | what it measures | status |
|---|---|---|
| `bandwidth_probe` | achievable DRAM bandwidth | **956 GB/s** |
| `bench_launch` | launch / bracket / graph-node cost, by grid size | valid; parts (1)–(7) |
| `bench_ovh` | A: grid sweep; B: time vs bytes; C: same with the bracket; D: graph replay; E: occupancy and wave quantisation, on the real kernels | valid |
| `bench_shapes` | **mode 0**: 343 empty kernels with exact engine shapes = the tollbooth floor. **mode 1**: every node reads its real bytes, no dependencies | mode 0 valid; **mode 1 INVALID, see below** |
| `bench_norm`, `bench_small` | true marginal cost of small kernels, back-to-back under one event pair | valid |
| `LK_ABLATE_SMALL` | empties a group of non-streaming kernels, keeping grids and weight traffic | valid **only with the contamination guard** |
| `test_*` | G2 dots/kernels, G6 attention, G7 normfuse, G8 normscale | all PASS |
| `tools/paired.py` | paired per-token nll with a sign test | validated: same binary twice → **exactly 0.000000**; pairing sharpens the standard error **14.2x** (sd(nll) 2.3775 → sd(Δ) 0.2358, se 0.1420 → 0.00999). Honest bound: it resolves **2.0%** of perplexity on 557 tokens, not the 0.08% predicted for it |
| `tools/ab.py` | bit-identity over 7 trajectory cases | 7/7 |
| `tools/sweep.py`, `abbench.sh` | throughput, interleaved | — |
| `tools/lookup.py`, `g4b.py`, `g4c.py` | speculation and expert-reuse simulation | — |

### 3.3 An instrument that failed, kept rather than deleted
`bench_shapes` mode 1 gives every node its real byte count and reads it cold. It
reports **3.331 ms for 1398.70 MB = 440.3 GB/s**, which is *slower* than the real
engine's 646 GB/s on more traffic — impossible for a "floor". Its grid-stride read
loop is not unrolled, so nodes with a small grid and many bytes per thread run at
far lower memory-level parallelism than the real kernels. It is an invalid
instrument. It is **recorded, not deleted**, so it is not rebuilt by accident.
Mode 0 is unaffected: an empty kernel cannot be affected by an access pattern.

### 3.4 The contamination incident — the most instructive failure in the project
While ablating the router, the reported speedups were **+28.5%** and **+36.2%**.
Both were false.

`LK_ABLATE_SMALL` empties `k_router_top4`, which leaves `d_idx` holding zeros. But
`d_idx[k]` is a **WEIGHT ADDRESS**, so `moe_gate_up` then read **one expert four
times**. That expert is 3.5 MB and the L2 is 6 MB, so three quarters of
`moe_gate_up`'s DRAM traffic **silently vanished**. The ablation had changed
*where* memory is read, not how much work is done.

Guarded by a stub that writes four DISTINCT experts, the honest numbers are
**+6.0%** and **+10.8%**. The guard is in the source with that reasoning in a
comment, and `d_idx`/`d_w` are now `cudaMemset` to 0 at init, because an
uninitialised buffer feeding an address computation is a latent bug independent
of the experiment.

> **Generalised law: an ablation that changes WHERE memory is read is not
> measuring what it claims to.**

### 3.5 The tollbooth
`bench_shapes` mode 0, re-measured live:

```
device: NVIDIA GeForce RTX 3090 Ti  SMs=84  maxThreadsPerSM=1536
nodes reconstructed: 343   total bytes: 1398.70 MB
343 nodes, graph replay :  0.268 ms/token   0.781 us/node
as a fraction of the real 2.269 ms step: 11.8%
per-node cost the "2.07 us toll" model needs: 2.07 us
```

This is the pure launch/dispatch/drain toll with **no memory traffic at all**, and
it **refuted** the model that the ~0.71 ms of non-wire time was a 2.07 µs toll per
node. It is 0.781 µs. This agrees with the fusion experiments (T1 deleted 38 nodes
for +0.67% = 0.40 µs/node) and disagrees with the 2.07 µs model by 2.65x.

`bench_launch` part (5) adds the missing structure — **the empty-node cost depends
on the grid**, so 0.781 µs is an average, not a constant:

```
grid     1 :  0.605 us/node        grid   256 :  0.823 us/node
grid     8 :  0.618 us/node        grid   768 :  1.077 us/node
grid    64 :  0.673 us/node        grid  1024 :  1.208 us/node
                                   grid  1536 :  1.512 us/node
```

### 3.6 The tension the instruments have not resolved
`bench_launch` part (7) measures the marginal cost of a norm-shaped node
**inserted behind a busy stream**:

```
32 streaming nodes alone         :  690.483 us  (21.58 us each)
same 32 + 32 norm nodes          :  812.442 us  (25.39 us each)
--> MARGINAL COST PER NORM NODE  :  3.811 us
```
but the **observed** exchange rate from real fusions is **0.3–0.45 µs/node**.

The reconciliation the author believes is right, and which a reviewer should
attack: **fusion MOVES work, it does not remove it.** T1 moved `k_moe_accum4`'s
weighted sum into `k_moe_down_accum`, so only the launch and the 32 KB round trip
disappeared — 0.40 µs. `bench_launch` (7) *removes* a node that does real work in
a serial chain, so it saves the full serial latency. Three numbers that look
contradictory (0.3–0.45, 1.5, 3.811 µs) are three different questions. **This is
the least-settled measurement in the project.**

---

## 4. THE QUANTITATIVE MODEL

Measured step at ctx 272, live: **2.978 ms of bracketed kernel time** (which
includes 0.46 ms of instrument) against a **2.269 ms** wall-clock step.

The decomposition, every term measured rather than estimated:

```
1.557 ms   wire, 1466.75 MB at the 941.8 GB/s MARGINAL rate
           (the asymptote, never reached in practice)
0.220 ms   streaming size inefficiency: 7 MB kernels reach ~766 GB/s and
           14 MB kernels ~829, because each is too small to amortise its ramp
0.268 ms   launch toll: 343 nodes x 0.781 us, from an empty-kernel graph
           carrying the real grids and block sizes
0.221 ms   the 158 small kernels: execution plus the dependency stalls they
           impose on everything behind them
---------
2.266 ms
```

**The argument that matters, and the one to attack:** the 0.268 ms of launch GAPS
has **no kernel running**, so nothing can issue a load and it is unfillable at
batch 1. The 0.221 ms of small-kernel execution **does** have a kernel running, so
it is fillable in principle. Therefore the batch-1 ceiling is **488.9 tok/s** —
measured two independent ways (the ablation, and the window argument).

Independent hardware confirmation with no shared code path: `utilization.memory`
reads **82%** while streaming and **1%** while the small kernels run, reproducing
the 86%-while-streaming number from the hardware counter instead of a timer.

---

## 5. EVERY DECISION, WITH ITS MEASUREMENT AND ITS VERDICT

Negative results are kept behind flags in the tree so they stay reproducible.

### Phase 3 — the parity diagnosis (8 rounds, three of them wrong)
Symptom: plausible output, no greedy parity. Rounds 4 and 5 concluded the
`out_proj` GEMV was the culprit. **Round 6 proved the GEMV exact and showed round
5's conclusion was a harness artifact.** Round 7 found the real root cause: **fp32
summation order, amplified.** Round 8 attempted the obvious fix and **falsified its
own root-cause fix**, which is what produced the amplification model.

Round 8 replicated ggml-cuda's `rms_norm_f32<1024>` exactly — same 1024 threads,
same butterfly, same accumulation order — and it **regressed `factual` from 66/66
to 38/66.** Isolating the change confirmed the rewrite itself was the cause, and
restoring the original 256-thread tree reduce returned 66/66. So **llama.cpp is
not executing `rms_norm_f32<1024>` on this path for this model and batch size**;
there is a fused or differently-configured norm in its actual graph. *Which kernel
it really runs is still unidentified* — an honest open question, not a resolved
one. The falsification is also what produced the amplification model, so the round
paid for itself. What the author wrote at the time: *"I optimised toward a
hypothesis instead of measuring. The measurement falsified it in one run, which is
cheap, but I should have made the falsification test first."*

The chain, measured end to end in layer 0: a different summation order changes a
few low bits of the residual → the next layer's RMSNorm divides by a slightly
different rms → that rescales the whole vector → the Q8_1 quantiser rounds
differently → and the **discretisation makes the difference grow rather than
average out**. ~20% amplification per layer, 40 layers.

**Demonstrated, not asserted:** splitting the attention combine as `acc/l` versus
`acc*(1/l)` is one rounding difference of order **1e-8**. At a 1-token prompt the
two kernels are **bit-identical**. At 2 tokens the top-64 logits differ by
**0.276**. At 16 tokens by **0.724** — enough to flip a greedy token at position
39.

> **Consequence: multi-step greedy identity is not a usable correctness gate for
> this model.** C1 is a smoke alarm. C1a (step-1 top-1 plus full-vocabulary
> correlation) replaced it, and C2 is adjudicated against exact float64.

### Phase 4 — the router was the bottleneck
Decode was stuck at ~153 tok/s with healthy-looking kernels. Four cheap
falsification experiments, run *before* any optimisation, killed the three obvious
hypotheses and pointed at the real one:

```
-DLK_LOADONLY (delete ~80% of the arithmetic)   153.70 -> 153.47 tok/s  NOT instruction-bound
split-K probe -DLK_GEMV_WPR=2 (double the warps) 153.99 -> 152.98        NOT occupancy-bound
ONE k_gemv_q4_0 isolated, 2000 free launches     4.08 us = 577.8 GB/s    kernels are fine alone
q8 block re-read from HBM would imply 1733 GB/s  impossible              q8 is cached correctly
```
The model that fits: ~310 strictly serial kernels, 6.52 ms/token = **21.0 µs per
serial step**, of which one kernel with no dependency takes **4.08 µs** and graph
dispatch **0.78 µs** — so **~17 µs is exposed dependency, ramp and drain latency,
not work.** That also explains why fusing rmsnorm+quantiser (310 → 229 launches)
did *not* help: a step must be large enough to cover its own latency.

The decisive insight: **a kernel with one block cannot be fast, however many
threads it has.** Three of the most expensive kernels were single-block reductions
holding ~1 KB of loads outstanding; at ~500 cycles of HBM latency that is
**~2.8 GB/s**, a hard ceiling independent of block size. Each had previously been
"optimised" by *adding threads*, which could never work.

Fix, in two parts: split the router dot across 64 blocks (one warp per expert) and
the argmax across 125; then hand-unroll the top-4 *selection* out of a spilled
local-memory insertion sort into explicit registers, preserving the same insertion
and the same tie-breaking. Router dot **25.7 → 4.5 µs**, selection **15.3 → 2.2 µs**,
argmax **145 → 6 µs**. End to end **153.41 → 258.23 tok/s (+68%)**, and the engine
went from **0.68x to 1.14x llama.cpp** for the first time.

The suite parity moved 242/825 → 180/825 in that same change, and the author's
reasoning is worth stating because it recurs: the warp reduction changes the
router's summation order, ~1 in 4 of the 64-expert argmaxes sits close enough to
flip, and the criterion that would call that a regression is the one just retired
as unusable. **C2 improved.**

> **Generalised law: rank kernels by work-per-thread, not by traffic share. The
> router carried 1.4% of the bytes and was the single largest cost in the engine.**
> **VERDICT: ACCEPTED. The largest single win in the project.**

### Phase 4l — memory-level parallelism
Per-kernel timing with the instrument priced first. Unified root cause:
insufficient MLP, not insufficient bandwidth. **ACCEPTED.**

### Phase 4n/4o — 346 → 390 tok/s
`lm_head` was the last kernel not at the roofline. Quantisers fused into their
producers. The norm restructured to one load instead of two with no shared
staging. **ACCEPTED.**

### Phase 4u/4s/4v — the context cliff, and three instruments lying
The first diagnosis was **wrong about the cause**. `LK_ATTN_ABLATE` found it: the
attention score loop fetched one token's 64 dims — 128 contiguous bytes, exactly
one cache line — with **64 separate 2-byte loads**. At a fixed dim the 32 lanes of
a warp sit on 32 *different* tokens, so that is 32 lines touched per instruction,
64 times over: **2048 L1 accesses per warp per 32 tokens where 256 will do.**

Fix: load the line as **8 x uint4**. Dims are still accumulated `d = 0,1,...,63` in
that order, so the result is **bit-identical** (parity unchanged). 8k context
**243 → 334 tok/s (+37%)**; short context 410 → 417.

**The same treatment fails on the output scan, and was measured worse twice:**
there each of the 64 threads reads a different dim of the *same* line, so it is
already perfectly coalesced. *Two loops in one kernel with the same symptom and
opposite fixes.*

Split-K then fixed the parallelism: one block per query head leaves 32 blocks on
84 SMs and each scans the whole context **twice**. Split-K (32 heads x 8 context
slices, deterministic two-stage combine) costs nothing at short context because
empty slices retire immediately. **The split count must depend on the context.**

**Three cheaper fixes were tried and falsified first:** batching the V loads (no
effect), transposing the KV cache (worse — token-major was already coalesced),
widening the block to 1024 threads (neutral). Ablations put the score loop at
**33.7 ns** per context token and the V loop at **24.9 ns**, so **both** halves
needed the parallelism, not just the suspected one. Context slope fell from
**216 ns** per context token to **40 ns** (llama.cpp: 33 ns).

The GQA rewrite cut context traffic a further **4x** (8 kv-head blocks instead of
32, so DRAM sees each K and V byte exactly once): 8k context costs 164.5 MB
instead of 658 MB per layer-token.

**Honesty note the project recorded against itself:** an earlier context table
compared our fresh numbers against llama.cpp numbers from a **different session**
and reported a **1.76x** short-context lead. Re-measured in the same session it is
**1.62x**. The discrepancy was ~2% cross-session drift, not an engine difference.

### Phase 4x — the launch-fusion plan, measured and rejected
Step 1: fold the FFN norm into its producing projection. Step 3: fold the top-4
into the router dot (behind a threadfence+atomicAdd barrier). Both measured, both
lost (0.09–0.39% slower). **REJECTED.** Phase 4y later found the one fusion that
*does* pay — by carrying the **scale** instead of the **barrier**.

### Phase 4y — the scale-invariant Q8_1 norm split
Mechanism in §2.3. **81 norm nodes down to 1.** 414.7 → 438.8 tok/s over the
phase. Gate G8 (`tests/test_normscale.cu`) verifies the scale invariance directly,
and includes a **no-scale control that must be measurably wrong** — a negative
control for the gate itself.

Two bugs, and what caught them:
* `k_gemv_q4_0_prod` **read and wrote `xq`** → the model repeated one token
  forever. Fixed via a second buffer `xq2`.
* `lk_norm_prod_warp(..., np.yq8 + blockIdx.x, ..., blockIdx.x)` in fact wrote
  `yq8 + 2*blockIdx.x`.

Both were caught by `LK_LOGIT_DUMP` + `topk`, which compares the **full 128k logit
vector** (maxabs / rms / median / correlation) rather than a token trajectory.

> **Generalised law: the logit-vector dump is the fastest discriminator for a
> suspected numerical bug.** A token trajectory is chaotic and tells you almost
> nothing; the full vector tells you immediately and quantitatively.

**ACCEPTED, with the C1 cost stated rather than buried.**

The accuracy accounting, honestly, because this is the change that moved C1:

```
interleaved A/B, REPS=5, both halves (80 of the 81 norm nodes removed)
    ctx 272    414.73 -> 438.78   +5.80%
    ctx 2128   404.53 -> 426.07   +5.33%
    ctx 8128   364.78 -> 383.20   +5.05%

C2, lm_head vs exact float64 GIVEN the engine's residual stream
    baseline 0.0013%   ACCUM 0.0067%   GEMV half 0.0137%   both 0.0140%
    gate: <= 0.1%
end-to-end nll, 557-token natural corpus
    baseline 1.890377   ACCUM 1.903387   GEMV 1.888308   both 1.880269
engine-vs-oracle perplexity distance, same corpus
    baseline 1.5421%    GEMV 1.3322%     both 0.5209%
```
So the fused builds are **5–10x further from exact on C2 and still 7x inside the
gate**, and the end-to-end loss is marginally *better*. The honest size of the
residual-stream perturbation, comparing the full 128k logit vectors: **median
|diff| 0.110, max 0.892, correlation 0.99747** — about **1000x more than a naive
rounding-drift estimate**, and dominated by a common-mode shift that cancels in
the softmax (which is why nll moves 0.02% while individual logits move 1%). The
GEMV half adds 42 more sites and moves the median to 0.133, i.e. it behaves
exactly like the half already accepted.

Gate G8's diagnostics explain the residual rather than merely bounding it: the
int8 payload is **bit-identical to the reference (0 flips in 2048 elements)**,
every scale field reconstructs to within one fp16 ulp and **~78% bit-exactly**,
and the remaining **~22% are one ulp out** — the signature of two fp16 roundings
not commuting, which is exactly the predicted mechanism. An earlier draft of that
gate asserted the fields would match *exactly* with a 2e-4 tolerance; both were
guesses, and **2e-4 is below fp16's own relative precision of 4.9e-4**, so it was
unachievable by construction. The bound is now derived from the format, not from
the reading.

### Phase 4z — the profiler's fixed cost, and where node deletion stops paying
Retracted the author's own prior claim (the `moe_down` / `moe_gate_up` GB/s gap).
Measured the **exchange rate for deleting a node: 0.3–0.45 µs**, not the 1.5 µs
Phase 4y implied. That killed a class of planned work **on paper, before it was
built**. Also added a gate that can see residual-stream error.

> **Generalised law: a barrier at the END of a kernel is cheap; at the START it
> gates the critical path.**

T1 — `k_moe_accum4` folded into `k_moe_down_accum`. **ACCEPTED, +0.69%**
(439.48 → 442.43 at ctx 272; 383.18 → 385.31 at ctx 8128), **bit-identical 7/7**,
graph nodes 381 → 343. Also tried and rejected inside T1: hoisting all four expert
dots above their reductions so the loads overlap — **12.77 vs 12.32 µs**; four live
accumulators and four row pointers cost more registers than the extra MLP returns.

T4r — the router top-4 rebuilt in the consumer's prologue. **REJECTED, -0.17%**
(pre4z 438.31 → T1 441.32 → T1+T4r 440.90, spreads 0.03%/0.05%). The mechanism is
sound and the result is still negative **because of placement**: hosting the
selection in `moe_gate_up` puts a `__syncthreads` in front of the weight stream of
the hottest kernel in the engine. Kept behind `ROUTER_PRO`.

T2 — SwiGLU folded into the gate/up GEMV. **REJECTED, -1.75%** (441.69 → 433.96),
**bit-identical 7/7**, `moe_gate_up` 19.36 → 22.00 µs. Forcing each warp to own one
row of **both** projections drops the grid from 12288 warps to 3072, and there is
no shape that avoids it. This also required a dedicated `moe_act` buffer, because
a fused kernel has some blocks writing `xq2` while others still read it — *the
Phase 4y aliasing bug exactly*.

> **Generalised law: fusing into `moe_gate_up` loses.** Both failures were
> bit-identical and both were barrier-free. It is that `moe_gate_up` is the
> bandwidth-critical kernel (24% of the step, 14.16 MB per call) and its shape is
> already at a local optimum.

### Phase 4z.1 — the tollbooth, and the occupancy cliff that is not there
Tollbooth 0.268 ms, not 0.712. Then the occupancy question, attacked directly
because 1024-thread blocks give 66.7% occupancy on GA102. Six configurations
spanning **0.76 to 3.05 waves**, at a cold 7.078 MB read, re-measured live:

```
grid x blk   threads  blk/SM  resident  waves   unbr us    GB/s
64x1024        65536       1        84   0.76      9.42   751.1   1 wave, 1024-thr block
192x1024      196608       1        84   2.29      9.36   756.3   T2's shape
192x512        98304       3       252   0.76      9.40   752.7
384x512       196608       3       252   1.52      9.25   764.9   same THREADS as above
768x256       196608       6       504   1.52      9.24   766.0
1536x256      393216       6       504   3.05      9.22   768.0   moe_gate_up today
```
**The entire spread is 2.2%, and T2's exact shape is not the slow one.**
> **VERDICT: occupancy and wave quantisation are worth 1–2%, not 13%.** A whole
> family of proposed fixes closed with data.

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` on the real kernels adds that
`k_moe_down_accum`, `k_gemv_q4_0_prod`, `k_gemv_q4_0_exp2` and `k_gemv_q4_0` are
all **register-limited at 1024 threads** (1024 threads/SM against the 1536 cap),
reaching the cap only at 512 or 256 — which does not change the 64-block kernels,
where block count is what limits occupancy.

### Phase 4z.2 — the decomposition and the batch-1 ceiling
Required the contamination guard (§3.4) before any of it could be believed.

```
group 1  router_dot + router_top4   76 nodes   contaminated +28.5%   guarded +6.0%
group 2  swiglu                     38 nodes                         +1.0%
group 3  shortconv (the 3-tap conv) 30 nodes                         +1.1%
group 4  the rest                   14 nodes                         +1.0%
group 5  all of the above          158 nodes   contaminated +36.2%   guarded +10.8%
```
The router is the largest single item: **0.129 ms, 5.7%**.

`LK_ROUTER_UNROLL` was made sweepable and proved **bit-identical at every value
before being swept** (the accumulation runs `i = lane, lane+32, ...` in strictly
increasing index order and the groups are contiguous, so changing U only
re-partitions that sequence). Swept 16/32/64: **441.39 / 441.36 / 441.36 tok/s —
flat.** The router is **not** load-limited; its cost is kernel *latency*, and that
is exactly the cost that three separate fusion attempts failed to remove without
paying more elsewhere.

### Phase 4z.3 — the second GPU, measured, then ruled out by instruction
Measured first, as always: `cudaDeviceCanAccessPeer = 0` in **both** directions;
`nvidia-smi topo -m` reports **PHB**; no NVLink. Host-staged exchange through
pinned memory: 8192 B → D2H 1.19 µs, H2D 5.03 µs, round trip **6.52 µs**,
identical with GPU 1 idle (ruling out PCIe contention).

An earlier version of that probe reported 18.15 µs, which would have killed the
idea; it was measuring an **error path** — it passed a device pointer to
`cudaMemcpyHostToDevice`.

Tensor parallelism over that link projects to **509–534 tok/s**, at the cost of
sharding nearly every kernel. The cheaper use of the same card — `lm_head` alone,
which needs **one exchange per token** rather than per layer — is ~465 tok/s for
5.1% of the step in saved time and 0.3% in communication.

**VERDICT: measured, then PLACED OUT OF SCOPE BY EXPLICIT USER INSTRUCTION
("Only the one graphics card may be used").** Recorded because the measurement is
real and someone will ask.

### Phase 4z.4 — prompt-lookup speculation, measured and rejected
Greedy speculative decoding with exact verification is **lossless**, so its
acceptance rate can be simulated exactly offline from trajectories the engine
already generates. `tools/lookup.py` does that.

M=4 verification costs 2.11–2.41x a plain token, so break-even is
`E[tokens per pass] = 2.11..2.41`.

```
minmatch=2 (7 cases, ngen=384)
    factual    88.6% trigger   E 3.61   661-755 tok/s
    math       45.7%           E 1.50   275-314
    code       39.8%           E 1.45   265-303
    agentic    28.3%           E 1.38   253-289
    reasoning  48.1%           E 1.53   280-320
    prose      40.4%           E 1.70   312-357
    ALL                        E 1.87   -> DOES NOT PAY
minmatch=4  ALL E 1.75        -> DOES NOT PAY
```
Longer matches do not rescue it: precision rises more slowly than the trigger rate
falls. **And code, the case everyone expects lookup to win, is among the worst at
E 1.45** — it would make the engine 30–40% *slower*. The one case that pays,
`factual`, restates the same sentence in a visible reasoning trace; it is one case
of seven and it is not a workload.

**This measurement cost an afternoon and cancelled a week of work.**

### Phase 4z.5 — the power limit, measured (the user's own hypothesis)
The hypothesis was reasonable: 0.71 ms of the step is latency-bound work, and
latency-bound work scales with SM clock. Measured under load:

```
SM clock       2055 MHz  against a 2100 MHz maximum   = 97.9%
Memory clock  10501 MHz  against a 10501 MHz maximum  = 100%
Power draw      362.7 W   against a 450 W limit       = 80.6%
SW Power Cap   Not Active   (counter: 0 us)
HW Slowdown    Not Active
utilization.gpu 100%   utilization.memory 82% during streaming
```
**REFUTED, no headroom.** And structurally: this workload is memory-bound, so the
SMs spend their time **stalled** on loads rather than issuing arithmetic, and
stalled SMs do not draw power. A 450 W budget with 87 W unused is the signature of
a kernel that is **waiting, not computing**.

### Phase 4z.6 — speculative L2 prefetch, the last idea, tried and shipped
Premise: the ~20% idle splits into 0.268 ms of launch GAPS (unfillable) and
0.221 ms of small kernels EXECUTING with memory unused (fillable in principle).
**The prize is bounded by the WINDOW, not the traffic.**

Built: `k_swiglu` hosts it (38 calls, 12288 threads, ~1.5 µs window, and by then
`d_idx` is known). It prefetches the **heads** of all four `moe_down` experts,
because the first wave of `moe_down` reads all four concurrently. `LK_PF_BYTES` is
a runtime knob so the size could be swept without rebuilding. **Nothing consumes
the prefetched lines, so `tools/ab.py` stays 7/7 bit-identical** — which made it
the cheapest remaining idea to prove correct.

```
LK_PF_BYTES    0        512K     1M       2M       4M
mean tok/s   441.31    439.45   442.86   443.49   441.34
vs off           -      -0.42%   +0.35%   +0.49%   +0.01%
```
A clean inverted U with an interior optimum. 512K being **worse than off in all
three rounds** is systematic, not noise: a partial prefetch adds queue pressure
without covering enough to pay for it.

7 interleaved rounds at 0 vs 2M: **441.17 (spread 0.13%) vs 442.73 (spread 1.09%),
6 of 7 ahead.** Isolated against the T1-only build at 0.03% spreads:
**441.75 → 443.23 = +0.34%.**

**The mechanism test came out backwards.** The worry was that
`prefetch.global.L2` is a HINT the controller may deprioritise, so
`LK_PF_LOAD=1` swaps in a real `ld.global` whose destination register is never
read:

```
prefetch.global.L2 hint   +0.35%
real ld.global             0.00%   (441.42/441.44, 441.47/441.40,
                                    441.38/441.37, 441.45/440.41)
```
**The hint beats the demand load.** The priority hypothesis was backwards: a
demand load occupies LSU and scoreboard resources and buys nothing, while the hint
is nearly free to issue and does consume whatever bandwidth is spare. The author
does not have a complete explanation and wants one (§12 A6).

**VERDICT: ACCEPTED, shipped as the default (2 MB), bit-identical — at +0.34%
against a theoretical ceiling of 9.7%.** It captured ~14% of the `moe_swiglu`
window and ~3.5% of the total idle window. The dependency-forced idle windows
cannot be meaningfully filled by prefetching, because the memory system will not
service enough hint traffic inside a 1.5 µs window to matter. **The remaining
0.221 ms is structural**, exactly as the ablation independently said — and two
unrelated methods agreeing on the bound is why the attempt was worth making.

### Phase 4z in one line
```
438.8  start of Phase 4z          (pre4z re-measured live: 439.37)
440.8  + T1                                          +0.69%
443.0  + speculative L2 prefetch    (live)           +0.84% total
losses kept on the record: T4r -0.17%, T2 -1.75%
```

---

## 6. ACCURACY AND THE GATES

### 6.1 How fragile greedy parity is
One rounding difference of order **1e-8** in the attention combine:
**bit-identical at 1 token; top-64 logits differ by 0.276 at 2 tokens and 0.724 at
16; a greedy token flips at position 39.** A perturbation injected into the
residual stream at every layer is amplified by all 40 of them; a perturbation in
the final `lm_head` is not amplified at all. **The metric is scale-dependent.**

### 6.2 The gates
| gate | result |
|---|---|
| G2 dots | **8/8** against an exact integer reference |
| G2 kernels | RMSNorm / RoPE / ShortConv / router — PASS |
| G6 attention | equivalence against **both** the serial kernel and exact float64 — PASS, and it caught a real bug |
| G7 normfuse | PASS, across replays |
| G8 normscale | PASS, **with a no-scale negative control that must be measurably wrong** |
| C1 greedy parity | pre-4y build EXACT 66/66 on `factual`, 191/825 full suite. Default build **88/825** |
| C2 perplexity | **0.0140% from exact on 200 paired positions** against a 0.1% target — **PASS** (re-measured live) |
| C2 engine-vs-oracle | 0.5209% on the full 557-token corpus; top-1 agreement 93.36% |

`make BIGBLOCK=0 NORM_ACCUM=0 NORM_GEMV=0` restores the 191/825 build, **verified
bit-identical to it.**

**C1 is not claimed as passing, and the reason is not an excuse.** §6.1
demonstrates the gate cannot adjudicate for this model. What adjudicates is C2 and
the end-to-end loss.

### 6.3 C2's reference is not llama.cpp
`tools/perplexity.py` recomputes the `lm_head` in **float64 straight from the
GGUF**, because comparing two engines cannot say which one is right. Re-measured
live on the shipped build:

```
over 200 paired positions:
  exact (float64 lm_head) nll=1.795644  ppl=6.023353
  liquid-kernel           nll=1.795895  ppl=6.024863   -> 0.0140% from exact
  llama.cpp               nll=1.791696  ppl=5.999621   -> 0.2199% from exact
  oracle is 15.8x further from exact than the engine
full corpus (557 positions), engine vs oracle:
  ppl 6.555266 vs 6.521300 -> 0.5209% apart
  top-1 agreement: 520/557 = 93.36%
  GATE C2 vs exact (<=0.1%): PASS
```

**The engine is 15.8x closer to the exact model than llama.cpp is**, and the
entire two-engine "delta" is *the oracle's error*.

**This paragraph carried a wrong number into an earlier draft of this document,
and two wrong numbers are in the README.** The README's own C2 table says
`liquid-kernel 0.0222% / llama.cpp 2.0192% / 91x`, while its own status list
(line 352) says `0.0402%`, and `DIAGNOSIS.md` Phase 4m says
`0.0402% / 0.6896% / 17x`. **All four figures are stale; the measurement above is
current and matches the Phase 4y record's `0.0140%`.** The reason the comparison
ratio moved so much is that the *reference itself* moves: "exact" is recomputed
in float64 from the engine's own residual stream, so it changes whenever the
residual stream changes. Only the engine's distance from exact is a stable
quantity; the ratio is not.

The caveat the author states against himself: the exact reference is recomputed
from the engine's own dumped residual stream, so this validates the final norm +
`lm_head` stage against float64 mathematics — **not** the 40 layers upstream. That
blind spot is a real gap in the gate (Phase 4y flagged it: "a fusion that perturbs
the residual stream is only weakly constrained by C2") and `tools/paired.py` was
built to cover it.

The single-token comparison isolates the same effect with no upstream involvement:
feeding one prompt token and recomputing the `lm_head` in float64,

| | max abs deviation at top-64 logits | mean |
|---|---|---|
| liquid-kernel | **0.0235** | **0.0083** |
| llama.cpp | 0.6496 | 0.1880 |

A numpy simulation of Q8_1 quantisation reproduces the engine's figures **to five
decimals**, so the engine's `lm_head` is exactly what its own quantisation
predicts. This is why `tools/c1a.py`'s "Pearson >= 0.9999 against llama.cpp"
criterion is **not achievable by any implementation that quantises the activation
independently** — it measures the oracle's error. C1a top-1 agreement is 6/7.

> **Generalised law: a two-implementation comparison cannot say which
> implementation is right. Compute the exact answer from the weights; here it
> costs one float64 GEMM.**

---

## 7. CONTEXT SCALING, RE-MEASURED LIVE ON THE SHIPPED BUILD

```
# corpus: 558 natural tokens, ngen=64
     ctx      tok/s         GB/s pct_peak     kv_MB
     223     444.81        652.4     64.7       3.3
     936     438.50        643.2     63.8      17.9
    1711     432.88        634.9     63.0      33.7
    3292     423.56        621.3     61.6      66.1
    6454     399.52        586.0     58.1     130.9
```

**The README's context table is STALE and this is a real defect in the
documentation.** That table quotes 417.4 / 412.1 / 406.5 / 398.6 / 377.9 — the
pre-Phase-4y build — and the "1.62x" ratios in it were measured against llama.cpp
in a session that predates Phase 4y and Phase 4z. The engine is now **6.6% faster
at short context and 5.7% faster at 6.4k** than that table says.

Consequence for the headline: **"1.72x vs llama.cpp" is a short-context,
same-session figure and is sound. "1.61–1.72x out to 3.2k" is inherited from a
pre-4y session and needs re-measuring against the oracle on the current build.**
Until it is, treat the long-context ratios as indicative.

Also stale in the README body: **509 nodes** (now 343), **461 kernels**, and
"the realistic maximum is ~450–480 tok/s" (the measured ceiling is 488.9). The
header is current; the body is historical. Both cannot be read as current.

---

## 8. WHERE THE TIME GOES — WHAT THE PROFILER PRINTS, LIVE

`LK_PROFILE=1`, ctx 272, re-measured. Reproduced **as the instrument produces it**,
including the instrument's own defects, because reading this table correctly is
the point:

```
category          calls   total_ms    avg_us         MB      GB/s   %step
moe_gate_up          38      0.733     19.29     537.92     733.7   24.6%
moe_down             38      0.410     10.78     268.96     656.8   13.8%
shortconv_in         30      0.343     11.44     212.34     618.9   11.5%
lm_head               1      0.231    231.42     215.04     929.2    7.8%
shortconv_out        30      0.202      6.73      70.78     350.6    6.8%
router_dot           38      0.183      4.82      19.92     108.8    6.1%
moe_swiglu           38      0.171      4.50       0.00       0.0    5.7%
router_top4          38      0.156      4.10       0.00       0.0    5.2%
attention            10      0.129     12.90       5.57      43.2    4.3%
shortconv_conv       30      0.101      3.35       0.74       7.3    3.4%
attn_qkv             10      0.082      8.19      35.39     432.0    2.8%
attn_out             10      0.066      6.55      23.59     360.0    2.2%
dense_down            2      0.038     18.94      27.13     716.1    1.3%
dense_gate            2      0.036     17.92      27.13     757.0    1.2%
dense_up              2      0.036     17.92      27.13     757.0    1.2%
qk_norm_rope         10      0.036      3.58       0.00       0.0    1.2%
dense_swiglu          2      0.007      3.58       0.00       0.0    0.2%
norm_attn             1      0.006      6.14       0.02       2.7    0.2%
argmax_part           1      0.005      5.12       0.51     100.0    0.2%
embed                 1      0.004      4.10       0.00       0.4    0.1%
argmax_final          1      0.004      4.10       0.00       0.0    0.1%
TOTAL               333      2.978              1472.17     494.3 100.0%
(sum of kernel times; excludes launch gaps)
```

How to read this **correctly**, which is what Phase 4z was about:

1. **333 rows, 343 kernels.** The `attention` row is a two-kernel sum (§3.1d).
2. **The `GB/s` column is not comparable across rows** (§3.1b). The monotonic
   decline from 929 to 0.4 is the bracket plus size, not silicon.
3. **Every row below ~5 µs is quantised to a ~1 µs grid** (§3.1c). `router_top4`
   and `embed` both printing 4.10 is one clock tick, twice.
4. **TOTAL 2.978 ms is not the step.** It decomposes as 2.269 ms of step
   + ~0.46 ms of instrument (333 brackets × 1.38 µs) + ~0.21 ms because `prof`
   mode calls `forward()` directly instead of replaying the graph (343 nodes ×
   ~0.6 µs). 2.269 + 0.46 + 0.21 = 2.94 against the printed 2.978, i.e. the
   residual is 1.3% and is not yet itemised.
5. What *is* trustworthy: the **relative** sizes of the large streaming rows, and
   the fact that `moe_gate_up` at 24.6% is the bandwidth-critical kernel — which
   is measured independently by the 14.16 MB/call byte count and the `LK_ABLATE_SMALL`
   ablation, not by this table.

---

## 9. WHAT IS STILL OPEN, IN PRIORITY ORDER

1. **The fp16 range audit is NOT DONE.** This is the one genuine hole the author
   knows of. `LK_NORM_AUDIT` (default 0) would record min/max of `ds.x`, `|ds.y|`
   and `inv_rms` over the 558-token corpus; acceptance is
   `min(ds.x) >> 6.1035e-5` and `max(|ds.y|) << 65504`. Until it runs, §2.3 rests
   on an **assertion about range, not a measurement.** ~30 minutes.
2. **A persistent grid-stride GEMV.** Estimated +2–3% and **claimed bit-identical**,
   because it changes only the assignment of work to blocks, not the arithmetic
   within a row. **Attack the claim** (§12 A5) before building it — if block
   assignment changes the summation order inside a row, the claim is false.
3. **Small-with-small node fusion.** Estimated +2–3%.
4. **T3: fold the ShortConv conv into `shortconv_in`.** `shortconv_conv` is
   `<<<8,256>>>`, 0.026 ms over 30 layers.
5. **Further small-node work:** `swiglu`, `qk_norm_rope`, `argmax`.
6. **Batch-4 / M=4 — the big one.** The only remaining step change on one card.
   731–832 tok/s aggregate from the measured decomposition. Staged plan:
   (a) re-derive the M=4 byte model and verify `tools/g4c.py` does not double-count
   the non-expert ~45% read once per batch, or `lm_head`; (b) M=4 GEMV with shared
   weights; (c) expert-union dispatch validated against the route trace;
   (d) benchmark, then triggered prompt-lookup speculation.

   **Batching uses the INDEPENDENT row, not the consecutive row.** A batch holds B
   tokens from B *different* sequences, which share only the static marginal expert
   skew. Using the consecutive figure would be optimistic by exactly the amount
   that is easy to get wrong:

   ```
   B    consecutive        independent      uniform baseline
   4    11.22 (1.65x)      12.89 (1.51x)    14.56 (1.39x)
   8    17.91 (2.04x)      21.07 (1.83x)    25.81 (1.60x)
   ```
   Byte floor 971.7 MB/token at B=4 against **1447.9** at B=1. An earlier README
   revision claimed 2x from the consecutive figure; the honest figure is
   **1.35–1.45x**, corrected against the author's own interest. Only the launch
   toll fully amortises, so batch-4 lands at **1.47–1.88x** depending on whether
   the 0.444 ms execution term is latency-bound (constant) or work-bound (4x).

   **Known inconsistency #5, to resolve before building any of this:** the
   batching byte model uses **1447.9 MB/token at B=1** while §2.1 derives
   **1466.75 MB/token** for the same single token. The 18.85 MB gap is 1.3% and
   is not itemised. One of the two models is counting a category the other
   omits, and every batch-4 speedup figure above inherits whichever is wrong.

7. Re-measure the context table and the llama.cpp ratios on the current build (§7).
8. Promote the end-to-end nll table into Gate C2.
9. **Identify which norm kernel llama.cpp actually executes.** Phase 3 round 8
   proved it is *not* `rms_norm_f32<1024>` on this path (replicating that exactly
   regressed parity 66/66 → 38/66), so there is a fused or differently-configured
   norm in its real graph. Nothing in the engine depends on the answer, but the
   question was left open and should stay on the record.
10. **Six of the ten self-inconsistencies in §12.4 are documentation drift, not
    measurement error.** Reconciling them is cheap and would stop a reviewer
    spending their budget on stale numbers.

**What the author believes is not reachable, and wants challenged:**
**500 tok/s single-stream on ONE card at batch 1 for this weight format.** The
roofline is 652. The measured ceiling with this kernel structure is 488.9. Four
separate attacks on the overhead have been tried and three lost. Reaching 500
needs 0.266 ms of the 0.71 ms recoverable, i.e. ~50% of the small kernels, ~30% of
the launch toll and ~30% of the streaming ramp **simultaneously** — and every one
of those three has already defeated a direct attempt. The honest expectation for a
lot of grinding is **475–490**.

---

## 10. MEASURED DEAD — DO NOT RE-PROPOSE

Attack the measurement instead, and say what was wrong with the instrument.

* Tensor parallelism / any second-GPU scheme. Out of scope by user instruction;
  no P2P on this platform anyway.
* Speculative decoding, in both the draft-model and prompt-lookup forms
  (E 1.87 against a break-even of 2.11–2.41).
* Raising the power limit. Already at the 450 W card maximum, drawing 363 W, no
  throttle reason active, counter 0 µs.
* Occupancy tuning / wave quantisation. 1–2% across six configurations spanning
  0.76–3.05 waves.
* Clearing the launch toll by node deletion: 0.3–0.45 µs/node measured, 0.781 µs
  average toll. The arithmetic does not close.
* Fusing into `moe_gate_up`. **Three attempts, three losses** (FUSE_ROUTER
  0.09–0.39%, T4r 0.17%, T2 1.75%).
* Router MLP widening. Flat at `ROUTER_UNROLL` 16/32/64.
* Hoisting the four `moe_down` dots above their reductions. 12.77 vs 12.32 µs.
* Batching the attention V loads (no effect); transposing the KV cache (worse);
  1024-thread attention blocks (neutral).
* More prefetch anywhere. The magnitude is now known to be small and the mechanism
  is understood.
* The "`moe_down` is 100 GB/s slower than `moe_gate_up`" story. **Retracted by the
  author** — an artifact of the profiler's non-comparable `GB/s` column.

---

## 11. THE GENERALISED LAWS, WITH THEIR EVIDENCE

Stated as laws; each is an induction from 2–4 observations. These are the most
reusable output of the project and therefore the most worth attacking.

| # | law | evidence |
|---|---|---|
| a | **A kernel with one block cannot be fast, however many threads it has.** | router 25.7→4.5 µs, argmax 145→6 µs |
| b | **Rank kernels by work-per-thread, not by traffic share.** | the router carried 1.4% of the bytes and was the largest cost |
| c | **An ablation that changes WHERE memory is read is not measuring what it claims to.** | the +28.5%/+36.2% contamination, guarded to +6.0%/+10.8% |
| d | **A two-implementation comparison cannot say which is right.** | llama.cpp is 91x further from exact than the engine |
| e | **Measure the gate before building what it gates.** | G4b cancelled a week of speculation work |
| f | **Price the instrument before believing it.** | 1.38 µs/bracket; the retracted GB/s gap |
| g | **A barrier at the END of a kernel is cheap; at the START it gates the critical path.** | T4r |
| h | **Fusing into the bandwidth-critical kernel loses.** | T2, T4r, FUSE_ROUTER |
| i | **Deleting a graph node is worth 0.3–0.45 µs, not 1.5.** | T1, T4r |
| j | **Occupancy and wave quantisation are worth 1–2% on this workload.** | six configurations, 2.2% spread |
| k | **The logit-vector dump is the fastest discriminator for a numerical bug.** | two Phase 4y bugs found immediately |
| l | **Greedy parity is not a usable correctness gate when the model amplifies perturbations.** | 1e-8 → 0.724 at 16 tokens |

---

## 12. HOW TO AUDIT THIS

Adversarial audit. I am the author. I do not want agreement, encouragement,
restatement, or a summary of what I wrote. I want the errors.

1. Every claim gets a verdict: **VERIFIED / REFUTED / UNVERIFIABLE-AS-STATED**.
2. **VERIFIED requires you to show the arithmetic**, not assert that it sounds right.
3. **REFUTED requires either arithmetic that contradicts it, or a named
   physical/architectural fact that contradicts it.** "I doubt it" is not a refutation.
4. **UNVERIFIABLE-AS-STATED requires you to say what single measurement would
   settle it**, and roughly what it would cost to run.
5. Do not propose anything in §10. If you think one of those was measured *wrong*,
   attack the measurement and name the flaw in the instrument.
6. Rank everything by impact on the headline number, largest first.
7. Where §7 says a figure is stale, treat the live number as current and say so if
   you disagree about which is which.

### Required output format
1. A table of every claim you checked: `claim | verdict | your arithmetic or the
   contradicting fact`. Terse arithmetic.
2. **FINDINGS, ranked by impact.** For each: what is wrong, how you know, and the
   ONE measurement that would confirm or kill it.
3. Anything in §12's asks you can close.
4. **Any place where two of my own numbers are mutually inconsistent.** Ten are
   already listed, so do not spend your budget rediscovering them — find the ones
   I have *not* listed:
   (i) §3.1a the README's "~2.1 µs bracket" versus the measured 1.38 µs increment;
   (ii) §3.1d 333 profiler rows versus 343 graph nodes;
   (iii) §3.6 three irreconciled exchange rates for deleting a node
   (0.3–0.45 / 1.5 / 3.811 µs), plus 1.29 and 2.07 in the same record;
   (iv) §7 the README body quoting 417.4 tok/s and 509 nodes against its own
   header's 443.2 and 343;
   (v) the README's "4.27x vs the Phase-3 baseline" does not follow from its own
   `443.2 / 102.79 = 4.31` — the ratio was computed against the older 438.8;
   (vi) §9.6 the batching model's 1447.9 MB/token at B=1 versus §2.1's
   1466.75 MB/token for the same token;
   (vii) §6.3 **four different C2 figures for the same measurement** — README §C2
   `0.0222%`, README status list `0.0402%`, DIAGNOSIS Phase 4m `0.0402%`, live
   `0.0140%` — and three different oracle distances (`91x / 17x / 15.8x`);
   (viii) the full-suite greedy parity baseline appears as **242/825** (Phase 4),
   **180/825** (Phase 4, after the router fix) and **191/825** (Phase 4y, called
   "the baseline" for the pre-4y build). 242 → 180 is explained; **180 → 191 is
   not**, and the 191 figure is the one this document quotes;
   (ix) the pre-Phase-4z baseline is quoted as **438.8** in the README and Phase
   4z.6 but was measured as **438.31** inside Phase 4z itself;
   (x) the "before Phase 4z" byte model, the phase-4x budget (`2.398` stated vs
   `2.399` summed) and the reference RMS values for layers 21/22/final, which
   differ between Phase 3 rounds 3 and 6 while round 6 calls its table "unchanged
   from round 3".
5. For each law in §11, predict a specific experimental outcome that would
   falsify it if you believe it is false.
6. **One thing I should do next**, justified in bytes and microseconds, not
   adjectives.

Do not open with a summary of this document. Do not compliment it. Start with the
table.

### ASKS TO THE GROUP
* **A1 — AUDIT §2.3**, the scale-invariant Q8_1 norm split. Is
  `dot(inv*u) = inv*dot(u)` valid as stated given `ds.x`/`ds.y` are **fp16** and
  `qs` is int8? Is there a range or precision interaction I have missed?
* **A2 — AUDIT §3.1b**, the two-term law. Is fitting `time = a + bytes/BW` over 11
  kernels a legitimate way to characterise a kernel population, or circular? What
  would break it?
* **A3 — AUDIT §3.1c**, the 1.024 µs bracket quantum. 311 of 333 brackets landed
  exactly on 2.048/3.072/4.096 µs and the timer resolves 32 ns in isolation. Is a
  scheduling-timeslice explanation right, or is there a better one? And does the
  same quantum invalidate `bench_launch` part (7) (3.811 µs/norm node)?
* **A4 — AUDIT §9.1.** Can you construct an input that drives `ds.y` or `ds.x` out
  of fp16 range given the §1 architecture? This is the one hole I know of, and a
  plausible adversarial activation pattern is worth more than the audit itself.
* **A5 — AUDIT §9.2.** Is a persistent grid-stride GEMV really bit-identical for
  this reduction structure, or does block assignment change the summation order
  inside a row?
* **A6 — EXPLAIN §5/4z.6.** Why does `prefetch.global.L2` (+0.35%) beat a real
  `ld.global` (0.00%), four times over?
* **A7 — Identify "Ninfer."** A reviewer referenced borrowing techniques from
  something called Ninfer. I could not identify it (no web search in my
  environment; the GitHub search page returned only navigation chrome). Name the
  repo and the specific technique.
* **A8 — CHALLENGE §9's conclusion.** Is 500 tok/s single-stream on ONE card really
  unreachable at batch 1 for this format? If you believe it is reachable, give
  **bytes, microseconds and a kernel structure** — not an idea.
* **A9 — MOST IMPORTANT.** Tell me what is **wrong** in this document. I have
  retracted my own claims three times in this project and each retraction was
  worth more than the optimisation it cancelled. A refutation with arithmetic
  attached is the most valuable thing you can send.
