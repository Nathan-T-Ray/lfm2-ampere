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
| **measured batch-1 ceiling, current 343-node graph** | **488.9 tok/s** |
| batch-1 if the launch toll is removed entirely (§13.5, measured additive) | **500.5 tok/s** |
| batch-4 on the same one card | 682–827 tok/s (§13.3; supersedes 731–832) |
| batch-8 on the same one card | 819–1068 tok/s (§13.3) |
| single-stream with triggered speculation, `factual` | **695–849 tok/s** (§13.4) |
| single-stream with triggered speculation, 7-workload average | break-even, 0.96x of the bar (§13.4) |
| C2 accuracy, re-measured live | **0.0140%** from exact float64 against a 0.1% gate — **PASS**; llama.cpp is 0.2199% from the same reference |

Build reproducibility: **NOT byte-reproducible, and this took three attempts to
establish honestly.**

The claim used to read "`make engine` reproduces `build/engine` md5-identical
(`e3051b18b27b1fb08e0d1f288bed68a0`)". It does not. Then a second version of the
claim said the linker's 25-byte `.note.gnu.build-id` was the only cause and that
`-Wl,--build-id=none` fixed it, "verified by three identical rebuilds". **Both
were wrong**, and the second was worse, because it was "verified" by running
`make engine >/dev/null 2>&1` — and `-Wl,` is not nvcc syntax, so **every build
was failing, the binary was stale, and three identical hashes of an unchanged
file looked exactly like reproducibility.**

What is actually true, from two rebuilds whose builds were confirmed to have
succeeded:

```
  file size            989688   both
  host .text           BYTE-IDENTICAL   md5 676bef6064748e665f27f881ca21ec66
  .nv_fatbin (device)  differs by EXACTLY 1 byte out of 847000
  .strtab              differs by 4 bytes
  ------------------------------------------------
  whole-file md5       differs by 5 bytes out of 989688
  behaviour            tools/ab.py 7/7 bit-identical; C2 to the digit
```

So the compiled **code** is deterministic in substance — the host stub exactly,
the device code to within one byte — but the **file** is not byte-reproducible,
because something nvcc emits carries a per-invocation value. `-Xlinker
--build-id=none` (the correct spelling) removes the linker's note and reduces the
variance from 25 bytes to 5; it does not eliminate it, and no md5 claim should be
made.

> **The generalised law, and it is the most expensive lesson in this project:
> never silence a build when the build is the thing under test.** Hiding stderr
> turned a hard failure into a perfect-looking result, and the result was
> believed for a full round. Every verification in this document that pipes a
> build through `>/dev/null` or `2>&1 | grep` should be treated as unverified.

The behavioural claims do hold and are the ones that matter: 7/7 bit-identical
trajectories, and C2 reproduced to the digit. The FP16-audit hooks of §13.8
compile to *nothing* at `AUDIT=0`.

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
   **682–827 tok/s aggregate**, corrected in §13.2. Staged plan:
   (a) ~~re-derive the M=4 byte model~~ **DONE, §13.2 — the model was wrong in two
   places**; (b) M=4 GEMV with shared
   weights; (c) expert-union dispatch validated against the route trace;
   (d) benchmark, then triggered prompt-lookup speculation — **which §13.3 has now
   simulated in full and which pays on copy-heavy text only**.

   **Batching uses the INDEPENDENT row, not the consecutive row.** A batch holds B
   tokens from B *different* sequences, which share only the static marginal expert
   skew. Using the consecutive figure would be optimistic by exactly the amount
   that is easy to get wrong:

   ```
   B    consecutive        independent      uniform baseline
   4    11.22 (1.65x)      12.89 (1.51x)    14.56 (1.39x)
   8    17.91 (2.04x)      21.07 (1.83x)    25.81 (1.60x)
   ```
   Byte model corrected in §13.2. B=4 is **1.540–1.867x** — **682–827 tok/s** —
   against the corrected 1467.9 MB/token at B=1. An earlier README revision
   claimed 2x from the consecutive figure; the honest byte reduction is **1.24x**
   (k=4 independent reuse), corrected against the author's own interest. Only the
   launch toll fully amortises, so batch-4 lands at **1.540–1.867x** depending on
   how the 0.441 ms ramp+small term scales — now a **bracket** rather than a coin
   flip; see §13.2.

   ~~**Known inconsistency #5, to resolve before building any of this:** the
   batching byte model uses **1447.9 MB/token at B=1** while §2.1 derives
   **1466.75 MB/token** for the same single token. The 18.85 MB gap is 1.3% and
   is not itemised.~~ **RESOLVED, §13.2: it was a missing `router_dot` term**
   (`lk_prof_bytes` charges 524288 B/layer x 38 = 19.92 MB). The base now
   reconciles to within **0.077%** and `tools/lkbytes.py` asserts it.

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
**500 tok/s single-stream on ONE card at batch 1, *without changing the graph's
structure*.** The roofline is 652. The measured ceiling with the current 343-node
graph is 488.9, and it is reached by filling the small kernels — four separate
attacks on the overhead have been tried and three lost. Grinding inside the
existing graph lands at **475–490**.

**This was revised in §13.5, and the revision is upward.** Part (8) of
`bench_launch` measured the per-node toll and found it **additive**
(`busy_node = empty_toll + pure_transfer`), so the 0.268 ms is not already
overlapped. Removing the node boundaries entirely gives
`2.266 − 0.268 = 1.998 ms = 500.5 tok/s` on paper — which is a *structural*
change, not an optimisation, and is therefore not bounded by the 488.9 figure.

It remains unmeasured whether an intra-kernel handoff is cheaper than a kernel
boundary's 0.78 µs, and that single number decides whether the megakernel is
worth building at all.

---

## 10. MEASURED DEAD — DO NOT RE-PROPOSE

Attack the measurement instead, and say what was wrong with the instrument.

* Tensor parallelism / any second-GPU scheme. Out of scope by user instruction;
  no P2P on this platform anyway.
* Speculative decoding **as a general-purpose win**. §13.3 re-ran the sweep across
  the whole `minmatch` x `drafts` space, which the original verdict never did, and
  the position is now: **`factual` pays 1.57–1.92x (695–849 tok/s); the aggregate
  over 7 workloads is break-even at 0.96x of the bar.** Not dead — but confined to
  text the model is copying, and the binding constraint is first-draft precision
  (`p1`), not trigger rate.
* **A draft MODEL**, for this checkpoint specifically: viable arithmetic (the
  corrected M=2 bar means a drafter only has to cost < 0.26 of a step) but
  **no sibling checkpoint exists** — the QAD weights do not match public
  `LiquidAI/LFM2.5-24B-A2B` (embedding correlation −0.036), so `p1` for any
  public small model is unknown and probably poor.
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
4. **Any place where two of my own numbers are mutually inconsistent.** Nine are
   already listed (one closed in §13.2), so do not spend your budget
   rediscovering them — find the ones I have *not* listed:
   (i) §3.1a the README's "~2.1 µs bracket" versus the measured 1.38 µs increment;
   (ii) §3.1d 333 profiler rows versus 343 graph nodes;
   (iii) §3.6 three irreconciled exchange rates for deleting a node
   (0.3–0.45 / 1.5 / 3.811 µs), plus 1.29 and 2.07 in the same record;
   (iv) §7 the README body quoting 417.4 tok/s and 509 nodes against its own
   header's 443.2 and 343;
   (v) the README's "4.27x vs the Phase-3 baseline" does not follow from its own
   `443.2 / 102.79 = 4.31` — the ratio was computed against the older 438.8;
   (vi) ~~§9.6 the batching model's 1447.9 MB/token at B=1 versus §2.1's
   1466.75 MB/token for the same token~~ — **RESOLVED in §13.2: a missing
   `router_dot` term, 19.92 MB. This is now nine, not ten.**
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

> **§13 was written after this section and supersedes parts of it.** A2 (the
> two-term law), A5 (persistent GEMV bit-identity) and A8 (is 500 unreachable)
> have all moved: §13.3 corrects the bars, §13.5 measures the toll as additive
> and puts 500.5 tok/s on the table, and §13.6 gives the block-identity rule that
> A5 was asking for. Read §13 before answering A2, A5 or A8.

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

---

## 13. PHASE 4z.7 — THE CORRECTIONS, AND THREE OF MY OWN ERRORS

This section exists because an external reviewer re-derived a model this
document had already refuted, and chasing it down found **three errors in my own
work**: two in the byte model that every multi-token figure in §9.6 inherited,
and one hypothesis I had written into a draft brief. All three are recorded here
in the same style as the earlier retractions.

The tools now live in one place, `tools/lkbytes.py`, so the two consumers
(`tools/g4b.py`, `tools/g4c.py`) cannot drift apart again. It self-tests, and the
self-test asserts the invariant `ratio(1) == 1.000`.

### 13.1 ERROR 1 — `lm_head` was charged one time too many per pass

`tools/g4b.py` and `tools/g4c.py` both computed

```python
per_pass = K * LM_HEAD_MB + len(MOE_LAYERS) * U * EXPERT_MB + FIXED_MB
```

and `DIAGNOSIS.md`, `tools/lookup.py` and §9.6 all inherited it. `lm_head` is
215.04 MB — **14.7% of a single-token step**, second only to the MoE experts — so
this was the most expensive line item to get wrong.

A verification pass over `K` positions produces `K` logit vectors, but the
**first** is the vector the previous step already computed to emit the token that
seeds the pass. It is in hand. So the pass computes `K-1`:

| K | `lm_head` charged | needed | phantom MB/pass |
|---|---|---|---|
| 1 | 1 | 1 | 0 |
| 2 | 2 | 1 | **215.04** |
| 3 | 3 | 2 | **215.04** |
| 4 | 4 | 3 | **215.04** |

It is a **fixed** 215.04 MB, independent of `K`. That is exactly why it survived
this long: every headline number in this document comes from `K=1`, where the
charge is correct.

Batching is **not** the same formula. `B` independent sequences each need their
own logits to continue, so batching runs `lm_head` `B` times with no redundancy.
`lkbytes.py` exposes `lm_head_evals_spec` and `lm_head_evals_batch` as separate
functions precisely so the two cannot be conflated again.

### 13.2 ERROR 2 — the router term was missing, and it closes inconsistency #5

`lk_prof_bytes` in `src/engine.cu` contains `{"router_dot", 524288.0}`. Over 38
MoE layers that is **19.92 MB**. `FIXED_MB = 283.0 + 59.0 + 81.0 + 3.0` never had
a router row.

That 19.92 MB is the whole of the unreconciled **18.85 MB** gap of §9.6 — known
inconsistency #5, carried unresolved for two phases. The model base now
reconciles:

```
model base 1467.88 MB  vs measured 1466.75 MB   -> residual +0.077%
```

and the residual is the rounding in `283.0` (true 283.85, from
`30 x (shortcout_in 7077888 + conv 24576 + out 2359296)`). `tools/lkbytes.py`
asserts the residual is under 0.1%.

**Inconsistency count: ten, now nine.**

### 13.3 The break-even bracket, and two figures the document had both published

`T(1) = 2.266 ms` decomposes into four terms, and they do not all scale the same
way with the number of tokens in a pass:

```
1.557 ms  wire   -- per BYTE, scales with R = pass_MB / base_MB
0.220 ms  ramp   -- per KERNEL per pass; kernels do not get bigger, but grids do
0.268 ms  toll   -- per NODE, and the node count barely moves -> CONSTANT
0.221 ms  small  -- per TOKEN of work, plus dependency stalls which are not
```

Rather than pick the flattering end:

```
lower = R*WIRE + TOLL + RAMP + SMALL
upper = R*(WIRE+RAMP) + TOLL + SMALL*ntok
```

**The document had already published both ends without noticing.**
`DIAGNOSIS.md:1419` gives M=4 as `1.98x .. 2.56x`; `DIAGNOSIS.md:1675-1683` gives
M=4 as `2.11x .. 2.41x`. They are not in conflict — they are the two ends of a
bracket nobody wrote down. `lkbytes._selftest` now asserts that the legacy model
reproduces both, which makes the bracket a **check** rather than an excuse.

Corrected bars (`bar` = break-even `E[tokens/pass]`):

| M | drafts | lm_head | U_M | pass MB | bar corrected | bar legacy | change |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 1 | 4.00 | 1467.9 | 1.001 | 0.991 | — |
| 2 | 1 | 1 | 6.73 | 2019.4 | **1.259 – 1.393** | 1.350 – 1.497 | −6.8% |
| 3 | 2 | 2 | 9.12 | 2716.1 | **1.585 – 1.863** | 1.677 – 1.967 | −5.5% |
| 4 | 3 | 3 | 11.22 | 3355.2 | **1.885 – 2.302** | 1.976 – 2.407 | −4.6% |

Batching (independent row, `lm_head` x B): B=2 **1.270–1.411x**, B=4
**1.540–1.867x = 682–827 tok/s**, B=8 **1.848–2.410x = 819–1068 tok/s**. This
supersedes the 731–832 in §0 and §9.6.

### 13.4 The speculation verdict, re-measured across the space that was never swept

`tools/lookup.py` hardcoded `ratio = 2.11, 2.41` and never varied it, so the
recorded verdict *"speculative decoding … measured dead"* was taken at **one
point in a two-dimensional space** (`minmatch` x `drafts`). M=4 is the **worst**
point on the curve: the bar rises ~0.3x per extra draft while the marginal
acceptance decays.

Now swept, `ngen=384`, 7 cases, log at `ref/lookup_sweep.log`:

| config | trigger | E | bar | E/bar | tok/s | verdict |
|---|---|---|---|---|---|---|
| **M=4 minmatch=2, `factual`** | 88.6% | 3.61 | 1.885–2.302 | **1.57–1.92** | **695–849** | **PAYS** |
| M=3 minmatch=2, `factual` | 88.6% | 2.75 | 1.585–1.863 | 1.47–1.73 | 653–768 | PAYS |
| M=2 minmatch=2, `factual` | 88.6% | 1.88 | 1.259–1.393 | 1.35–1.49 | 597–661 | PAYS |
| M=4 minmatch=2, ALL | — | **1.873** | 1.885–2.302 | 0.81 | 360–440 | DOES NOT PAY |
| M=2 minmatch=2, ALL | — | 1.34 | 1.259–1.393 | 0.96 | 427–473 | BREAK-EVEN |

**The last-but-one row is the whole story.** `E = 1.873` against the legacy bar
`1.976–2.407` was 5.5–21.9% short — clearly dead. Against the corrected bar
`1.885–2.302` it is **0.6% short of the optimistic end**. The verdict did not
flip, but it moved from *comfortably dead* to *on the line*, and the case
breakdown shows the win is concentrated where the text is copied.

**The binding constraint is precision, not trigger rate.** `code` at
`minmatch=1` triggers **67.5%** of the time and still gets `E = 1.27`, because
its continuation is right only **40%** of the time. Per-case first-draft
accuracy, from `E = 1 + trigger * p1`:

```
factual 99.3% | reasoning 41.3% | code 40.0% | agentic 40.0%
              | prose 36.0%     | math 31.8%
```

To clear 500 tok/s needs `p1 > 0.468` at M=2 or `p1 > 0.418` at M=4 with a 90%
trigger. **Prompt-lookup delivers ~0.40. It misses on a threshold, not a
mechanism** — which is why the next question is a better predictor, not a bigger
M.

The **monotone cross-check** passes: `E(K=4)=1.873` bounds `E(K=2) >= 1.291`
arithmetically, and the simulation gives `1.344` and `1.625`, both above it. A
bug in the simulation would have shown up here.

**Caveat, stated plainly:** `E` is exact (lossless greedy speculation can be
simulated offline), but **the bars are a model, not a measurement** — no M-pass
kernel exists. The build must re-measure the bar before the payoff is claimed.

### 13.5 ERROR 3 — my own hypothesis that the launch toll was already overlapped

A draft of the external brief argued:

> *the bracket-corrected profiler sum (2.978 − 0.351 = 2.627 ms) exceeds the
> 2.269 ms wall clock, which cannot happen unless kernels overlap, so much of the
> 0.268 ms toll may already be hidden and a megakernel would recover far less.*

**Tested and refuted.** `tests/bench_launch.cu` part (8): `N` nodes, each reading
a distinct cold slab, **total bytes held constant at 512 MB**, grid fixed at
768x256 so only `N` varies. Log at `ref/bench_launch.log`.

```
    N    MB/node     us/node     pure us    overhead      GB/s
   16      32.00      40.729      35.628       5.101     823.8
   32      16.00      20.954      17.814       3.140     800.7
   64       8.00      11.076       8.907       2.169     757.3
  128       4.00       6.250       4.453       1.797     671.0
  256       2.00       3.562       2.227       1.336     588.7
  512       1.00       2.361       1.113       1.248     444.0
```

The decisive check is `busy_node ?= empty_node_toll + pure_transfer`:

| N | empty toll | pure | sum | measured | excess (= ramp) |
|---|---|---|---|---|---|
| 512 | 1.154 | 1.113 | 2.267 | **2.361** | +0.094 |
| 256 | 1.154 | 2.227 | 3.381 | **3.562** | +0.181 |
| 128 | 1.154 | 4.453 | 5.607 | **6.250** | +0.643 |

**The toll is additive, not hidden.** My hypothesis is withdrawn. Part (8b) also
shows *why* the argument from the profiler sum was unsound: over 343 **uniform**
1.5 MB nodes the instrument gives `sum(brackets) = 1854 µs` against a `2323 µs`
wall — it **under**-counts there and **over**-counts on the engine's mixed
small/large load. **The instrument's sign depends on the node mix, so the
engine's 2.978 vs 2.269 comparison cannot carry the weight I put on it.** That is
now inconsistency (xi), found by my own probe.

**Consequence:** removing only the 0.268 ms toll gives `2.266 − 0.268 = 1.998 ms`
= **500.5 tok/s**. This does **not** contradict the 488.9 ceiling of §4 — 488.9
was measured by filling the small kernels while *keeping* the 343 nodes; 500.5
comes from removing the node boundaries, which is a structural change rather than
an optimisation inside the existing graph.

**The one thing still unmeasured, and it is the whole risk:** whether an
intra-kernel handoff costs less than a kernel boundary's 0.78 µs. If a
band-local producer/consumer flag plus `__threadfence` costs 0.8 µs, a megakernel
buys nothing. That is a microbenchmark, not a build, and it must be written first.
And the design risk is already measured: this project's own law is that placing a
barrier in front of a bandwidth-critical kernel loses (`FUSE_ROUTER` 0.09–0.39%,
`T4r` 0.17%, `T2` 1.75% — three attempts, three losses).

**Second finding from the same table:** achieved bandwidth collapses from
**823.8 GB/s at 32 MB/node to 444.0 GB/s at 1 MB/node**, with the engine's
overall 650 GB/s sitting inside that range. That is the "0.220 ms streaming size
inefficiency" seen directly, and it suggests **shaping kernels larger is a lever
nobody has attacked at the kernel level.**

### 13.6 The block-identity rule (needed before any M-pass build)

From `k_moe_down_accum`:

```c
const int row = blockIdx.x * 32 + tw;          /* ONE OUTPUT ROW PER WARP */
for (int k = 0; k < LK_TOP_K; ++k) {
    float acc = 0.0f;
    for (int j = lane; j < bpr; j += 32)       /* lane-strided accumulator */
        acc += dot_q4_0_q8_1(qrow, drow, j, xk + j);
    a += d_w[k] * warp_sum(acc);               /* warp tree, then k = 0..3 */
}
```

The entire reduction order for a row lives **inside one warp**. Therefore:

> **Which warp and which block computes a row is free to change. The lane loop,
> the `warp_sum` tree, the `k = 0..3` expert order, and the per-token Q8_1 block
> ownership are not.**

That is what makes an expert-union M-pass possible at all without breaking
bit-identity — and `tools/ab.py` gives a 7/7 check on every build. **Any M-pass
kernel that is not 7/7 bit-identical is not a lossless speculative verifier,
whatever else it is.** Note this constraint binds *speculation*; batching has no
such requirement, which is why batching is the easier of the two to build.

### 13.7 New open items from this phase

1. **Price the intra-kernel handoff** (§13.5). Gates the megakernel. Cheap.
2. **Fix or fold `tools/lklookup`'s trajectory cache** — done in this phase, but
   note the key must include `ngen`; a 64-token run poisoned a 384-token one
   during development.
3. **Shape the kernels larger** (§13.5 second finding). The 823.8 → 444.0 GB/s
   collapse across node sizes is measured and unexploited.
4. **Find a better single-token predictor** (§13.4). The bar is `p1 ≈ 0.42–0.47`;
   prompt-lookup gives 0.40. A drafter costing < 0.26 of a step is now viable in
   principle, but **no QAD sibling checkpoint exists**, so `p1` for any public
   small model is unknown and probably poor.

### 13.8 THE FP16 RANGE AUDIT (open item §9.1, carried since Phase 4y) — RUN

The one genuine reviewer hole. Phase 4y carries a scale invariance through
`block_q8_1`, whose `ds.x` and `ds.y` are **fp16**; the argument is exact in fp32
but only holds while both fields stay in range. Now instrumented:
`make AUDIT=1 engine` records min/max/mean of `d`, max `|SUM|`, and **counts the
blocks below the fp16 normal floor** — because a range alone cannot say whether a
subnormal excursion matters, and the count can.

Seven cases, `ngen=192`:

```
                                 range across the 7 cases
  ds.y = SUM   max|.|            3.9e+01 .. 9.9e+01     headroom 660 .. 1661x
  ds.x = d     min               1.61e-07 .. 1.83e-07   headroom above the
                                                        subnormal floor 2.7 .. 3.1x
  ds.x = d     mean              ~1.01e-03 .. 1.11e-03  ~17x ABOVE the normal floor
  d below the fp16 normal floor  13.73% .. 15.19% of blocks
  d below 1e-6 (<=17 quanta)     0.095% .. 0.109% of blocks
```

**Answer: the overflow concern was unfounded and the subnormal concern is real
but bounded.**

* `SUM` never approaches 65504 — a 660x margin at worst. That half of the hole is
  closed outright.
* `d` **does enter the fp16 subnormal range, for ~14% of blocks**, on every case.
  This was not predicted; the expectation was "safe by a wide margin" and that
  expectation was **wrong for `d`**. It never reaches zero (2.7x above the
  subnormal floor), so it stays representable — but below the normal floor
  `inv*fp16(d)` and `fp16(inv*d)` diverge, so the invariance is *approximate*
  there rather than exact.
* Those blocks are small by construction: `d = amax/127`, so `d ~ 1.7e-7` means
  `amax ~ 2.2e-5`, roughly 1000x below the typical block (`mean d ~ 1.1e-3`).
  Their weighted contribution to the induced error is
  `0.14 x 1e-3 x 0.17 ~ 2.4e-5 = 0.0024%` — two orders below the 0.1% C2 gate and
  consistent with the measured **0.0140%**.

**Stated honestly:** the 0.0024% is *estimated from* the measured distribution,
not itself measured. The measured facts are the 14%, the 2.8x subnormal headroom
and the 1.07e-3 mean. An audit that measured the induced error directly would
still be worth having, and is the sharpened form of A6.

### 13.9 THE INTRA-KERNEL HANDOFF, MEASURED — THE MEGAKERNEL IS JUSTIFIED

§13.5 established that the per-node toll is **additive**, so the 0.268 ms of
boundaries is real and unfilled. But a megakernel does not delete the
dependency, it *moves* it: a kernel boundary becomes an L2 flag plus a
`__threadfence()`. Nothing had priced that. `tests/bench_megakernel.cu` does:
S stages x 504 blocks (all resident, from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`),
identical traffic and identical indexing across five variants, so the only
difference is *how stage s is ordered against stage s-1*.

```
 slab     S   MB tot    graph   m1 BASE   m2 self  m4 cross  m3 grid |  boundary  h cross  gridbar
 8 KB   130      512    784.0     583.3     595.6     614.7   596.3 |     1.544    0.242    0.100
32 KB    32      504    619.2     567.2     568.7     571.4   569.0 |     1.625    0.131    0.056
128 KB   16     1008   1155.1    1122.6    1122.0    1123.3  1124.0 |     2.027    0.043    0.085
512 KB   16     4032   4544.2    4477.8    4474.9    4475.9  4475.2 |     4.147   -0.120   -0.163
                                                          (us, per stage, vs mode 1)
```

**A kernel boundary costs 1.54–2.03 µs per stage. A real cross-block handoff
costs 0.13–0.24 µs.** The handoff is **6–12x cheaper** than the boundary it would
replace. (The 512 KB row is too noisy to use — the graph spread was 136 µs on a
4.5 ms run — and is shown only for completeness.)

**Two traps this instrument only avoided because a first version was run and was
wrong**, and both are worth recording:

1. **A "no synchronisation at all" control is not a valid baseline.** The first
   version compared against a variant with no flag write, and that control
   measured *slower* than the treatment by up to 4% — it differed in something
   other than the thing being measured. The baseline is now mode 1, which is
   instruction-identical to the treatment except for the spin.
2. **A block waiting on its OWN flag never waits.** It set that flag itself one
   stage earlier, so the spin falls straight through. `m2 self` measures
   0.095/0.046/0.0 µs — i.e. nothing — which is exactly right and is why mode 4
   waits on *another* block's flag.

**What it means for the estimate.** The engine's own toll average is
0.781 µs/node:

```
  against the engine's own toll : 343 x (0.781 - 0.15) = 0.216 ms  ->  489 tok/s
  against the busy-graph boundary: 343 x (1.54  - 0.15) = 0.477 ms  ->  553 tok/s
```

**489–553 tok/s, straddling 500.** Both ends are soft in the same direction: the
synthetic workload has no straggler skew, so both the boundary *and* the handoff
here are **lower bounds** — in the engine a consumer waits on a producer doing
different work.

> **VERDICT: superseded by §13.13, which BUILT it and refuted it.** The
> paragraph below was the conclusion at the time and is kept for the record.
>
> ~~**VERDICT: the megakernel premise survives, and designing it is justified.**~~
> The toll alone lands at 489–553; clearing 500 robustly needs the persistent
> kernel to also hide part of the 0.221 ms of small-kernel execution — which is
> precisely what a single persistent kernel makes possible and a graph cannot.
> The measured design risk is unchanged and real: this project's own law is that
> a barrier in front of a bandwidth-critical kernel loses (`FUSE_ROUTER`
> 0.09–0.39%, `T4r` 0.17%, `T2` 1.75% — three attempts, three losses).

### 13.10 THE SKEW PRE-TEST — THE MEGAKERNEL PREMISE SURVIVES IMBALANCE

§13.9 measured a boundary at 1.54–2.03 µs and a grid barrier at 0.06–0.10 µs,
but with **every block doing identical work**. That is not the engine. A grid
barrier waits for the *slowest* block; a graph can dispatch the next kernel's
blocks onto SMs freed by the previous kernel's early finishers, i.e. the graph
**overlaps stragglers and a barrier cannot**. That is the one mechanism that
could have erased the 6–12× advantage, so it was measured before any porting.

Block `b` is given `1 + b % K` units of work, with the slab size normalised so
**total bytes stay constant at 504 MB** across all `K`:

```
  K  spread   MB tot     graph   m1 BASE   m3 grid |  boundary   gridbar  graph/base
  1      1x      504     620.2     566.3     569.7 |     1.685     0.107      1.095
  2      2x      504     712.4     638.3     647.1 |     2.315     0.274      1.116
  4      4x      504     774.5     713.0     723.6 |     1.920     0.331      1.086
  8      8x      504     769.4     714.4     725.3 |     1.717     0.341      1.077
                                                    (us, per stage, vs mode 1)
```

**The boundary cost does not collapse with skew.** It stays at 1.7–2.3 µs across
an 8× spread in per-block work, and the graph costs 7.7–11.6% more than the
un-ordered persistent control at *every* skew. The grid barrier does rise with
skew — 0.107 → 0.341 µs, exactly the wait-for-the-slowest penalty — but it rises
to a level still **5× below the boundary it replaces**.

> **VERDICT: superseded by §13.13.** The skew result holds — the boundary does
> not collapse under imbalance — but the boundary figure it was compared against
> was itself a synthetic artifact. ~~**The megakernel premise survives skew.**~~ Net of the barrier,
> 1.72 − 0.34 = **1.38 µs per stage at K=8**, i.e. 343 × 1.38 = 0.473 ms, or
> `2.266 − 0.473 = 1.793 ms = 555 tok/s`. The honest range stays **489–553**
> because the engine's own 0.268 ms toll estimate is itself a subtraction, and
> that has not changed.

The skew sweep also re-confirms the *balanced* numbers from §13.9 on a second,
independently normalised dataset: boundary 1.685 vs the earlier 1.625 at the
same 32 KB slab.

### 13.11 ERROR 4 — THE REPRODUCIBILITY CLAIM, RETRACTED TWICE, AND THE LAW IT TAUGHT

Recorded here because it is the most expensive mistake in the project and the
least visible.

The sequence, in full:

1. §0 claimed `make engine` reproduced `e3051b18` md5-identical. **False.** The
   check passed because `make` reused a cached `kernels.o` and never relinked.
2. That was diagnosed as the linker's 25-byte `.note.gnu.build-id`, and
   `-Wl,--build-id=none` was added to `LDFLAGS` as the fix, "verified" by three
   identical rebuilds. **Also false.** `-Wl,` is not nvcc syntax; **every CUDA
   build in the Makefile was failing with `nvcc fatal: Unknown option`**, and
   because the verification ran `make engine >/dev/null 2>&1` the failures were
   invisible. Three identical hashes of a binary that never changed looked
   exactly like reproducibility.
3. Corrected to `-Xlinker --build-id=none` — which does build — and the truth
   turned out to be neither of the above: the file differs by **5 bytes out of
   989688** (1 inside `.nv_fatbin`, 4 inside `.strtab`), with the host `.text`
   byte-identical. `--build-id=none` reduces the variance from 25 bytes to 5 and
   does not eliminate it. **No md5 reproducibility claim is made.**

Two laws, both earned:

> **Never silence a build when the build is the thing under test.** A piped
> `>/dev/null` converts a hard failure into a perfect-looking result. Every
> verification in this document that hides a build's output should be treated as
> unverified — and one of them was wrong for a full round.

> **For a CUDA binary, `.text` is the host stub. The device code is in
> `.nv_fatbin`.** The first section-level analysis checked `.text`, found it
> identical, and concluded the code was deterministic — while the section that
> actually contains the kernels was never compared. It was only found by
> extracting and hashing `.nv_fatbin` directly.

### 13.12 G5, THE OCCUPANCY GATE — PASSED. THE MEGAKERNEL IS BUILDABLE.

The go/no-go for the whole persistent-kernel plan is whether a fused kernel can
hold enough blocks resident to saturate DRAM, because a persistent kernel **must**
have every block resident or its grid barrier deadlocks. Register pressure is the
binding constraint, and a fused kernel takes the **max over every stage it
contains** — so one register-hungry stage poisons all of them.

`ptxas -v` over `src/kernels.cu` (no `cuobjdump` in this CUDA install):

```
   k_router_dot            127 regs      <-- LIVE, and the outlier
   k_router_dot_top4       128 regs      retired, never launched
   k_attention_decode       94 regs   34048 B smem   serial fallback, DEAD PATH
   k_rmsnorm_q8_1           48 regs    1024 B smem
   k_gemv_q6_K              48 regs
   k_moe_down_accum         41 regs     256 B smem
   k_gemv_q4_0_norm         41 regs    1072 B smem
   k_gemv_q6_K_dp4a         40 regs
   k_gemv_q4_0{,_3,_exp2}   39 regs
   k_attn_split             39 regs    8192 B smem   <-- largest live smem
   k_gemv_q4_0_prod         37 regs
   k_swiglu / k_shortconv   24 / 28 regs
   everything else          <= 30 regs
```

Occupancy at 256 threads/block (65536 registers per SM, 1536-thread ceiling):

| max regs in the fused kernel | threads/SM | blocks/SM | blocks total |
|---|---|---|---|
| 127 (`router_dot` inlined) | 516 | 2 | 168 |
| 48 (everything else) | 1365 | 5 | **420** |
| ≤ 42 (forced with `__launch_bounds__(256,6)`) | 1536 | 6 | **504** |

**Verdict: PASS, with one exclusion.**

* **`k_attention_decode` must be excluded by `#if`.** 94 registers *and* 34 KB of
  static smem (`s_score[LK_ATTN_MAX_SEQ]`), and it is only reached when
  `LK_ATTN_SPLIT` is disabled. This confirms §1.3 by measurement rather than by
  reading.
* **`k_router_dot` should stay a graph node**, at least for the first version.
  One stage at 127 registers would cut the whole kernel to 168 blocks — a third
  of the residency — and it is **38 nodes out of 343** (one per MoE layer), so
  keeping it costs 38 boundaries and buys back 252 blocks. That is a good trade
  and it can be revisited later by forcing `__launch_bounds__` and accepting
  spills in a stage that is 0.52 MB of traffic.
* Without it, the ceiling is **48 registers → 5 blocks/SM → 420 blocks =
  107 520 threads**, which is 83% of the 504 blocks that `bench_megakernel`
  measured at 878 GB/s. The `__launch_bounds__(256,6)` variant reaching 504 is
  worth trying second.
* Shared memory is **not** a constraint once `k_attention_decode` is out: the
  largest live stage is `k_attn_split` at 8 KB, and 6 blocks × 8 KB = 48 KB
  against ~100 KB usable.

**This is the last structural unknown.** The premise is measured (§13.9), it
survives imbalance (§13.10), and the residency budget closes here. Porting can
proceed.

### 13.13 ERROR 5 — THE MEGAKERNEL IS REFUTED, ON A REAL STAGE, BY BUILDING IT

§13.9 measured a kernel boundary at 1.54–2.03 µs and a grid barrier at
0.06–0.10 µs, and §13.10 showed the gap survives imbalance. On that basis Phase 5
was planned. **Both numbers were wrong, and the synthetic instrument was missing
the one thing that mattered.**

`tests/test_megakernel.cu` ports the most-executed shape in the model (Q4_0 x Q8_1
GEMV) into the persistent form — `src/megakernel.cuh`, grid-stride over rows, grid
barrier between stages — and compares it against the shipped kernel with
`memcmp`. Cold staging: 31–130 separate weight slabs so nothing is L2-resident.

```
                                                 2048 rows    12288 rows
  separate launches                              4.881        17.190
  one launch + N grid barriers                   5.169        18.544
  one launch, NO barrier (pure execution)        2.606        16.364
  ------------------------------------------------ us per stage
  LAUNCH BOUNDARY   (ref - nobar)/N              2.275         0.826
  GRID BARRIER      (bar - nobar)/N              2.563         2.180
  NET               (ref - bar)/N               -0.288        -1.354
  --> ratio                                      0.948x        0.929x
```

**The grid barrier costs 2.14–2.18 µs. The kernel launch it replaces costs
0.78–0.83 µs. Replacing launches with barriers makes the engine 6–8% SLOWER.**

Two things fall out of this, and both are corrections to §13.9:

1. **The launch boundary in a real dependent chain is 0.78–0.83 µs — not 1.54–2.03.**
   The large-shape figure, 0.826 µs, reproduces the engine's own independently
   measured 0.781 µs/node toll (`bench_shapes` mode 0) to within 6%. §13.9's
   1.54–2.03 µs was an artifact of the synthetic graph, and the document's own
   0.268 ms was right all along.
2. **The barrier is not 0.06–0.10 µs — it is ~2.1 µs, 20x more.** The synthetic
   test's stages were **independent**: stage s read slab s, so the barrier ordered
   *nothing* and the blocks merely touched the same counter. Here every stage
   writes `y` and the next stage reads it, so the barrier carries a real
   producer→consumer ordering through global memory. **A cross-block handoff that
   has nothing to order is free; one that has to order 12 288 rows of stores is
   not.** That was the missing ingredient, and it is the whole cost.

Falsified hypotheses, recorded so they are not retried:

* **A contended spin.** The first version spun on `atomicAdd(&bar[1],0)`; that was
  fixed to a volatile load with `__nanosleep` backoff, and the barrier stayed at
  ~2.1 µs.
* **The redundant second `__threadfence()`.** Removing it moved the barrier from
  2.180 to 2.140 µs — 0.04 µs. Not the cost.

> **VERDICT: the megakernel is CLOSED.** Not "risky" — measured, on the real
> stage shape, bit-identical and slower. This is also the mechanism behind the
> three earlier barrier losses (`FUSE_ROUTER` 0.09–0.39%, `T4r` 0.17%, `T2`
> 1.75%): the project's own law that *a barrier in front of a bandwidth-critical
> kernel loses* now has a number attached, and it was never about the barrier
> being in front of the kernel. It is about what the barrier has to order.
>
> **SUPERSEDED IN PART BY §13.14.** This section closed the megakernel on ONE
> barrier implementation, in ONE regime, and generalised. §13.14 re-opens it,
> finds three genuine confounds in this instrument, and re-measures with 14
> barrier implementations and a cadence sweep. The verdict *CLOSED* survives and
> is now far better supported; **the explanation above does not** — padding, PTX
> acquire/release and a two-level arrival tree all leave the barrier at 2.5 µs,
> so it is not the ordering of the stores. Read §13.14 for the corrected
> mechanism and for the lever it redirects the effort to.

**What survives from Phase 5, and it is not nothing.** The porting rule of §13.6 is
**verified on real hardware**: 2048/2048 and 12288/12288 rows bit-identical, on
both the plain and the Phase-4y scale path. Any future fusion or multi-token
kernel can use it with confidence, and `tools/ab.py` remains the gate. The
`LK_GEMV_ROWS`-style grid-stride rewrite is safe; the persistence is not.

**What this leaves for single-stream >500:** nothing that is currently measured.
The 443 tok/s wall stands, the batch-1 ceiling inside the graph stands at 488.9,
and the two remaining levers are **batch-4 (682–827 tok/s aggregate)** and
**triggered speculation (695–849 tok/s on copy-heavy text)**, both of which were
measured before this phase and are unaffected by it.

### 13.14 CAMPAIGN 2 — THE MEGAKERNEL IS RE-OPENED, GIVEN A FAIR TRIAL, AND REFUTED FOR A DIFFERENT REASON

§13.13 closed the megakernel on a single barrier implementation, in a single
regime, and generalised. That is not enough to close a design, and the
generalisation was challenged: *"did we just try a single implementation of the
megakernel and then scrap it?"*

We had. §13.13 measured **one** barrier — a monotonic counter pair, two fences,
`__nanosleep(64)`, in a chain of large homogeneous GEMV stages. Three confounds
were visible in that instrument on re-reading, and all three were real defects:

1. **False sharing.** `d_bar` was `cudaMalloc(&d_bar, 2 * sizeof(int))` = **8
   bytes**, so the arrival counter, the release flag and every poller's volatile
   load occupied **one 128-byte L2 line**. Every arrival RMW invalidated every
   poller.
2. **A hidden launch inside the timing.** The monotonic scheme needs the counters
   reset between replays, and the harness did it with `cudaMemsetAsync` **inside
   the timed region** — for the barrier mode only, never for the separate-launch
   mode. Part of the "barrier cost" was therefore a memset kernel launch, i.e.
   precisely the toll the megakernel exists to remove.
3. **One cadence, one size.** Every stage paid a barrier, and every stage was a
   large balanced GEMV — the regime where a kernel boundary is *cheapest* and a
   barrier is *hardest*. The claim was about the 158 **small** nodes of the 343,
   and those were never measured.

`src/mkbar.cuh` + `tests/bench_mkbar.cu` give the barrier a fair trial. Fourteen
implementations, every one gated by the same `memcmp` bit-identity test, with the
factors moved one at a time:

| V | barrier | isolates |
|---|---|---|
| V0 | legacy published (unpadded, 2 fences, ns=64) | the §13.13 baseline |
| V1 | unpadded + post-spin acquire | the acquire |
| V2 | **padded, 2 fences, ns=64** | **false sharing** |
| V3–V6 | padded, ns = 0 / 32 / 128 / 256 | backoff |
| V7–V9 | padded, PTX `atom.release.gpu` / `st.release.gpu` / `ld.acquire.gpu` | fences |
| V10–V13 | two-level arrival tree, G = 4 / 8 / 16 / 84 | arrival serialization |

The trees exist because after V0–V9 landed within 4% of each other the remaining
candidate was **true** sharing — not two addresses on one line, but 504 blocks
incrementing *the same* address, ~2–4 ns of L2 serialization each, i.e. 1.0–2.0 µs
of the 2.5 µs. The tree splits the grid by `blockIdx % G`, so the deepest serial
chain is `nb/G + G` (71 at G=8) instead of 504, and it deletes the release
broadcast entirely: consumers poll the root arrival count directly, so the
rendezvous completes in one L2 hop rather than two.

**RESULT 1 — every variant is bit-identical. 14/14, 12288/12288 rows, both scale
paths. The porting rule of §13.6 holds against every synchronization primitive
tried.**

**RESULT 2 — and none of them is faster.** Large shape, 12288 rows, 31 stages of
14.16 MB, counters reset **outside** the timing:

```
  separate launches (graph)                 17.177     (the graph)
  NO barrier (cost control)                 16.001
  LAUNCH BOUNDARY                           1.176     <- what must be beaten
  legacy unpadded 2fence ns64               18.526      2.525   +1.349
  PADDED 2fence ns64                        18.538      2.537   +1.361   <- best flat
  PADDED 2fence ns0                         19.111      3.110   +1.934
  PADDED PTX ns64                           18.505      2.504   +1.328
  TREE G=4 / G=8 / G=16                     18.630      2.629   +1.453
  TREE G=84                                 18.604      2.603   +1.427
```

**Padding changed the barrier by +0.012 µs. The trees changed it by +0.09 µs —
i.e. made it slightly worse. PTX acquire/release saved 0.033 µs.** Arrival
serialization is not the cost, false sharing is not the cost, the fence is not
the cost, and the backoff is not the cost.

**RESULT 3 — the hidden memset is real but negligible: 0.013–0.119 µs**, not the
0.8 µs a kernel launch would suggest, because an 8-byte `cudaMemsetAsync` does not
behave like a grid launch here. Both suspected confounds were genuine defects in
the instrument and both were worth ~nothing. **§13.13's 2.17–2.56 µs was not an
artifact.** The old verdict was reached by luck rather than by rigour, but it was
not wrong.

**RESULT 4 — cadence is the one thing that helps, and it changes the conclusion
about what to build.** Barriers measured round-robin, twice, at 12288 rows
(round 1 / round 2; the first, single-shot version of this table read 23.9 µs at
cad=1 where parts 1 and the published test both read 18.5 — re-measuring resolved
it and is why the table is now round-robin):

```
  cad   flat r1   flat r2   tree r1   tree r2   boundary
    1    18.505    18.513    18.505    18.505
    2    17.514    17.507    17.507    17.509
    4    16.959    16.926    16.979    16.959    1.194 us/node
    8    16.703    16.695    16.721    16.676
   16    16.485    16.505    16.496    16.516
   32    16.358    16.331    16.364    16.344
```

Total barrier overhead **falls** with cadence (2.51 → 0.33 µs per node) while the
cost **per barrier rises** (2.50 → 10.0 µs), because a barrier that covers more
work has more straggler skew to expose. This is the seductive result: at cad ≥ 4
the persistent column is *below* the 1.194 µs boundary, and it looks like the
megakernel wins.

**It does not, and the reason is the comparison, not the measurement.** A cadence
of K means K nodes are fused. The fused nodes can be executed either inside a
persistent kernel with one barrier per K, **or as ordinary kernels with one launch
per K**. Both are "fusion"; only the second is real. Per node:

```
                              K=1     K=2     K=4     K=8    K=16    K=32
  fused kernel + launches    1.194   0.597   0.299   0.149   0.075   0.037
  persistent + barriers      2.512   1.506   0.925   0.694   0.504   0.330
  --> margin for persistent -1.318  -0.909  -0.626  -0.545  -0.429  -0.293
```

**The fused kernel with launches wins at every cadence**, because the eliminating
move is to reduce the *number* of boundaries, and once you have decided to fuse,
a launch boundary costs 1.19 µs while a barrier costs 2.50 µs. Substituting a
barrier for a launch is never the win; it is always a 1.3 µs/node loss that
fusion then has to pay for.

**RESULT 5 — the small-node regime, which §13.13 never visited, also loses.** One
stage read by the whole grid, 400 stages, graph of N launches vs one persistent
kernel with N barriers:

```
   stage    graph us/stage   persist us/stage   ratio
     8 KB      1.334           2.054           0.649x
    32 KB      1.451           2.054           0.707x
   128 KB      1.470           2.054           0.716x
   512 KB      1.901           2.907           0.654x
     2 MB      3.855           4.312           0.894x
     8 MB     11.784          12.284           0.959x
```

The persistent column is **flat at 2.054 µs** across 8 KB → 128 KB: that is pure
rendezvous cost with essentially no work to hide it. The graph's cost at those
sizes is 1.33–1.47 µs and is also launch-dominated, and it is lower. There is no
crossover, at any stage size, in either direction.

> **VERDICT: the megakernel is CLOSED, and §13.13's explanation is corrected.**
> The cost is not what the barrier has to *order* — padding, PTX acquire/release,
> and a two-level arrival tree all leave it at 2.5 µs. It is the rendezvous
> itself: waiting for the last block, releasing to 504 pollers, and restarting
> with no overlap. **On sm_86 the kernel launch boundary is the cheapest correct
> grid-wide synchronization primitive available, at 1.19 µs against a software
> barrier's 2.50 µs, and it cannot be optimized below that in software.**

**What this changes.** The project's law *"a barrier in front of a
bandwidth-critical kernel loses"* now has its mechanism, and the mechanism
redirects the effort. If a launch boundary is the cheapest boundary, the win is
**fewer boundaries, not better ones** — and §13.5 already measured where that
pays: *"achieved bandwidth collapses from 823.8 GB/s at 32 MB/node to 444.0 GB/s
at 1 MB/node … shaping kernels larger is a lever nobody has attacked at the kernel
level."* **Fusion is that lever and the megakernel was never the way to pull it.**
A concrete, verified candidate already exists: `k_shortconv` (kernels.cu:521) is
launched with 8 blocks x 256 threads, one thread per channel, and thread `ch`
touches only `B[ch]`, `C[ch]`, `X[ch]`, `conv_w[3ch..]` and `hist[2ch..]` — it is
**elementwise in the channel dimension**. A block that owns a 32-aligned channel
band can therefore compute the three `shortconv_in` rows for its band
(`k_gemv_q4_0` over 6144 rows) and then run the depthwise conv and its Q8_1
quantisation **with no cross-block communication at all**. That turns 2 of the
4 nodes in each of 30 ShortConv layers into 1, with no barrier anywhere.

**What survives from Phase 5, unchanged:** the porting rule of §13.6 is now
verified against **14 distinct synchronization implementations**, which is a
stronger statement than §13.13 could make. The grid-stride rewrite is safe; the
persistence is not.

---

## 14. LONG CONTEXT — WHAT THE FALL-OFF ACTUALLY IS

This section exists because the question changed. Up to §13 the question was
"how fast is a token", measured at context 272 where attention is free. The
question that decides whether this engine is usable as a serving engine is "how
fast is a token when the prompt is 30k characters of system prompt", and the
answer to that was not merely worse, it was **wrong in a way the reported number
was hiding**.

### 14.1 The engine was reporting the flattering half

`RESULT` reported `tok_s` and nothing else. Decode is one token per step; prefill
is also one token per step, because **there is no batched prefill** — every prompt
token costs a full decode step. So ingesting a 131040-token system prompt costs
131040 sequential steps, and none of that was in the number.

Both are now reported. Same weights, same KV cache, same greedy 32 tokens, one
RTX 3090 Ti, `LK_ATTN_MAX_SEQ` raised to 131072:

| ctx | engine tok/s | llama.cpp tok/s | ratio | engine prefill | llama.cpp prefill | prefill ratio |
|---|---|---|---|---|---|---|
| 256 | 451.08 | 256.29 | **1.76x** | 0.58 s | 0.07 s | 8.4x |
| 1024 | 444.57 | 261.46 | 1.70x | 2.36 s | 0.17 s | 13.7x |
| 4096 | 419.18 | 255.63 | 1.64x | 9.71 s | 0.62 s | 15.6x |
| 8192 | 385.64 | 250.23 | 1.54x | 20.23 s | 1.27 s | 15.9x |
| 16384 | 354.69 | 237.17 | 1.50x | 43.31 s | 2.65 s | 16.3x |
| 32768 | 290.73 | 218.11 | 1.33x | 96.23 s | 5.76 s | 16.7x |
| 65536 | 215.83 | 186.95 | 1.15x | 233.22 s | 13.48 s | 17.3x |
| 131040 | 141.69 | 144.73 | **0.98x** | **627.49 s** | 34.47 s | **18.2x** |

**At maximum context the engine is no longer faster, and reading one maximum
prompt takes ten and a half minutes where llama.cpp takes thirty-four seconds.**
The 1.69x at short context is real and it is not what a serving workload sees
first: the workload pays the prefill before it gets a single token.

### 14.2 The curve is a property of the engine, not of the text

A fall-off that depends on the *content* would mean something is wrong with how
the model is being run. A fall-off that depends only on *length* is the physics of
reading a KV cache. The two are distinguishable by running a second corpus.

The second corpus is 148913 tokens of source code — a different token
distribution, different entropy, different router traffic, from a different
document. If the curve moved, the kernel was reacting to the text. It did not
move:

| ctx | 256 | 1024 | 4096 | 8192 | 16384 | 32768 | 65536 | 131040 |
|---|---|---|---|---|---|---|---|---|
| regular | 1.76x | 1.70x | 1.64x | 1.54x | 1.50x | 1.33x | 1.15x | 0.98x |
| code | 1.70x | 1.75x | 1.63x | 1.55x | 1.49x | 1.32x | 1.16x | 0.97x |

Every point agrees to within about one percent, and prefill agrees from 8.2x to
17.9x against 8.4x to 18.2x. **The fall-off is length, not content.**

### 14.3 The diagnosis, and then the measurement that overturned it

The first diagnosis was a curve fit: decode marginal cost at long context implies
attention moves KV at about 570 GB/s while every streaming kernel in the same
engine moves weights at 942 GB/s, and that gap *is* the fall-off.

That reasoning has a defect that end-to-end timing cannot expose: it attributes
**all** marginal cost to attention, so it cannot distinguish "attention is slow"
from "attention is slow for reason X" from "something near attention is slow". An
end-to-end A/B at 131040 costs 627 s of prefill per side and still cannot separate
them.

`tests/bench_attn.cu` was written to separate them. It times `k_attn_split` +
`k_attn_combine` — the exact pair the engine launches per attention layer — at a
size where the KV cache is far larger than the 6 MB L2, and uses the kernel's own
`LK_ATTN_ABLATE` switch to split the kernel into its parts. At seq=131040, per
layer, 268.4 MB of KV, median of 5×200 iterations, reproducible to 0.1%:

| configuration | time | bytes actually read | effective BW |
|---|---|---|---|
| A3: neither loop | 23.2 µs | 0 | — (fixed overhead) |
| A1: K read + score dot | 285.0 µs | 134.2 MB | **471 GB/s** |
| A2: V read + accumulate | 187.3 µs | 134.2 MB | **716 GB/s** |
| A0: both (real kernel) | 456.4 µs | 268.4 MB | **588 GB/s** |

The parts sum to 449.1 µs against 456.4 µs measured, so the two phases barely
overlap and the decomposition is real.

**The two loops read the same number of bytes over the same layout and differ by
1.5x.** The reason is the mapping, and it is visible in the source:

* **V** maps `tid -> (half, head, pair)`, so a warp covers all 32 half2 pairs of
  one head — 128 contiguous bytes, **exactly one cache line per warp load**.
* **K** maps `tid -> token`, so at a given `i` the 32 lanes of a warp address
  `base + t*1024 + i*4`: **thirty-two different cache lines, four useful bytes
  each.**

The old comment above the K loop was right that DRAM traffic is identical either
way — the line is fully consumed over the 32 iterations. What it missed is that
one warp instruction issuing 32 sector lookups against 8 is a different
instruction to the memory system even when the bytes are the same.

### 14.4 What the fix bought, verified two ways

Two changes to K, and the weaker one is the one that was already written:

1. **16 × `uint4` loads replace 32 × `__half2`.** The arithmetic is untouched —
   `q` and `i` ascend in the same order, `v0` before `v1`, into the same four
   accumulators — so the result must be **bit-identical**.
2. The access pattern itself is *not* fixed by (1). It is still one thread per
   token and still 32 lines per warp load.

(1) is verified by `tools/ab.py`, which compares the token trajectories of the old
and new binaries across seven regimes: **7/7 BIT-IDENTICAL**. That gate is
meaningful rather than ceremonial because `NVFLAGS` contains no `-use_fast_math`,
so the compiler is not permitted to reassociate the add chain. If the reasoning
about the loop order had been wrong, the gate would have failed.

In isolation (median of 5×200, interleaved rounds):

| seq | before | after | gain | before BW | after BW |
|---|---|---|---|---|---|
| 8192 | 52.4 µs | 45.9 µs | 1.14x | 320 GB/s | 366 GB/s |
| 32768 | 130.7 µs | 104.5 µs | 1.25x | 513 GB/s | 642 GB/s |
| 65536 | 246.6 µs | 203.9 µs | 1.21x | 544 GB/s | 658 GB/s |
| 131040 | 512.1 µs | 456.4 µs | **1.12x** | 524 GB/s | 588 GB/s |

The isolated number only matters if it survives the trip to a real workload. Two
engine binaries, interleaved round-robin so that any drift over a forty-minute run
hits both sides equally, same corpus, same 32 greedy tokens:

| ctx | before | after | gain | prefill before | prefill after |
|---|---|---|---|---|---|
| 32768 | 290.14 tok/s | **315.28 tok/s** | **1.087x** | 96.1 s | 91.9 s |
| 131040 | 140.81 tok/s | **150.86 tok/s** | **1.071x** | 627.8 s | **580.9 s** |

Run-to-run spread is 0.2-0.3%, an order of magnitude below the effect, and the
two runs of each side bracket each other. More importantly the **isolated
measurement predicts the end-to-end result**: at 32768 the kernel saves
(130.7 - 104.5) us x 10 layers = 0.262 ms out of a 3.44 ms token, which is
314.5 tok/s against 315.28 measured; at 131040 it predicts 153.8 against 150.86.
That agreement is the check the original curve fit did not have.

**At maximum context this moves the engine from 0.98x to 1.04x against
llama.cpp** -- from "no longer faster" back to faster. Prefill improves too
(7.5% at 131040), because prefill is attention-bound at long context for exactly
the same reason.

**And it does not close the gap.** 588 GB/s against 942 GB/s streaming. The
original diagnosis was directionally right and quantitatively wrong: the K load
pattern costs about 12%, not the whole 40%, and the ~570 GB/s I inferred from the
end-to-end fit was a coincidence that happened to land between the before and
after numbers. **A curve fit that cannot be checked against an isolated
measurement should not be reported as a mechanism.**

### 14.5 The counter-check the fit was missing

The fit is not useless — it is a good *model*, it just is not a *mechanism*. It
says the per-token cost is a constant plus an attention term:

    t(ctx) ~= C + attention(ctx)

with `C` the fixed weight-streaming cost, which does not grow with context.
Measured: the engine's `C` is **2.3 ms** (at context 256 the whole token is
2.22 ms and attention is negligible). At 131040 the measured attention term is
4.56 ms, predicting 6.86 ms; the engine delivers 141.69 tok/s = 7.06 ms. The model
holds to 3%.

llama.cpp fits the same shape with a much worse `C` and much better attention:
3.90 ms fixed, and 6.91 - 3.90 = **3.01 ms** of attention at 131040, i.e.
**891 GB/s**. So:

* short context, the engine wins on `C` (2.3 vs 3.90 ms) — that is the 1.76x;
* long context, the engine loses on attention (4.56 vs 3.01 ms) — that is the 0.98x;
* **the entire long-context deficit is one loop's access pattern**, and the prize
  for fixing it properly is 2.3 + 3.0 = 5.3 ms = **~188 tok/s at maximum context,
  1.30x faster than llama.cpp**, against 141.69 today.

### 14.6 The unbounded write, found twice

Nothing in the forward path bounds-checks `d_pos`, and `kcache`/`vcache` are
exactly `max_seq` long. Any mode that feeds more tokens than `max_seq` writes past
the end of them. It does not crash — it corrupts the neighbouring allocation, and
the model then emits blank or repetitive text, which reads exactly like **"the
model gives up at long context"**.

It was found in `gen` first, where a 131040-token prompt at the default
`LK_MAX_SEQ` of 131072 hits it on the first generated token, and it was diagnosed
as a model failure until the arithmetic was done. It was **still open in `ppl`** —
the mode the perplexity and NLL evidence runs through — where this project's own
149837-token corpora against a 131072 cache were overrunning by 18765 positions
while continuing to print a plausible perplexity. It was also open in both
argv-fed diagnostic modes.

All four now call one shared guard. It was capped rather than fatal in `ppl`,
because dying would remove exactly the high positions the NLL curve exists to
measure.

### 14.7 Coherence — the check that the model is still there

A benchmark cannot tell a fast kernel from one that has lobotomised the model, and
the whole point of this engine is that it does not. Three independent checks:

* **Nonce retrieval in a haystack**, both engines fed byte-identical token streams,
  needle at depth 0.5: **HIT/HIT at 2048, 8192, 32768, 65536 and 98304; MISS/MISS
  at 126976 and 131072.** The engine never misses where llama.cpp hits. **The
  lobotomy signature is absent.** It also means the honest description of this
  model's usable context is about **98k, not 131072** — 131072 is nominal, and it
  is nominal for llama.cpp too.
* **Factual / reasoning / code**, on ChatML-wrapped prompts: factual 4/5 against
  3/5, reasoning 4/5 against 4/5, code 4/6 against 6/6 where one of the two
  engine failures is a defect in my own answer extractor. Six code problems
  cannot distinguish 4/6 from 6/6, so the correct claim is **"no evidence of a
  deficit"**, not parity.
* **Per-position NLL against llama.cpp, over the 126937 positions both engines
  scored of the same corpus** -- the sharpest of the three, because it measures
  the whole output distribution rather than one retrieved string.  A fault in the
  attention split would not move the mean; it raises NLL for the tokens that sit
  FAR from the start, and a corpus-wide average dilutes exactly that away.  One
  teacher-forced pass each, bucketed by position:

  | position | n | engine NLL | llama.cpp NLL | delta | top-1 agree |
  |---|---|---|---|---|---|
  | 1-256 | 255 | 2.803144 | 2.806608 | **-0.003464** | 86.67% |
  | 256-1024 | 768 | 3.572755 | 3.543318 | +0.029437 | 90.36% |
  | 1024-4096 | 3072 | 4.365313 | 4.303838 | +0.061475 | 87.53% |
  | 4096-16384 | 12288 | 4.383152 | 4.335751 | +0.047401 | 87.61% |
  | 16384-65536 | 49152 | 4.411635 | 4.385888 | +0.025747 | 87.97% |
  | 65536-131072 | 61401 | 4.340046 | 4.343465 | **-0.003419** | 88.48% |

  The oracle's pass scored positions 0-126937; the top 4133 positions are not in
  it, because llama.cpp's dump stopped when the process that was reading its
  output was killed and its pipe filled -- it blocked on write rather than
  failing, which is worth knowing before trusting a stalled long run again.  The
  last bucket is therefore 61401 of 65536 positions, and the conclusion below does
  not depend on the missing 6%.

  **The delta does not grow with position.**  It peaks in the 1024-4096 bucket and
  returns to zero -- marginally negative, the engine very slightly BETTER than
  llama.cpp -- in the 65536-131072 bucket, which is where an attention defect
  would be loudest.  Top-1 agreement is flat at 87-90% across the entire range.
  The engine also scores better above 65536 than it does at 16384-65536, which is
  simply what more context does.  **If split-K attention were subtly wrong at long
  range, this is the instrument that would show it, and it shows nothing.**
* The engine's **first nine tokens on a clean prompt are identical** to
  llama.cpp's.

### 14.8 The engine did not stop, and that mattered more than any kernel

The engine produced the correct answer, emitted `<|im_end|>`, and then kept
going — through `<|endoftext|>` and into a repeat of the answer. Nothing was
wrong with the model or the kernel; the loop counted to `ngen` and never looked at
what it had produced. **A reader sees an answer repeated forever and concludes the
quantisation broke the model.**

`<|im_end|>` is 124900, which is the model's own `tokenizer.ggml.eos_token_id` in
the GGUF metadata, and `<|endoftext|>` is 124895; both were read out of the
running llama-server's `/tokenize` rather than guessed. Generation now stops on
them unless `LK_BENCH` is set — bench mode must run a fixed length or every
throughput number in this document becomes irreproducible — and `LK_EOS=""`
disables the stop. The parity gate sets `LK_EOS=""` explicitly, because a gate
that silently shortens itself is worse than one that fails.

Same prompt, same kernel, 600-token budget, the only difference being the stop:

| | tokens emitted | `stopped` | tail of the transcript |
|---|---|---|---|
| stop **on** | 99 | 1 | `...construction began in the 1910s."` |
| stop **off** | 600 | 0 | `...ever since.\n</think\n>The capital of Australia is **Canberra**...` |

The answer is identical and correct in both. Without the stop the model reaches
the end of its turn and starts over, which is precisely the output that reads as
a broken quantisation.

### 14.9 What is still open, in priority order

1. **Rewrite the K loop to be warp-coalesced.** The measured target is 891 GB/s
   (§14.5). The shape that reaches it gives up thread-private dot products and
   pays a shuffle reduction instead; instruction issue goes up by roughly 5x and
   is still under 20% of the machine, which is the whole reason the trade is
   available. **This will NOT be bit-identical** — the summation order changes —
   so it must be gated on `make test-attn` against the float64 reference instead,
   and the two gates are not interchangeable.
2. **Batched prefill.** 8.4x → 18.2x is the number a serving workload pays first,
   and it is a consequence of having no batched path at all.
3. **RoPE inverse frequencies in double, precomputed.** The phase is computed as
   `powf(1e7, -2d/64)` per thread per token in fp32; the error is `pos * eps`,
   about 0.013 rad at 131072 today, and 0.026 rad if `-use_fast_math` is ever
   added — at which point nothing in the test suite would catch it.
4. **Extend `make test-attn` to seq=131072.** Today it validates a few hundred
   tokens. The long-context evidence is behavioural (the needle test), not
   numerical, and item 1 above cannot be gated without this.
5. **Document the effective context as ~98k.**
6. **Do not quantise the KV cache.** FP16 KV is what makes exact parity with
   llama.cpp possible, and llama.cpp is also FP16 here.

---

## 15. PHASE 4z.8 — THE K LOOP IS WARP-COALESCED

§14 ended with the diagnosis and left the fix as item 1 of §14.9: *"Rewrite the K
loop to be warp-coalesced. The measured target is 891 GB/s."* This section is that
rewrite, its measurement, and the two things it turned out to be wrong about.

### 15.1 What the old loop actually cost, stated as a mechanism

The old K loop was one thread per token:

```cuda
for (int j = threadIdx.x; j < n; j += blockDim.x) {
    const __half * krow = kc + ((size_t)(t0 + j) * LK_N_KV_HEADS + kv_head) * LK_HEAD_DIM;
    ... 8 x uint4 over the token's 128 bytes ...
}
```

Lane `l` owns token `t0 + l`, so at uint4 index `q` the warp's 32 lanes address
`base + t*128 + q*16`. That is **32 distinct 128-byte lines per instruction**, and
each lane's 16 bytes is half of a 32-byte sector. The other half of each sector is
only touched by the *next* iteration. With ~40 resident warps each holding 32
lines, the working set is ~160 KB against a 128 KB L1, so the second half was, in
practice, fetched again.

§14.3 had this right, and §14.4 is where it stopped: the `uint4` form was already
written and bought 1.12x, and the section says plainly *"the access pattern itself
is not fixed by (1). It is still one thread per token and still 32 lines per warp
load."* That sentence is the entire justification for this phase.

### 15.2 The new mapping

Lane `l` of a warp now covers **dims `(l & 7) * 8 .. +7` of token `t0 + j + (l >> 3)`**
— one `uint4`, 16 bytes. Eight lanes tile one token's 128 bytes exactly, so a warp
instruction covers **4 tokens = 512 contiguous bytes = 16 sectors, every one
consumed whole by that single instruction.** It is the same coalescing the V loop
always had.

Three consequences that had to be designed for, not discovered:

1. **The dot product is now partial.** Each lane sums 8 of the token's 64 dims, so
   the eight lanes sharing a token must be combined. That is 3 rounds of
   `__shfl_down_sync(mask, v, off, 8)` at offsets 4, 2, 1, for each of the 4 heads:
   **12 shuffle instructions per warp per 4 tokens.** `width = 8` keeps every round
   inside its own token's lane group, and the full mask is correct because the
   liveness predicate is uniform across a group.

2. **The dead-lane tail needs a safe address, not a branch.** The last iteration of
   a slice rarely has a multiple of 4 tokens. A dead lane loads `t0` (always in
   range, because the early-out guarantees `t1 > t0`) and zeroes its accumulator
   afterwards, rather than branching around the load — a branch there costs exactly
   the memory-level parallelism the rewrite exists to buy.

3. **`q` had to be re-staged, and this is the part that is easy to get wrong.** The
   eight lanes sharing a token read the same eight dims' worth of q, so they address
   `base + dsub*128 B` — a stride of exactly 128 B, which is exactly the 32-bank
   cycle. Stored the obvious way (one `float4` of the four heads per dim) **every q
   read is an 8-way bank conflict.** Each 8-dim group is therefore staged as **9
   `float4` (36 floats) instead of 8**; the extra 16 bytes shifts group `dsub` to
   bank group `(4 * dsub) mod 32`, and the eight lanes hit eight different bank
   groups. The array grows from 1024 B to 1152 B and the shared-memory
   `static_assert` was updated to size it.

**Register pressure did not move.** `ptxas -v` on the shipped build reports
`k_attn_split` at **62 registers, 0 spill stores, 0 spill loads, 23680 B smem** —
the same 62 registers as the kernel it replaces, which matters because §14's
comment block records that an earlier attempt at this kernel hit 255 registers with
spills and lost the V loop's unrolling to it. The union of `s_qf` (1152 B),
`s_score` (16384 B), `s_m` (4096 B) and `s_v` (2048 B) is 23680 B against the 48 KB
sm_86 static limit.

### 15.3 Measured, against the real binary rather than against history

`bench_attn`, median of 200 iterations, three binaries **interleaved round-robin**
inside one session so that drift hits all three equally. `uint4_HEAD` is the
committed HEAD (`7590cb7`), `pre_uint4` is the binary from before that change, and
the two runs of each bracket each other:

| seq | pre_uint4 | uint4_HEAD | 4z.8 | 4z.8 vs HEAD |
|---|---|---|---|---|
| 8192 | 52.48 µs / 319.7 GB/s | 45.90 µs / 365.5 GB/s | **43.65 µs / 384.4 GB/s** | 1.052x |
| 32768 | 134.05 µs / 500.6 GB/s | 104.45 µs / 642.5 GB/s | **100.27 µs / 669.3 GB/s** | 1.042x |
| 65536 | 246.72 µs / 544.0 GB/s | 203.85 µs / 658.4 GB/s | **180.12 µs / 745.2 GB/s** | 1.132x |
| 131040 | 512.35 µs / 523.8 GB/s | 456.32 µs / 588.1 GB/s | **336.95 µs / 796.5 GB/s** | **1.354x** |

Every row is first-party from that one session — all four sizes, all three
binaries, six interleaved rounds each — because the numbers this replaces had been
carried forward from two earlier sessions and mixing those in would have made the
comparison a claim about history rather than about the binaries. Run-to-run spread
was under 0.1%: the 131040 triples were 336.99/336.90, 456.23/456.41, 512.67/512.03.

**A two-term fit over 65536 and 131040 gives `23.3 µs + bytes / 856 GB/s`.** Check
at the two points it was not fitted to: `23.3 + 67.1*1.1687 = 101.7` against 100.27
measured at 32768, and `23.3 + 16.8*1.1687 = 42.9` against 43.65 at 8192 — 1.4% and
1.7%. The marginal rate implies **856 GB/s** against the engine's measured streaming
rate of 942 GB/s and a 1008 GB/s spec, so the loop is at 91% of what this engine
achieves anywhere. The 23.3 µs fixed part is launch and occupancy, and at 131040 it
is 6.9% of the kernel.

Note what the fit does *not* do for the old kernel. Fitting `uint4_HEAD` over the
same two points gives **531 GB/s with a −48.6 µs intercept** — a negative fixed
cost, which is not a physical thing. It means the old loop's efficiency *fell* with
size and had no fixed part to blame: the per-byte cost itself grew with the working
set, which is what cache eviction looks like from the outside.

### 15.4 The ablation had to be re-read, and it changed meaning

§14.3's decomposition used `LK_ATTN_ABLATE`: 1 ablates V, 2 ablates K, 3 ablates
both. **Under the new loop, "ablate K" no longer removes the whole K phase** — it
removes the load and the dot but keeps the shuffle reduction, which is now part of
that phase. So A3 is no longer "fixed overhead"; it is "shuffle plus loop control".
Reading it the old way makes V look like it regressed, which it did not.

Per layer at seq=131040, all four configurations in one session:

| configuration | uint4_HEAD | 4z.8 |
|---|---|---|
| A3: shuffle/control only | 23.2 µs | 48.6 µs |
| A1: K load + dot | 285.0 µs | 181.8 µs |
| A2: V load + accumulate | 187.3 µs | 204.3 µs |
| A0: real kernel | 456.4 µs | **337.1 µs** |

Subtracting A3, which is the only honest way to compare the two designs:

| phase | uint4_HEAD | 4z.8 | rate now |
|---|---|---|---|
| K load + dot | 261.8 µs | **133.3 µs** | **1007 GB/s** |
| V load + accumulate | 164.1 µs | **155.8 µs** | **862 GB/s** |
| shuffle + control | 0 µs | 48.6 µs | — |
| parts sum | 449.1 | 337.6 (measured 337.1) | |

**The marginal K rate is now 1007 GB/s, which is the DRAM spec.** The coalescing
worked completely: K went from the slowest phase to the fastest, at 1.96x. The
phases still barely overlap — the parts sum to 337.6 against 337.1 measured.

The honest reading of the new kernel is therefore:

* **K is done.** 133.3 µs at 1007 GB/s marginal; nothing left there.
* **V is now the larger phase** at 155.8 µs / 862 GB/s, and it is still on the old
  4-byte-per-lane shape (a warp covers 128 contiguous bytes, one line, but only
  moves 4 B per lane per instruction).
* **The shuffle reduction is not free.** It moved 48.6 µs of cost out of the memory
  system and into the instruction stream, which is 14.4% of the kernel. The trade is
  strongly net-positive — 456.4 → 337.1 — but it is a trade, and §14.9's estimate
  that instruction issue "is still under 20% of the machine" was about the FMA and
  load stream; the shuffles are a separate unit and are the thing that grew.

### 15.5 Numerics: this change makes the kernel MORE accurate, not less

This is **not bit-identical and cannot be** — one thread summing 64 terms in order
became eight partials combined by a tree — so `tools/ab.py` is not the gate and was
never going to be. `make test-attn` against the float64 reference is the gate, and
the test had to be extended first because **it stopped at 8192, which is exactly
where `LK_ATTN_SPLIT` saturates and every longer sequence becomes the same code path
with a bigger number.** The shipped gate validated the one regime in which a
long-context addressing bug cannot appear.

It now runs 12 lengths, adding 16384, 32768, 65536, **100003** (prime, so not a
multiple of the warp's 4-token stride, of 32, of TARGET_N or of SPLIT — it exercises
the dead-lane tail the interior cases never reach) and **131072**. Measured max
absolute error against an exact float64 reference:

| seq | serial kernel vs float64 | 4z.8 split vs float64 |
|---|---|---|
| 512 | 8.625e-08 | 1.867e-08 |
| 4096 | 6.674e-08 | 1.037e-08 |
| 8192 | 7.621e-08 | 1.639e-08 |
| 16384 | — (cannot run) | 9.852e-09 |
| 32768 | — | 8.472e-09 |
| 65536 | — | 5.961e-09 |
| 100003 | — | 3.463e-09 |
| 131072 | — | **3.612e-09** |

**The error is five to twenty times SMALLER than the serial kernel's, and it falls
with length.** That is not a coincidence: a tree summation has logarithmic depth and
the serial chain has linear depth, so the longer the row the more the tree wins. The
change made the kernel both faster and more accurate, and the second half of that was
not the goal.

### 15.6 The test harness was lying, and the lie was in the reference

The first run of the extended test **FAILED at 131072** with
`split diverges from serial by 6.160e-03`, and printed a per-head error table that
looked exactly like a real long-range defect.

It was not. `k_attention_decode` keeps its whole score row in
`__shared__ float s_score[LK_ATTN_DECODE_MAX_SEQ]` with a cap of 8192; past that it
prints `FATAL` and **returns early, leaving its output buffer untouched.** The test
then compared the kernel under test against memory that had never been written. The
kernel's actual error against float64 at that length was 3.612e-09.

The fix is in the harness: the serial kernel is a *second opinion*, not the
definition of correct — `host_ref` is the definition — so it is now skipped above
`LK_ATTN_DECODE_MAX_SEQ` and the `split-vs-serial` comparison is reported as skipped
rather than as a failure. `tol_split_serial` was left at **1e-4**, unchanged,
because the measured value at the lengths where it can run is 7.5e-08, a thousand
times inside it. Loosening a tolerance because a broken reference failed it would
have been the wrong repair, and it was tempting.

### 15.7 End to end, and the instrument predicted it

An isolated kernel number is a hypothesis until it survives a real workload. Two
engine binaries, `LK_BENCH=1`, 32 greedy tokens, corpus `ref/corpus_regular.bin`,
interleaved round-robin so drift hits both sides equally:

| ctx | uint4_HEAD | 4z.8 | gain | prefill HEAD | prefill 4z.8 |
|---|---|---|---|---|---|
| 32768 | 317.74 tok/s | **320.40 tok/s** | 1.008x | 91.9 s | 92.0 s |
| 131040 | 152.06 tok/s | **185.22 tok/s** | **1.218x** | 580.5 s | **526.3 s** (0.907x) |

Per-side runs were 317.89/317.59 and 319.92/320.89 at 32768, and 151.97/152.15 and
185.14/185.30 at 131040 — a spread of 0.1%, two orders of magnitude below the
effect at 131040 and comfortably below it even at 32768.

**The microbenchmark predicted this before it was run.** The two-term fit gives
`2.3 ms + 10 x 0.337 ms = 5.67 ms`, or 176.4 tok/s; the engine delivered 185.22
tok/s, i.e. 5.40 ms. The instrument was **5% conservative**, which is the direction
that matters: §14.4's rule is that the isolated measurement must predict the
end-to-end result or the model is wrong, and a model that under-promises satisfies
it. (At 32768 the same arithmetic predicts 3.12 ms = 320.5 tok/s against 320.40
measured — 0.03%.)

Against the llama.cpp column in §14.1, which is the number that decides whether this
engine is worth running:

| ctx | llama.cpp | engine before | engine now | ratio before | **ratio now** |
|---|---|---|---|---|---|
| 32768 | 218.11 tok/s | 290.73 | 320.40 | 1.333x | **1.469x** |
| 131040 | 144.73 tok/s | 141.69 | **185.22** | 0.980x | **1.280x** |

§14.5 set the prize as *"2.3 + 3.0 = 5.3 ms = ~188 tok/s at maximum context, 1.30x
faster than llama.cpp"*. Measured: **185.22 tok/s, 1.280x — 98.5% of the stated
prize**, from a starting point that was slower than llama.cpp.

Prefill improved by 9.3% as a side effect (580.5 → 526.3 s). That is not a
coincidence and it is the same mechanism: prefill in this engine is one decode step
per prompt token, so it is attention-bound at long context for exactly the reason
decode was. It is far from fixed — 526 s against llama.cpp's 34.5 s is a 15.3x
deficit, and that is the next piece of work, not this one.

### 15.8 The change is not bit-identical, and the quality gate says so in numbers

`tools/ab.py`, 7 cases x 48 greedy tokens, HEAD against 4z.8: **5 of 7 identical,
2 diverged** — `agentic` first differing at token **39 of 48**, `reasoning` at token
**45 of 48**.

That is the expected signature of a ~1e-8 perturbation and not of a behaviour
change, and §11 law (l) is the reason it cannot be read as one: *greedy parity is
not a usable correctness gate when the model amplifies perturbations*. A single
flipped argmax after 39 correct tokens is what a tiny numerical difference looks
like; a broken kernel does not agree for 39 tokens and then disagree.

The gate that can actually see the difference is the whole-corpus teacher-forced
perplexity — the same instrument as §14.7, over all 131070 scorable positions of
`ref/corpus_regular.bin`, one pass per binary, `LK_NO_GRAPH=1`:

| binary | nll | ppl | top1_acc |
|---|---|---|---|
| `uint4_HEAD` | 4.360372 | 78.286219 | 0.3017 |
| `4z.8` | 4.367156 | 78.819151 | **0.3019** |
| delta | **+0.006784 (+0.156%)** | **+0.533 (+0.68%)** | +0.0002 |

The baseline column is also a cross-check on the baseline binary itself: §14
recorded `ppl=78.284998 nll=4.360356 top1_acc=0.3017` for it over 131071
positions, and this run reproduces that to five significant figures from a
separate invocation days later. The binary being compared against is the binary
§14 measured.

**This is a real cost and it is not noise** — the computation is deterministic, so
+0.156% NLL is a systematic shift caused by the changed summation order. For scale,
§14.7's engine-vs-llama.cpp per-position deltas run +0.026 to +0.061 across the
position buckets, so this change moves the engine by roughly a quarter to a ninth
of the distance it already sits from the oracle. Top-1 agreement is unchanged
(+0.02 points, i.e. one token in 65000). §11 law (l) is the relevant precedent: this
model amplifies 1e-8 perturbations, so a 1e-8-level change to a kernel that runs 40
times per token over 131070 positions landing as a 0.16% aggregate NLL move is the
expected magnitude, not a surprise.

The honest framing is therefore a **trade, stated as a trade**: 1.218x end-to-end at
maximum context, bought with +0.156% NLL and +0.68% perplexity against the kernel it
replaces, on a kernel that is simultaneously 5-20x closer to an exact float64
reference at every length tested. It is possible for both of those to be true at
once — being closer to exact arithmetic is not the same as being closer to whatever
rounding the checkpoint was trained under — and the second number is the one a user
of this engine experiences.

**Still owed, and named rather than assumed:** the needle-in-a-haystack retrieval
test at 98304 and the per-position NLL buckets of §14.7, both re-run against this
binary. Those are the two instruments §14.7 built for exactly this question, a
whole-corpus average is not a substitute for either, and neither has been re-run
against 4z.8 yet.

### 15.9 What this does and does not settle

**Settled:** the K loop's access pattern was worth 1.354x at maximum context, the
marginal K rate is at the DRAM spec, the kernel is more accurate than the one it
replaced, and the gate that says so now covers the lengths the engine actually runs
at.

**Not settled — and these are the next three items, in order:**

1. **V is now the slow half** (155.8 µs, 862 GB/s marginal, still 4 B per lane per
   instruction). The same argument that fixed K applies to it: widen the lane
   payload from `__half2` to 8 bytes and a warp covers 256 contiguous bytes per
   instruction instead of 128, halving the instruction count for the same FMA count.
   Worth about 23 µs/layer, **6.8%**, if it reaches K's marginal rate.
2. **The shuffle reduction costs 48.6 µs/layer (14.4%)** and is now the single
   largest non-memory term. Nothing is proposed here yet; the number is recorded so
   that the next attempt starts from a measurement instead of from the instruction
   count in §14.9, which was optimistic about this specific unit.
3. **The fixed 23.3 µs per layer** is launch and occupancy, and at 131040 it is 6.9%
   of the kernel. It is the same quantity `bench_shapes` measured as the empty-kernel
   toll, and it does not shrink unless the grid does.

**Not attempted here at all:** batched prefill (§14.9 item 2), which is a separate
and much larger piece of work, and the RoPE inverse frequencies in double (§14.9 item
3), which is a long-context *coherence* item rather than a speed item.
