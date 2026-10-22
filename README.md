# liquid-kernel

A hand-written sm_86 CUDA inference engine for **LFM2.5-24B-A2B-QAD-Q4_0** on an
RTX 3090 Ti. Single stream, greedy decode, every kernel written by hand and every
number in this file measured on the machine rather than estimated.

| | |
|---|---|
| decode, ctx 272 | **443.2 tok/s** (interleaved A/B, 5 rounds, spread 0.03%) |
| before Phase 4z | 438.8 tok/s |
| before Phase 4y | 414.7 tok/s |
| effective bandwidth | 650 GB/s = **64.5% of the 1008 GB/s spec** |
| llama.cpp, same GGUF/GPU, same session | 257.9 tok/s |
| speedup vs llama.cpp | **1.72x**, and 1.67-1.72x to 3.2k context |
| speedup vs this engine's own Phase-3 baseline | **4.27x** (102.79 tok/s) |
| traffic | 1466.75 MB/token, so the bandwidth floor is 1.534 ms = **652 tok/s** |
| CUDA graph | **343 nodes**, replayed per token |
| measured batch-1 ceiling | **488.9 tok/s** (all 158 small kernels made free, weight traffic preserved) |
| node deletion is worth | ~0.3-0.45 us each, NOT the 1.5 us Phase 4y measured (see DIAGNOSIS) |
| **prefill, 4k / 32k / 131k prompt** | **21.2k / 18.2k / 11.0k tok/s** (0.193 s / 1.80 s / 11.9 s), batched |
| prefill vs llama.cpp tuned (`-ub 4096 -b 4096`) | **~2x from 4k to 131k** (1.96-2.08x; 1.4-1.9x below 4k; 2.03x at 131k: 11.9 s vs 24.2 s) |
| prefill vs llama.cpp default | **2.9-3.3x from 4k to 131k** |
| prefill vs this engine's own sequential prefill | **16-47x** (256: 0.58 s -> 0.036 s; 131k: 526 s -> 11.9 s) |
| whole-corpus NLL, batched prefill (126,936 positions) | **4.3502** vs llama.cpp 4.3503 -- no degradation to 131k |

## Batched prefill

Until now every prompt token cost one full decode step: a 131072-token prompt was
131072 sequential graph replays, 526 s against llama.cpp's 34.5 s. `src/prefill.cu`
ingests the prompt a CHUNK at a time (default 8192 tokens): every layer is a
handful of launches over the whole chunk. The last prompt token still goes
through the decode graph, so generation starts from exactly the state it always
did.

Measured 2026-10-22, same GGUF, same corpus ids, one request per point, best of
2 (`tools/engsweep.py`, `tools/llamasrv.py`; raw in `bench/*2026-10-22*`; the
engine column is `bench/engine_v2_2026-10-22.json`, after the second round
below, with the morning's first-round numbers beside it; llama.cpp was measured
once, in the morning, on the same GPU). The
engine runs on GPU 0, llama.cpp b10342 on GPU 1 (both RTX 3090 Ti), flash
attention on, f16 KV, one slot. llama.cpp is shown at its default and at its
best measured prefill setting (`-ub 4096 -b 4096`; 1024, 2048 and 8192 were also
tried):

| prompt | engine prefill | (morning) | llama.cpp tuned | x | llama.cpp default | x | engine decode | llama.cpp decode | x |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 0.026 s | 0.025 s | 0.038 s | 1.43 | 0.038 s | 1.43 | 455.7 | 258.7 | 1.76 |
| 256 | 0.036 s | 0.036 s | 0.068 s | 1.91 | 0.068 s | 1.92 | 454.6 | 257.4 | 1.77 |
| 1024 | 0.066 s | 0.070 s | 0.119 s | 1.81 | 0.161 s | 2.45 | 446.5 | 264.3 | 1.69 |
| 4096 | 0.193 s | 0.213 s | 0.383 s | 1.98 | 0.615 s | 3.18 | 424.6 | 259.1 | 1.64 |
| 8192 | 0.376 s | 0.417 s | 0.782 s | 2.08 | 1.248 s | 3.32 | 392.7 | 251.7 | 1.56 |
| 16384 | 0.799 s | 0.896 s | 1.596 s | 2.00 | 2.613 s | 3.27 | 364.0 | 239.7 | 1.52 |
| 32768 | 1.801 s | 2.041 s | 3.529 s | 1.96 | 5.720 s | 3.18 | 319.5 | 220.4 | 1.45 |
| 65536 | 4.396 s | 5.043 s | 8.688 s | 1.98 | 13.397 s | 3.05 | 257.2 | 188.1 | 1.37 |
| 131040 | 11.936 s | 13.929 s | 24.221 s | 2.03 | 34.569 s | 2.90 | 184.9 | 148.1 | 1.25 |

What it is:

* **Q4_0 x Q8_1 on int8 tensor cores.** `mma.m16n8k32` consumes exactly one
  32-element block per k-step, and the Q4_0 nibble layout already IS the int8
  A-fragment layout (word w of a block gives k = 4w.. in its low nibbles and
  16+4w.. in its high ones), so weights go from memory to the tensor core with
  two logic ops and no dequantisation. The accumulator is seeded with the bit
  pattern of 1.5*2^23 so the int32 sum reads directly as an exact float --
  int->float conversion runs at 1/8 rate on sm_86 and would otherwise bound the
  kernel. Per block the term is the decode path's own `d4*(d8*sumi - 8*sum)`.
* **Grouped MoE GEMM.** The chunk's 4 slots per token are counted, planned
  into 64-slot tiles per expert and scattered into a permutation in slot order,
  so every result is deterministic and the whole prefill is bit-identical at any
  chunk size.
* **Causal GQA flash attention** on fp16 tensor cores: a block owns 32 positions
  of one kv head and all four query heads that share it, so each K/V tile serves
  128 query rows. Scores accumulate in fp32 (an fp16-score variant was measured
  and rejected: +0.06% NLL for 8% at 131k); P.V accumulates in fp16 inside each
  64-key tile and in fp32 across tiles, which runs at 2.2x the rate and moved the
  whole-corpus NLL by 0.0009, inside noise. At 127k context it runs at 45.7
  TMAC/s.
* **Per-token kernels carry the decode arithmetic.** Embedding, both norm
  forms (plain and the Phase 4y split), shortconv taps and state, qk-norm + RoPE
  into the KV cache, the short-prompt router and the top-4 selection are
  byte-identical to the decode kernels given the same input
  (`make test-prefill`).

Why it is not bit-identical to the sequential path, and why that is the right
gate: the GEMM sums 32-element blocks in block order where the GEMV sums
lane-strided partials through a butterfly (reassociation, ~2e-7 of the summed
magnitude), and attention runs on fp16 tensor cores. Every Q8_1 quantisation
then turns a 1e-7 difference into flipped int8 roundings -- one layer takes a
8.5e-6 difference to 1% -- so the two paths are equally valid samples of the
quantisation noise and per-position NLL differs by ~0.4 nats sd while the mean
does not move. The gates are therefore distributional:

| positions | batched | llama.cpp | sequential (a3e15db) |
|---|---|---|---|
| 1-256 | 2.8050 | 2.8066 | 2.8031 |
| 256-1024 | 3.5199 | 3.5433 | 3.5728 |
| 1024-4096 | 4.2765 | 4.3038 | 4.3653 |
| 4096-16384 | 4.3307 | 4.3358 | 4.3832 |
| 16384-65536 | 4.3966 | 4.3859 | 4.4116 |
| 65536-131072 | 4.3355 | 4.3435 | 4.3400 |
| **all 126,936** | **4.3493** | **4.3503** | **4.3648** |

After the second round below (int8 scores, 2-op GEMM epilogue, fused SwiGLU):
**4.3502** over the same 126,936 positions (per bucket 2.7871 / 3.5405 / 4.3204 /
4.3200 / 4.3988 / 4.3353), against llama.cpp's 4.3503.

plus `tools/pfcheck.py` (full-vocabulary KL / top-k against the sequential
path), `tools/pfstate.py` (KV and conv state layer by layer) and `make
test-prefill` (every kernel against the decode kernel or a float64 reference).
`tools/ab.py` now pins `LK_PREFILL=seq` on both binaries, so it remains a proof
about the decode path: 7/7 bit-identical against 6ab01db.

Where prefill time goes at 4096 tokens (`LK_PF_PROF=1`): the MoE gate|up and
down GEMMs 50%, the dense projections 25%, the rest bandwidth-bound per-token
kernels. The GEMM runs at ~62 TMAC/s (~125 TOPS; 67 after the second round)
against a measured 158.7 TMAC/s int8 peak; with the math removed its copies alone run at 1.57 TB/s from
L2 against a measured 1.73 TB/s wall, and taller tiles that cut that traffic
put 16 warps behind each stage barrier and measured slower (see the table in
`src/prefill.cu`). At 131k attention is 55% of prefill.

Long-context decode is a bandwidth result, not a kernel one: at 131k both
engines read the 2.68 GB KV cache at ~0.9 TB/s (this engine 893 GB/s, llama.cpp
927 GB/s, from the decode-time increase over short context), so the remaining
lead is the weight stream and the ratio converges toward 1 for any f16-KV
engine. Fewer KV bytes is the only further lever.

### Second round (2026-10-22, afternoon): 9-14% from 4k up

Profile first (`LK_PF_PROF=1`, before this round):

| prompt | GEMMs | attention |
|---|---|---|
| 4k | ~77% | 4% |
| 32k | ~60% | 25% |
| 131k | ~34% | 57% |

**Attention: int8 scores (`k_pf_attn8`), 131k -10%.** The fp16 kernel was at
~98% of its own tensor ceiling (fp32-accumulate fp16 mma runs at 40 TMAC/s for
Q.K, fp16-accumulate at 87.7 for P.V), so the only lever was the precision of
Q.K. Q and K are quantised to int8 per (token, head) row of 64 and scored with
the int8 mma (158.7 TMAC/s), exact in int32 with the same magic-number seed as
the GEMM; P.V is unchanged. The int8 K is a prefill-only copy -- the fp16 cache
the decode path reads is untouched, which is why decode stays bit-identical.
On random data int8 scores are ~20x less accurate than fp32-accumulated fp16,
so the gate was measured on REAL activations (`LK_PF_ATTN_CMP=1` runs both
kernels on the same Q/K/V, 32k prompt, all 10 layers):

| int8 scores | K smoothing | exact tiles | attention output error vs fp16 kernel (rel. RMS, mean of 40 layer-chunks) |
|---|---|---|---|
| everywhere | off | none | 0.46% |
| everywhere | on (SageAttention's mean-key subtraction) | none | 0.49% |
| | on | tile 0 (attention sink) | 0.46% |
| | on | tile 0 + keys within 256 of the query block | 0.35% |
| | on | tile 0 + within 1024 | 0.28% |
| **shipped** | **off** | **tile 0 + within 1024** | **0.26%** |
| | off | tile 0 + within 2048 | 0.23% |
| for scale: the Q8_1 rounding the next GEMM ALREADY applies to this output | | | 0.63% |

Smoothing does not help here: the keys are RMS-normed and RoPE'd, so there is
no common offset for it to remove.

Tiles within 1024 positions of the query block and tile 0 are scored exactly as
`k_pf_attn` does (fp16, fp32 accumulation): at 131k that is ~17 of ~2000 key
tiles per query block, so it keeps nearly all of the speed (attention 1.32x
pure int8, ~1.15x with the exact window; prefill 13.75 -> 12.42 s at 131k,
-3.5% at 32k, nothing at 4k, where almost every tile is inside the window).
Whole-corpus NLL could not resolve any of these variants: run-to-run quantisation
chaos is ~+-0.005 nats on it, larger than every difference above, which is why
the attention-level measurement is the gate. The earlier fp16-score variant
(`PF_ATTN_QK_F16`) is 1.18x and doubles the score error, so it stays off.

**GEMM: two FP ops per value instead of three (`PF_EPI2`), +5-7%.** The
per-block term `d4*(d8*sumi - 8*sum)` cost FADD (undo the magic seed) + 2 FFMA.
Folding the seed into a per-(token, block) constant `c = d8*1.5*2^23 + 8*sum`,
computed once per stage per token rather than per value, makes it
`fma(d4, fma(d8, bits(magic + sumi), -c), acc)`. The one new rounding (half an
ulp of c, i.e. half a unit of sumi) is the size of the fp16 rounding Q8_1
already applies to `8*sum`; `make test-prefill` now carries that exact per-block
allowance and fails on anything beyond it. Dense 62.8 -> 67.4 TMAC/s, MoE 58.7 ->
61.4. With the cheaper epilogue the SwiGLU-fused gate|up kernel became 3%
faster end to end, so it is now the default.

**Producer/consumer GEMM: built, measured slower, not used.** Ablations of the
cfg-2 kernel (dense 6144x2048x4096) say where the time goes: copies alone 84
TMAC/s, the FP work alone 97, mma plus a single FADD 113, all three together
67 -- each part runs well above the whole, so the loss is in how they overlap,
which is what warp specialisation is for. `k_pf_mmq_ws` (cfgs 20-28) has 1, 2
or 4 copy warps feeding 8 compute warps through mbarriers, with no block-wide
barrier in the loop. Every shape was slower: 1 copy warp 50-57 TMAC/s, 2 warps
62.1, 4 warps 57.1. Without a bulk-copy engine (TMA arrives with sm_90) a few
warps cannot issue enough cp.async requests to feed the tensor cores, which is
why Ampere GEMMs (CUTLASS sm80) use the all-threads multistage pipeline this
kernel already has. The remaining GEMM headroom is in that overlap, and the
cheapest unexplored lever is fewer instructions per mma (the nibble unpack and
scale loads, amortised over wider warp tiles without losing the 2-blocks/SM
overlap).

Switches: `LK_PREFILL=seq|batch` (default batch), `LK_PREFILL_CHUNK` (8192),
`LK_PF_PROF=1` (per-stage GPU time), `LK_PF_CFG_DENSE` / `LK_PF_CFG_MOE`
(GEMM launch configs), `LK_PF_FUSE=0` (unfused SwiGLU; bit-equivalent),
`LK_PF_ATTN8=0` (fp16-score attention), `LK_PF_A8_NEAR` (exact window, 1024),
`LK_PF_A8_SINK` (exact tile 0, 1), `LK_PF_KSMOOTH=1` (key smoothing). Build-time:
`PF_EPI2=0` (3-op GEMM epilogue), `PF_ABL` (ablation bitmask, bench only).
Diagnostics: `LK_PF_ATTN_CMP`, `LK_DUMP_KV`, `LK_PF_DUMP`, `LK_DBG_IL` /
`LK_DBG_OUT`, `LK_ATTN_DUAL`, `LK_VERIFY_PACK`.

## Correctness

Three gates, all measured against references that can actually adjudicate.

**C1 — greedy parity vs llama.cpp.** The pre-Phase-4y build is **EXACT, 66/66**
on `factual` and 191/825 on the full 7-case suite. **The default build is not: it
matches 88/825.** Phase 4y reassociates `(x*inv)*w` into `inv*(x*w)` at the 80
sites where an RMSNorm is folded into its producer, which rounds the Q8_1
activation scales one fp16 ulp differently; the residual stream amplifies that
~20% per layer, so the greedy trajectory is no longer reproducible. That is a
real cost and it is stated rather than buried. It is also inherent: no
barrier-free norm fusion can reproduce `(x*inv)*w` bit-for-bit, because the
producer cannot know `inv`. What adjudicates is C2 and the end-to-end loss, both
of which are unmoved or better -- see below. `make BIGBLOCK=0 NORM_ACCUM=0
NORM_GEMV=0` restores the 191/825 build, verified bit-identical to it.

Multi-step greedy identity is chaotic and
this is *demonstrated*, not asserted -- see "How fragile greedy parity is" below;
the suite score moves by tens of tokens when a kernel's rounding changes by 1e-8,
so treat it as a smoke alarm rather than a measurement.

## Context scaling, against the oracle

llama.cpp was measured on the same GGUF, same GPU, over the same prompts. Its
decode rate is FLAT across the whole range -- which is what exposed the bug
below, because comparing our numbers only against our own short-context number
cannot detect a context-dependent defect.

Both engines were re-measured in the SAME session at the same prompt lengths
(`tools/oracle_bench.py`, which reads the token count back out of llama.cpp's own
timing line rather than assuming it).  An earlier version of this table compared
our fresh numbers against llama.cpp numbers from a different session and reported
a 1.76x short-context lead; the true figure is 1.62x, and the discrepancy was
cross-session drift of about 2%, not a difference between the engines.

| ctx | llama.cpp | liquid-kernel | ratio | our GB/s | % of 1008 peak |
|---|---|---|---|---|---|
| 159 | 257.9 | **417.4** | 1.62x | 612 | 60.7% |
| 872 | 254.2 | **412.1** | 1.62x | 604 | 60.0% |
| 1647 | 251.6 | **406.5** | 1.62x | 596 | 59.2% |
| 3228 | 247.4 | **398.6** | 1.61x | 585 | 58.0% |
| 6390 | 244.9 | **377.9** | 1.54x | 554 | 55.0% |

The context slope fell from 216 ns per context token to **40 ns** (llama.cpp's is
33 ns), and the short-context lead is now constant out to 3.2k rather than
decaying immediately.

### The bug that flat llama.cpp exposed

The attention score loop fetched one token's 64 dims -- 128 contiguous bytes,
exactly one cache line -- with **64 separate 2-byte loads**. At a fixed dim the
32 lanes of a warp sit on 32 *different* tokens, i.e. 32 different lines, so
that is 32 lines touched per instruction, 64 times over: 2048 L1 accesses per
warp per 32 tokens where 256 will do.

Loading the line as **8 x uint4** cuts the instruction count 8x. Dims are still
accumulated d = 0,1,2,...,63 in that exact order, so the result is bit-identical
(parity unchanged). 8k context went 243 -> 334 tok/s (+37%) and short context
also improved, 410 -> 417.

The same treatment does NOT work on the output scan: there each of the 64
threads reads a different dim of the *same* line, so it is already perfectly
coalesced, and batching its loads was measured worse **twice**. Two loops in one
kernel with the same symptom and opposite fixes.

At 8k context attention was 4.76 ms of an 8.14 ms step -- 58%, at 34 GB/s against
a 956 GB/s ceiling -- because one block per query head leaves 32 blocks on 84 SMs
and each scans the whole context twice. Split-K (32 heads x 8 context slices,
deterministic two-stage combine) fixes it, and costs nothing at short context
because empty slices retire immediately.

Three cheaper fixes were tried and falsified first: batching the V loads (no
effect), transposing the KV cache (worse -- token-major was already coalesced),
and widening the block to 1024 threads (neutral). Ablations put the score loop at
33.7 ns per context token and the V loop at 24.9, so both halves needed the
parallelism, not just the suspected one.

## How fragile greedy parity is

The split-K combine originally used `acc / l` where the serial kernel used
`acc * (1/l)`. That is a single rounding difference of order 1e-8.

At a one-token prompt the two kernels are **bit-identical**. At two tokens the
top-64 logits already differ by **0.276**, and at sixteen by **0.724** -- enough
to change a greedy token at position 39.

This is a clean, reproducible demonstration of why multi-step greedy identity is
not a usable correctness gate for this model, and why C1a (step-1 top-1 agreement
plus full-vocabulary correlation) replaced it. It also means the metric is
*scale-dependent*: a perturbation injected into the residual stream at every
layer is amplified by all 40 of them, while a perturbation in the final lm_head
is not amplified at all.

**C2 — teacher-forced perplexity.** 0.0222% from the exact model on 200 paired
positions of natural text, against a 0.1% target. **PASS.**

The reference matters here, and it is not llama.cpp. `tools/perplexity.py`
recomputes the lm_head in float64 straight from the GGUF and reports three
numbers, because comparing two engines cannot say which one is right:

| | nll | vs exact | sd per position |
|---|---|---|---|
| exact (float64 lm_head) | 1.804137 | -- | -- |
| **liquid-kernel** | -- | **0.022%** | -- |
| llama.cpp | -- | 2.0192% | 0.315 |

The engine is **91x closer to the exact model than llama.cpp is**, and the whole
two-engine "delta" is the oracle's error.  (The exact reference is recomputed
from the engine's own dumped residual stream, so this validates the final norm +
lm_head stage against float64 mathematics; the single-token comparison below
isolates the same effect with no upstream involvement.)

The same effect shows up at the single-token level. Feeding one prompt token to
each engine and recomputing the lm_head in float64:

| | max abs deviation at top-64 logits | mean |
|---|---|---|
| liquid-kernel | **0.0235** | **0.0083** |
| llama.cpp | 0.6496 | 0.1880 |

A numpy simulation of Q8_1 quantisation reproduces the engine's figures to five
decimals, so the engine's lm_head is exactly what its own quantisation predicts.
This is why `tools/c1a.py`'s "Pearson >= 0.9999 against llama.cpp" criterion is
not achievable by *any* implementation that quantises the activation
independently — it measures the oracle's error. C1a's top-1 agreement is 6/7.

## Negative result: speculative decoding is not worth building here

Phase 5 was gated on measuring how much consecutive tokens share experts.
`LK_ROUTE_LOG` records the real top-4 choices in-graph over 519 positions:

| K | E[distinct experts] | 4K | reuse | MB/token at 100% accept | break-even accepted |
|---|---|---|---|---|---|
| 1 | 4.00 | 4 | 1.00x | 1448 | 0.99 |
| 2 | 6.73 | 8 | 1.19x | 1107 | 1.51 |
| 4 | 11.22 | 16 | 1.43x | 888 | 2.42 |
| 8 | 17.91 | 32 | 1.79x | 720 | 3.93 |

Routing is almost perfectly uniform — entropy 4.142 of a maximum 4.159 nats, no
expert unused, most-used expert only 1.5x uniform. Consecutive tokens barely
share experts, while lm_head must produce logits at every verified position and
so costs K x 215 MB per pass regardless of reuse. **Phase 5 would have been a net
loss.** The measurement cost an afternoon and cancelled a week of work.

## Where the time goes

`LK_PROFILE=1` brackets every kernel launch in its own event pair and aggregates
GPU time by category. **Price the instrument first** (`make bench-launch`): an
event bracket costs ~2.1 us, and a CUDA-graph node costs 0.606 us. Without that
calibration, ~200 kernels measuring "3.2 us" look like launch overhead when they
are in fact doing nothing.

At 390.3 tok/s the step is 2.56 ms:

| | |
|---|---|
| kernel execution | 2.24 ms |
| graph dispatch | 0.31 ms (509 nodes x 0.606 us) |
| **total** | **2.55 ms** vs 2.56 ms measured |

Against the 1.534 ms bandwidth floor that leaves 0.70 ms of non-bandwidth time:
router 167 us, 81 rmsnorms 195 us, GEMV inefficiency 119 us, attention 34 us,
misc 78 us.

**550 tok/s is not reachable for this weight format on this part.** It needs
1.818 ms/step; the bandwidth floor alone is 1.534 ms and 509 graph nodes add
0.308 ms, giving 1.842 ms — 543 tok/s — with perfect bandwidth utilisation.
Reaching 550 would need to move fewer bytes per token, and the one lever for that
was measured dead above. The realistic maximum is ~450-480 tok/s.

## The four things that actually mattered

1. **A kernel with one block cannot be fast, however many threads it has.** Three
   of the engine's most expensive kernels were single-block reductions holding
   ~1 KB of loads outstanding: at ~500 cycles of HBM latency that is ~2.8 GB/s, a
   hard ceiling independent of block size. Splitting the router across 64 blocks
   and the argmax across 125 took them from 25.7 -> 4.5 us and 145 -> 6 us. Each
   had previously been "optimised" by adding threads, which could never work.
2. **Rank kernels by work-per-thread, not by traffic share.** The router carried
   1.4% of the bytes and was the single largest cost in the engine.
3. **A two-implementation comparison cannot say which implementation is right.**
   Compute the exact answer from the weights; here it costs one float64 GEMM.
4. **Measure the gate before building what it gates.** See Phase 5 above.

## Why 550 is out of reach at batch 1, and what would reach it

At batch 1 this workload is neither compute-bound nor bandwidth-bound.  It is
**fixed-cost-bound**, and that is what bounds the lead over llama.cpp:

| | tok/s | effective GB/s | % of the 956 GB/s achievable |
|---|---|---|---|
| llama.cpp | 257.9 | 378 | 40% |
| liquid-kernel | 417.4 | 612 | 64% |
| roofline (100% of achievable) | 652 | 956 | 100% |

Both engines must stream the same 1466.75 MB per token -- that is the format, not
an implementation choice, and we already read only the 4 of 64 experts the router
selects (807 MB of the 1466.75).  So the only thing an implementation can win at
batch 1 is bandwidth efficiency, and the largest possible multiple over llama.cpp
is 956/378 = 2.5x, of which 1.62x is taken.

The 0.68 ms between our 2.40 ms step and the 1.53 ms bandwidth floor is itemised:
0.27 ms launching 461 kernels at ~0.6 us each, 0.30 ms inside 80 normalisation
kernels and the router that each reduce 2048 numbers in a single block and cannot
be made wider, 0.11 ms of compute-only kernels.  None of it is bandwidth.

The way past it is batching, which amortises every weight byte across the tokens
in the batch.  MoE reuse is the constraint, and the number depends on a
distinction that is easy to get wrong in the optimistic direction: G4b measured
K *consecutive* tokens (the speculative-decoding question), but a batch holds B
tokens from B *different* sequences, which share only the static marginal expert
skew.  `tools/g4c.py` measures both from the same router trace.

    B    consecutive   independent   uniform baseline
    4    11.22 (1.65x) 12.89 (1.51x) 14.56 (1.39x)
    8    17.91 (2.04x) 21.07 (1.83x) 25.81 (1.60x)

Batching uses the **independent** row: 971.7 MB/token at B=4 and 799.6 MB/token
at B=8, against 1447.9 at B=1.  Those are byte floors, before any loss of
bandwidth efficiency at larger working sets, so the realistic gain at batch 4 is
**~1.35-1.45x** rather than the 2x an earlier revision of this file claimed from
the consecutive-token figure.  The uniform row is the floor of what is even
theoretically available, and the measurement beats it, which is the whole reason
the MoE is worth exploiting.

**How much of the step can a batch amortise?** `tests/bench_shapes.cu` builds a
graph of 343 EMPTY kernels with the engine's exact grids and block sizes -- the
tollbooth floor, with no memory traffic at all:

    343 nodes, empty, exact shapes   :  0.268 ms/token  =  0.782 us/node
    the real step at ctx 272         :  2.269 ms
    the toll as a fraction of it     :  11.8%

So the step decomposes as 0.268 ms of launch toll + 1.557 ms of wire + 0.444 ms
of latency-bound execution, and **only the first term amortises across a batch**.
Batch-4 therefore lands at **1.47-1.88x** depending on whether the execution term
is latency-bound (constant) or work-bound (4x), against a pure byte floor of
1.51x.  The equivalent shift for speculative verification is a cost ratio of
1.98-2.56x rather than the 2.42x assumed, i.e. a break-even of 1.98-2.56 accepted
tokens of 4 instead of 2.42.

## Architecture ground truth

Extracted from llama.cpp `src/models/lfm2.cpp` (which `lfm2moe.cpp` aliases):

- **RoPE: NEOX half-split**, `n_rot=64`, `theta=1e7`, `freq_scale=1`, no YaRN;
  applied to Q and K per head. `n_rot = head_dim`, so no partial rotation.
- **ShortConv x30**: `in_proj -> [B|C|x]` -> `bx = B*x` -> causal *unflipped*
  3-tap conv -> `y = C*conv_out` -> `out_proj` -> residual. **No non-linearity.**
  State = last two `B*x`, F32, `[tap, channel]`, tap-fastest.
- **Router is SIGMOID** (`expert_gating_func=2`): top-4 by `sigmoid + exp_bias`
  (bias for **ranking only**), weights from the **unbiased** sigmoid,
  renormalised with `clamp(sum, 6.103515625e-5, inf)`. Routers are F32 and
  consume the **post-`ffn_norm`** hidden state in F32.
- Q/K per-head RMSNorm(64) **before** RoPE; V unnormalised; attention scale is
  exactly 0.125; SwiGLU `down(silu(gate)*up)`; attention iff `layer % 4 == 2`;
  dense FFN only at layers 0-1; no SWA.

Recon facts that corrected earlier assumptions:

1. **The GGUF is 100% standard v3.** 428 tensors, `general.alignment=32`, data at
   byte 8,250,912, offsets contiguous, no trailing bytes. Q4_0 x220, F32 x207,
   **Q6_K x1** (`token_embd.weight`). ggml type 14 *is* Q6_K; there is no custom
   format.
2. **QAD = Quantization-Aware Distillation**, a retrained checkpoint. Its weights
   do **not** match public `LiquidAI/LFM2.5-24B-A2B` (embedding correlation
   -0.036), so the public checkpoint is not a valid reference.
3. The only valid external oracle is llama.cpp on this exact GGUF.

## Layout

```
setenv.sh                  portable env (host + container); resolves CUDA 12.6
Makefile                   nvcc-driven build; gate and oracle-tool targets
scripts/oracle.sh          start|stop|status the llama.cpp oracle (GPU 1)
scripts/setup-toolchain.sh installs CCCL headers into the assembled CUDA tree

src/quant.cuh              quantisation types, dp4a dots, Q8_1 quantiser
src/kernels.cuh/.cu        all forward-pass kernels
src/engine.cu              weights staging, 40-layer forward, CUDA graph, modes
src/tensor_table.h         generated: 428 tensor records

tools/gguf.py              GGUF reader + reference dequantiser (vs gguf-py)
tools/repack.py            lossless repack into the planar (SoA) layout
tools/verify_pack.py       bit-exactness of the repack
tools/capture_ref.py       oracle ground truth + baseline timings
tools/make_suite.py        freezes the C1 parity suite as raw token IDs
tools/oracle_logits.cpp    full 128k logits from libllama (HTTP only gives top-k)
tools/oracle_ppl.cpp       whole-corpus teacher-forced pass, one decode
tools/parity.py            Gate C1
tools/c1a.py               Gate C1a
tools/perplexity.py        Gate C2, with the exact float64 reference
tools/g4b.py               routing-skew analysis, gates Phase 5

tests/test_quant.cu        dot products vs exact integer reference (Gate G2)
tests/test_kernels.cu      RMSNorm/RoPE/ShortConv/router (Gate G2)
tests/bench_launch.cu      launch / event-bracket / graph-node cost
```

The engine is pinned to **GPU 0**; the oracle owns **GPU 1**.

## Usage

```bash
source ./setenv.sh
scripts/setup-toolchain.sh   # once
make devinfo                 # L2 / bandwidth capability probe
make test-all                # Gate G2: kernel primitives
scripts/oracle.sh start      # llama.cpp oracle on GPU 1
make oracle-tools            # builds oracle_logits + oracle_ppl (needs LLAMA_DIR)

make bench-launch            # price the profiler before trusting it
LK_PROFILE=1 ./build/engine prof <ids...>     # per-kernel GPU time
LK_BENCH=1 LK_NGEN=128 ./build/engine gen <ids...>

make gate-c1 gate-c1a gate-c2
```

Engine modes: `gen` (generate), `topk` (top-10 logits), `ppl` (teacher-forced
perplexity), `prof` (per-kernel profile).

Diagnostics: `LK_NO_GRAPH` (the dump hooks use memcpy/sync, which are illegal
inside a graph capture), `LK_DEBUG_STEPS`, `LK_DUMP_LAYERS`, `LK_LOGIT_DUMP`,
`LK_ROUTE_LOG`, `LK_Q6K_SCALAR` (falls back to the scalar lm_head for A/B).

## Status

- [x] **G0** repo, toolchain, oracle on GPU 1, baseline 227.07 tok/s, parity suite
- [x] **G1** repack, all 428 tensors bit-exact
- [x] **G2** all kernel primitives verified against independent references
- [x] **G3/C1** greedy parity: factual bit-identical 66/66
- [x] **G3/C2** perplexity 0.0402% from exact (target 0.1%) — **PASS**
- [x] **G4b** routing skew measured; speculative decoding **rejected on evidence**
- [x] **G6** attention equivalence against both the serial kernel and exact
      float64 — **PASS**, and it caught a real bug (see Phase 4v)
- [ ] **G4/C4** 417.4 tok/s vs the 550 target — **not met, and shown unreachable
      at batch 1 for this weight format**.  The roofline is 652 tok/s and the
      practical ceiling with the current kernel structure is ~450.  See
      "Why 550 is out of reach here, and what would reach it" below
- [ ] **G5** speculative decoding — cancelled by G4b
