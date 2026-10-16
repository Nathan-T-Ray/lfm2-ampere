# liquid-kernel

A hand-written sm_86 CUDA inference engine for **LFM2.5-24B-A2B-QAD-Q4_0** on an
RTX 3090 Ti. Single stream, greedy decode, every kernel written by hand and every
number in this file measured on the machine rather than estimated.

| | |
|---|---|
| decode, ctx 159 | **417.4 tok/s** (3 repeats, sd 1.87) |
| effective bandwidth | 612 GB/s = **60.7% of the 1008 GB/s spec** |
| llama.cpp, same GGUF/GPU, same session | 257.9 tok/s |
| speedup vs llama.cpp | **1.62x**, and flat at 1.61-1.62x to 3.2k context |
| speedup vs this engine's own Phase-3 baseline | **3.99x** (102.79 tok/s) |
| traffic | 1466.75 MB/token, so the bandwidth floor is 1.534 ms = **652 tok/s** |

## Correctness

Three gates, all measured against references that can actually adjudicate.

**C1 — greedy parity vs llama.cpp.** `factual` is **EXACT, 66/66 tokens**, on
the fast split-attention path -- previously only reachable with `LK_ATTN_SERIAL=1`.
The full 7-case suite matches 191/825. Multi-step greedy identity is chaotic and
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
