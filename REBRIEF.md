# REBRIEF — liquid-kernel, the multi-token economics, and what to attack next

Written for an external reviewer. Everything here is either measured on the
machine or is arithmetic that you can check by hand. Where a number is a model
rather than a measurement, it says so.

**Read section 0 first.** Three of the four claims in the last review were
computed from a model this project refuted a phase earlier, and one of them
praised the refutation in the same paragraph. But chasing them down found a
genuine error in *my own* model, and it changes a verdict I had recorded as
settled. That correction is section 1, and it is the reason this document
exists.

---

## 0. THE HEADLINE, AND WHAT CHANGED

| | |
|---|---|
| single-stream decode, ctx 272, one RTX 3090 Ti | **443.04 tok/s** |
| batch-4 aggregate, same one card | **682 – 827 tok/s** |
| single-stream with triggered speculation, `factual` workload | **695 – 849 tok/s** |
| single-stream with triggered speculation, averaged over 7 workloads | break-even (0.96x of the bar) |
| batch-1 ceiling **within the current 343-node graph** | 488.9 tok/s |
| batch-1 if the launch toll is removed entirely (section 5, now measured as additive) | **500.5 tok/s** |
| roofline | 652 tok/s |

The 488.9 ceiling and the 500.5 figure are not in conflict. 488.9 was measured by
filling the small kernels while **keeping** the 343 nodes; 500.5 comes from
removing the node boundaries themselves, which is a structural change to the
graph rather than an optimisation inside it.

Two things moved this phase, both in the byte model:

1. **Every prior analysis overcharged `lm_head` by one read per multi-token
   pass.** A K-position verification pass needs `K-1` logit vectors, not `K`,
   because the first one is what the previous step already computed. Fixed
   215.04 MB per pass, so it matters most at small `K`.
2. **`FIXED_MB` never had a router term.** `router_dot` is charged 524288 B per
   layer by the profiler's own table; over 38 MoE layers that is 19.92 MB, and
   it is the whole of the unreconciled 18.85 MB "known inconsistency #5".

Correcting those lowers every multi-token break-even bar by 4.4 – 8.2%, and it
moves the aggregate speculation verdict from *12–23% short* to *0.6% short*.

---

## 1. THE CORRECTIONS, WITH ARITHMETIC

### 1.1 `lm_head` is charged one time too many

`tools/g4b.py` and `tools/g4c.py` both computed, and this project's
`DIAGNOSIS.md`, `ARCHITECTURE.md` and `tools/lookup.py` all inherited:

```python
per_pass = K * LM_HEAD_MB + 38 * U_K * EXPERT_MB + FIXED_MB
```

`lm_head` is 215.04 MB (128000 x 2048, Q6_K) — **14.7% of a single-token step**,
the second largest term after the MoE experts. It is the most expensive line
item to get wrong.

A verification pass over `K` positions does produce `K` logit vectors, but the
**first** one is the vector the previous step already computed to emit the token
that seeds the pass. It is in hand. So the pass computes `K-1` of them:

| K (positions) | drafts | `lm_head` charged | `lm_head` needed | phantom MB/pass |
|---|---|---|---|---|
| 1 | 0 | 1 | 1 | 0 |
| 2 | 1 | 2 | **1** | **215.04** |
| 3 | 2 | 3 | **2** | **215.04** |
| 4 | 3 | 4 | **3** | **215.04** |
| 8 | 7 | 8 | **7** | **215.04** |

It is a **fixed** 215.04 MB, independent of `K`. That is precisely why it was
invisible: every headline number this project publishes comes from `K=1`, where
the charge is correct.

Batching is different and must not be confused with it: `B` *independent*
sequences each need their own logits to continue, so batching runs `lm_head`
`B` times. There is no redundancy and no acceptance rate. `tools/lkbytes.py`
implements both and they are separate functions for exactly this reason.

### 1.2 The missing router term resolves inconsistency #5

`ARCHITECTURE.md §9.6` records an unreconciled 18.85 MB gap between the batching
model's `1447.9 MB/token` and `§2.1`'s measured `1466.75`. `lk_prof_bytes` in
`src/engine.cu` contains:

```c
{"router_dot",    524288.0},   /* 524288 B x 38 MoE layers = 19.92 MB */
```

`FIXED_MB = 283.0 + 59.0 + 81.0 + 3.0` has no router row and never did.
Adding it gives 1467.88 MB against the measured 1466.75 — a **1.13 MB / 0.077%**
residual, which is the rounding in `283.0` (true value 283.85, from
`30 x (shortconv_in 7077888 + conv 24576 + out 2359296)`). The gap is itemised.
**Inconsistency #5 is closed, and it is worth 1.3% of every batch number.**

### 1.3 The bracket, and why the document had two answers for one quantity

`T(1) = 2.266 ms` decomposes, every term measured:

```
1.557 ms  wire, 1466.75 MB at the 941.8 GB/s marginal rate
0.220 ms  streaming size inefficiency -- small kernels do not amortise ramp
0.268 ms  launch toll, 343 nodes x 0.781 us, from an empty-kernel graph
0.221 ms  the 158 small kernels: execution plus the stalls they impose
```

Three of those scale differently with the token count of a pass, and two are
genuinely unknown. Rather than pick the flattering end:

- **TOLL** is per *node*, the node count barely moves, so it is **constant** and
  amortises. This is the only term batching gets for free.
- **WIRE** is per *byte*, so it scales with `R = pass_MB / base_MB`.
- **RAMP** is per *kernel per pass*. Weight bytes are shared across the pass so
  the kernels do not get bigger — but the grids do grow. **Bracketed.**
- **SMALL** is per *token* of activation work — but it also contains dependency
  stalls, which are not. **Bracketed.**

```
lower = R*WIRE + TOLL + RAMP + SMALL            (only bytes scale)
upper = R*(WIRE+RAMP) + TOLL + SMALL*ntok       (everything scales)
```

**The project had already published both ends of this bracket without noticing.**
`DIAGNOSIS.md:1419` gives M=4 as `1.98x .. 2.56x`; `DIAGNOSIS.md:1675-1683` gives
M=4 as `2.11x .. 2.41x`. They are not in conflict. They are the two ends of a
bracket nobody wrote down. `tools/lkbytes.py::_selftest` now asserts that the
legacy model reproduces both, which makes the bracket a *check* rather than an
excuse.

### 1.4 The corrected break-even bars

`bar` is the pass time as a multiple of one plain step, which **is** the
break-even `E[tokens/pass]`: a pass must emit more than this to win.

**Speculation** (consecutive tokens; the seed's logits are already held):

| M | drafts | lm_head | U_M | pass MB | bar corrected | bar legacy | change |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 1 | 4.00 | 1467.9 | 1.001 | 0.991 | — |
| 2 | 1 | 1 | 6.73 | 2019.4 | **1.259 – 1.393** | 1.350 – 1.497 | −6.8% |
| 3 | 2 | 2 | 9.12 | 2716.1 | **1.585 – 1.863** | 1.677 – 1.967 | −5.5% |
| 4 | 3 | 3 | 11.22 | 3355.2 | **1.885 – 2.302** | 1.976 – 2.407 | −4.6% |
| 8 | 7 | 7 | 17.91 | 5564.5 | 2.920 – 3.874 | 3.011 – 3.978 | −3.0% |

**Batching** (independent sequences, `U` from the *independent* row):

| B | U_B | pass MB | MB/token | speed-up corrected | → tok/s |
|---|---|---|---|---|---|
| 1 | 4.00 | 1467.9 | 1467.9 | 1.000 | 443 |
| 2 | 7.35 | 2358.7 | 1179.2 | 1.270 – 1.411 | 563 – 625 |
| 4 | 12.89 | 3906.3 | 976.7 | **1.540 – 1.867** | **682 – 827** |
| 8 | 21.07 | 6416.5 | 802.1 | 1.848 – 2.410 | 819 – 1068 |

**Batching must use the `independent` row, not `consecutive`.** A batch holds `B`
tokens from `B` *different* sequences, which share only the static marginal
expert skew. Using `consecutive` is optimistic by exactly the amount that is
easiest to get wrong. The tool prints both, labelled.

---

## 2. WHAT WAS MEASURED — THE FULL 2-D SWEEP

The previous revision of `tools/lookup.py` hardcoded

```python
ratio_opt, ratio_pes = 2.11, 2.41      # the M=4 cost, applied to every M
```

and never varied it. So the project's recorded verdict — *"speculative decoding
… measured dead"* — was measured at **one point in a two-dimensional space**
(`minmatch` x `drafts`) and reported for all of it. M=4 is the **worst** point on
the curve: the bar rises ~0.3x per extra draft while the marginal acceptance
decays.

Now swept over `minmatch` 1–6 x `drafts` 1–3. `E` is exact — greedy speculation
with exact verification is lossless, so the acceptance rate can be simulated
offline from trajectories the engine already generates. `ngen=384`, 7 cases.

`tools/lookup.py --ngen 384 --matches 1,2,3 --drafts 1,2,3` →
`ref/lookup_sweep.log`.

### 2.1 `factual` pays, and it is the only thing that does

| config | trigger | E | bar | E/bar | tok/s | verdict |
|---|---|---|---|---|---|---|
| **M=4, minmatch=2** | 88.6% | **3.61** | 1.885–2.302 | **1.57 – 1.92** | **695 – 849** | **PAYS** |
| M=3, minmatch=2 | 88.6% | 2.75 | 1.585–1.863 | 1.47 – 1.73 | 653 – 768 | PAYS |
| M=2, minmatch=2 | 88.6% | 1.88 | 1.259–1.393 | 1.35 – 1.49 | 597 – 661 | PAYS |
| M=2, minmatch=1 | 92.4% | 1.72 | 1.259–1.393 | 1.23 – 1.36 | 546 – 604 | PAYS |

`factual` is where the model restates the same sentence in a visible reasoning
trace, so prompt-lookup finds a long verbatim match and the continuation is
right **99.3%** of the time (`E = 1 + 0.886 * p1 = 1.88` → `p1 = 0.993`).

### 2.2 Nothing else pays at the aggregate

| config | E | E/bar (pessimistic) | tok/s | verdict |
|---|---|---|---|---|
| M=2, minmatch=2 | 1.34 | 0.965 | 427 – 473 | BREAK-EVEN |
| M=2, minmatch=1 | 1.34 | 0.960 | 425 – 471 | BREAK-EVEN |
| M=3, minmatch=2 | 1.62 | 0.872 | 386 – 454 | BREAK-EVEN |
| M=4, minmatch=2 | **1.873** | 0.814 | 360 – 440 | DOES NOT PAY |

**That last row is the whole story of the correction.** `E = 1.873` against a
legacy bar of `1.976–2.407` was 5.5–21.9% short — clearly dead. Against the
corrected bar of `1.885–2.302` it is **0.6% short of the optimistic end**. The
verdict did not flip, but it moved from *comfortably dead* to *on the line*, and
the case breakdown shows the win is concentrated exactly where the text is
copied.

### 2.3 The binding constraint is precision, not trigger rate

`code` at `minmatch=1` triggers **67.5%** of the time and still gets `E = 1.27`,
because when it fires the continuation is right only **40%** of the time. Compare
`factual`: 92.4% trigger, 99.3% precision.

Per-case first-draft accuracy `p1`, derived from `E = 1 + trigger * p1`:

```
factual    99.3%        reasoning  41.3%
                        code       40.0%
                        agentic    40.0%
                        prose      36.0%
                        math       31.8%
```

These are profiles, not English. Grouping them by language labels says nothing.

**The arithmetic that matters.** To clear 500 tok/s on one card you need
`E / bar_lo > 500 / 443.04 = 1.129`:

- at **M=2**, `E > 1.129 x 1.259 = 1.421`. With a 90% trigger, `p1 > 0.468`.
- at **M=4**, `E > 1.129 x 1.885 = 2.128`. With a 90% trigger and uniform `p`,
  `p > 0.418`.

**Prompt-lookup delivers `p1 ≈ 0.40` on six of seven cases. The requirement is
`p1 ≈ 0.42–0.47`. It misses by a hair, on a threshold, not by a mechanism.**

### 2.4 The monotone cross-check passes

`tools/lkbytes.py::monotone_lower_bound` is pure arithmetic: `E(K) = 1 + sum(p_i)`
with `p_1 >= p_2 >= ...`, so the mean per-draft survival
`(E(K)-1)/(K-1)` is a **lower bound** on `p_1`, hence
`E(want) >= 1 + (want-1) * (E(K)-1)/(K-1)`.

```
E(K=4, minmatch=2) = 1.873  ->  E(K=2) >= 1.291
   simulated E(K=2, minmatch=2) = 1.344   OK
   simulated E(K=3, minmatch=2) = 1.625   OK
```

The simulation is *above* its own bound, as it must be. This is the one place in
the speculation analysis where a bug would be caught rather than believed.

### 2.5 What this does NOT establish

The `E` values are exact. **The bars are a model, not a measurement** — no
M-pass kernel exists yet. `1.885` at M=4 assumes the step decomposition scales
the way section 1.3 says it does, and the bracket is wide precisely because two
of the four terms are unknown. **The build must re-measure the bar before the
payoff is claimed.**

---

## 3. THE REFUTATIONS — DO NOT RE-DERIVE THESE

| claim | status |
|---|---|
| "0.712 ms of tollbooth dead air, 343 steps x 2.07 us" | **REFUTED by measurement.** `ARCHITECTURE.md §3.5`: `bench_shapes` mode 0 gives **0.268 ms, 0.781 µs/node**. The 2.07 µs model is wrong by 2.65x. It also disagrees with the fusion experiments (T1 deleted 38 nodes for +0.67% = 0.40 µs/node). |
| "you forgot the tollbooth when computing batching" | **FALSE.** `DIAGNOSIS.md:1409` already says *"Only the first term amortises across a batch"*, and `:1414` already brackets `0.444 ms if latency-bound, 1.776 ms if it scales with work`. The toll was amortised in the original analysis. |
| "the proposed 2.01x / 1.84x" | **ALREADY ANSWERED.** `DIAGNOSIS.md:1424-1428`, verbatim: *"the proposed 2.01x verification ratio is the best case of a 1.98-2.56 range, and the proposed 1.84x batch speedup is the best case of a 1.47-1.88 range."* |
| "break-even is 2.01, and E is 1.87" | **THEIR OWN NUMBERS LOSE.** 1.87 < 2.01. The correction was proposed as a rescue and does not perform one. The corrected bar is 1.885–2.302, and 1.873 is still short. |
| "20 of 84 SMs idle (64-block grid) costs bandwidth" | **MEASURED FALSE.** `DIAGNOSIS.md:1216-1228`: `(64,1024)` = **737.9 GB/s** against a best config of **752.9** — a **2.0%** spread across six shapes. The 17% spread the profile table showed between `moe_down` and `moe_gate_up` was the **1.40–1.49 µs event bracket**, i.e. instrument, not silicon. |
| "Batch-8 crosses 1,000 tok/s" | **ABOVE THE BYTE FLOOR.** `g4c.py` gives 1.83x at B=8 = 811 tok/s. 1,033 amortises a term that scales with bytes. |
| "batch and verification use the exact same M=4 kernels" | **FALSE, and it is the important one.** Batching has no correctness constraint beyond being right. Verification must be **bit-identical to M=1** or the word "lossless" is false. Different bars, different kernels. |
| "greedy parity is a usable gate" | **UNUSABLE.** The residual stream amplifies perturbations ~20% per layer with two hard discontinuities per layer (Q8_1 rounding, discrete top-4 routing) — `DIAGNOSIS.md:26-54`. 40 layers is ~1000x. |

**Credit where it is due:** the *conclusion* — that byte reduction dominates and
batching is the largest remaining lever — matches `DIAGNOSIS.md:1424` and is
correct. The error is the mechanism and the magnitude, not the direction. And
`dense_gate` + `dense_up` merging into one `dim3(1472, 2)` launch is a real
4-node saving (≈0.07%), free.

---

## 4. THE BLOCK-IDENTITY RULE

This is what decides whether an M-pass build is even possible in the "lossless"
form, and it is why mode A (batching) is much easier than mode B (verification).

From `src/kernels.cu`, `k_moe_down_accum`:

```c
const int row = blockIdx.x * 32 + tw;          /* ONE OUTPUT ROW PER WARP */
...
for (int k = 0; k < LK_TOP_K; ++k) {
    float acc = 0.0f;
    for (int j = lane; j < bpr; j += 32)       /* lane-strided accumulator */
        acc += dot_q4_0_q8_1(qrow, drow, j, xk + j);
    a += d_w[k] * warp_sum(acc);               /* warp tree, then k = 0..3 */
}
```

The entire reduction order for a row lives **inside one warp**: a lane-strided
accumulator, a `warp_sum` tree, and the expert combination `k = 0..3` in order.
Therefore:

> **Which warp and which block computes an output row is free to change. The
> lane loop, the warp tree, the `k = 0..3` expert order, and the per-token Q8_1
> block ownership are not.**

That is a testable design constraint, not a hope. It means a fused M-token GEMV
can reassign rows to suit the expert union — a batching win — **as long as each
row's intra-warp reduction is byte-for-byte the same code path**. The same rule
applies to `k_attention_decode`: dims accumulate `d = 0..63` in order and tokens
accumulate `t = 0..n-1` in order with the same 256-lane halving tree
(`src/kernels.cu:1254`).

`tools/ab.py` gives a 7/7 bit-identity check for free on every build. **Any
M-pass kernel that is not 7/7 bit-identical is not a lossless speculative
verifier, whatever else it is.**

---

## 5. THE LAUNCH TOLL, MEASURED — AND A HYPOTHESIS OF MINE THAT IT KILLED

`ARCHITECTURE.md §4` derives the 0.268 ms launch toll **by subtraction**. The
whole megakernel proposal rests on that subtraction being dead air, and I wrote
a hypothesis in an earlier draft of this document that it might not be:

> *the bracket-corrected profiler sum exceeds the wall clock, which cannot happen
> unless kernels overlap, so most of the toll may already be hidden.*

**That hypothesis is now tested and it is wrong.** `tests/bench_launch.cu` part
(8): `N` nodes, each reading a distinct cold slab, **total bytes held constant at
512 MB**, grid fixed at 768x256 so only `N` varies.

```
    N    MB/node     wall us     us/node     pure us    overhead      GB/s
   16      32.00     651.670      40.729      35.628       5.101     823.8
   32      16.00     670.515      20.954      17.814       3.140     800.7
   64       8.00     708.886      11.076       8.907       2.169     757.3
  128       4.00     800.051       6.250       4.453       1.797     671.0
  256       2.00     911.974       3.562       2.227       1.336     588.7
  512       1.00    1209.037       2.361       1.113       1.248     444.0

  empty node, same N and grid :  1.05 - 1.15 us
```

The decisive comparison is **`busy_node ?= empty_node_toll + pure_transfer`**:

| N | empty toll | pure transfer | sum | measured | excess (= ramp) |
|---|---|---|---|---|---|
| 512 | 1.154 | 1.113 | 2.267 | **2.361** | +0.094 |
| 256 | 1.154 | 2.227 | 3.381 | **3.562** | +0.181 |
| 128 | 1.154 | 4.453 | 5.607 | **6.250** | +0.643 |

**The toll is additive, not hidden.** At the small-node end it accounts for
essentially the entire gap between measured and pure-transfer time. My
"already overlapped" hypothesis is refuted, and the argument from the profiler
sum is withdrawn as inconclusive: part (8b) runs the profiler's own instrument
over 343 *uniform* 1.5 MB nodes and gets `sum(brackets) = 1854 µs` against a
`2323 µs` wall — i.e. the instrument *under*-counts there, the opposite of what
it does on the engine's mixed small/large load. **The instrument's sign depends on
the node mix, so the engine's 2.978 vs 2.269 comparison cannot carry the weight I
put on it.**

### 5.1 What that does to the megakernel

The premise survives, and it is now supportable on a measurement rather than a
subtraction: **a kernel boundary costs ~0.78 µs (grid-weighted) and that cost is
real.** Removing only the 0.268 ms toll gives

```
2.266 - 0.268 = 1.998 ms  ->  500.5 tok/s
```

**which is exactly the target.** Recovering more of the other two terms:

```
0.268 ms  toll        -- replaced by intra-kernel handoff flags
0.220 ms  ramp        -- hidden if the next stage starts during the current one
0.221 ms  small exec  -- hidden behind the streaming traffic
--------
0.709 ms  ceiling 443 -> 642 tok/s if EVERYTHING were recovered
```

### 5.2 The one thing still untested, and it is the whole risk

**Nobody has measured whether an intra-kernel handoff costs less than a kernel
boundary.** 0.78 µs is the price of a node boundary; a band-local producer/consumer
flag plus a `__threadfence` has never been priced on this GPU. If it costs 0.8 µs
the megakernel buys nothing. If it costs 0.2 µs it buys 0.6 µs/node = 0.206 ms =
+10% = 487 tok/s on the toll alone.

That is a **microbenchmark, not a build**: two dependent stages inside one
persistent kernel, one flag, one fence, versus the same two stages as two graph
nodes. It should be written before any megakernel design work.

And the design risk is real and already measured: this project's own law is that
**placing a barrier in front of a bandwidth-critical kernel loses** —
`FUSE_ROUTER` 0.09–0.39%, `T4r` 0.17%, `T2` 1.75%, three attempts, three losses.
The counter-argument is that a megakernel replaces barriers that already exist as
kernel boundaries rather than adding new ones — a different comparison, but
genuinely uncertain.

### 5.3 The measured bandwidth collapse is a second, independent finding

The same table shows the achieved rate falling from **823.8 GB/s at 32 MB/node to
444.0 GB/s at 1 MB/node**. The engine's overall 650 GB/s sits inside that range,
which is the "0.220 ms streaming size inefficiency" of section 1.3 seen directly.
It also means **shaping the engine's kernels larger is worth more than the toll
in isolation suggests** — and it is a lever nobody has attacked at the kernel
level.

---

## 6. WHAT IS MEASURED DEAD — DO NOT RE-PROPOSE

Attack the *measurement* instead, and say what was wrong with the instrument.

- **Tensor parallelism, any second-GPU scheme.** Out of scope by explicit user
  instruction. No P2P on this platform anyway (`cudaDeviceCanAccessPeer = 0`
  both ways, `nvidia-smi topo -m` reports PHB).
- **Raising the power limit.** Already at the 450 W card maximum, drawing
  362.7 W, SM 2055/2100 MHz, MEM 10501/10501 MHz, every throttle reason
  `Not Active`, SW power-cap counter 0 µs.
- **Occupancy tuning / wave quantisation.** 1–2% across six configurations
  spanning 0.76–3.05 waves at constant bytes.
- **Clearing the launch toll by node deletion.** 0.3–0.45 µs/node measured
  against a 0.781 µs average toll. The arithmetic does not close. *(A megakernel
  is not this — it removes the node boundary itself. See section 5.)*
- **Fusing into `moe_gate_up`.** Three attempts, three losses.
- **Router MLP widening.** Flat at `ROUTER_UNROLL` 16/32/64.
- **Hoisting the four `moe_down` dots above their reductions.** 12.77 vs 12.32 µs.
- **Batching the attention V loads; transposing the KV cache; 1024-thread
  attention blocks.** All neutral or worse.
- **`bpr = 48` leaving lanes 16–31 idle.** Falsified by the `shortconv_in`
  control: `bpr = 64`, no half-empty iteration, same 7.078 MB/call, same
  622.7 GB/s. And 96 bytes is 3 x 32-byte sectors, so nothing splits.
- **`bench_shapes` mode 1** (every node reads its real bytes, no dependencies).
  Invalid instrument: 440 GB/s, *slower* than the real engine on more traffic,
  because its grid-stride loop is not unrolled. Recorded rather than deleted.

---

## 7. ASKS TO THE GROUP

**A1 — the one that decides a phase.** The corrected `M=2` bar is
`1.259–1.393` and prompt-lookup delivers `p1 ≈ 0.40` against a `0.42–0.47`
requirement. Is there a **better single-token predictor** that is (a) lossless
under exact verification and (b) cheaper than `0.26` of a step? A draft *model*
is now viable in principle — it only has to cost less than 0.26–0.39 of a step —
but this QAD checkpoint has **no available sibling**: `LiquidAI/LFM2.5-24B-A2B`
does not match (embedding correlation **−0.036**), so `p1` for any public small
model is unknown and probably poor. **Give me a concrete, buildable drafter for
this checkpoint, or say there is none.**

**A2 — the toll, now measured.** Section 5 shows the per-node toll is **additive**
(`busy = empty_toll + pure_transfer`), so removing it reaches **500.5 tok/s** on
paper. The untested step is whether an **intra-kernel handoff** costs less than a
kernel boundary's 0.78 µs. Design that microbenchmark: two dependent stages, one
flag, one fence, inside one persistent kernel, against the same two stages as two
graph nodes. If the handoff is not clearly cheaper, say so and **close the
megakernel before it is built.**

**A3 — the bracket.** Section 1.3 brackets `RAMP` and `SMALL` because their
scaling with pass size is unknown. Which end is right, and what measurement
settles it without building the M-pass? A microbenchmark that runs the 158 small
kernels at 1x and 4x token counts would do it.

**A4 — the `lm_head` correction.** Section 1.1 claims a K-position pass needs
`K-1` logit vectors. **Find the hole.** If the seed's logits are *not*
reusable — if there is any reason the first position of a pass must recompute
its own logits — the correction is wrong and every bar above moves back up.

**A5 — the fp16 range audit, still never run.** `ARCHITECTURE.md §9.1`. Phase 4y
carries a scale invariant through `block_q8_1`'s `ds.x`/`ds.y` on the claim
`dot(inv*u) = inv*dot(u)`. That is an *assumption* about fp16 range, not a
measurement. Construct an input that drives `ds.x` or `ds.y` out of range given
the architecture, or run the audit. This is the one genuine reviewer hole I know
of.

**A6 — bit-identity.** Is the section 4 rule *sufficient* for bit-identity, not
just necessary? Specifically: does `warp_sum` depend on the warp's *identity*
anywhere, or only on the sequence of lane values? Read the code and tell me.

**A7 — most important.** Tell me what is **wrong** in this document. This
project retracted its own claims three times during the architecture writeup and
twice more this phase, and every retraction was worth more than the optimisation
it cancelled. A refutation with arithmetic attached is the most valuable thing
you can send.
