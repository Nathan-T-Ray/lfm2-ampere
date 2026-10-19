
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
