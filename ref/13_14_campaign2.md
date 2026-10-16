
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
