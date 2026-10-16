// bench_mkbar.cu - Phase 5, campaign 2: give the megakernel a fair trial.
//
// THE PROBLEM WITH THE FIRST VERDICT.  tests/test_megakernel.cu measured the
// grid barrier at 2.17-2.56 us against a 0.78-0.83 us kernel boundary and the
// megakernel was closed.  Three things make that verdict untrustworthy, and all
// three are confounds rather than findings:
//
//   1. FALSE SHARING.  The counters were 4 bytes apart in one 128B L2 line, so
//      every arrival RMW invalidated every poller.  (parts 1-3)
//   2. A HIDDEN LAUNCH.  cudaMemsetAsync reset the counters INSIDE the timed
//      region for the barrier mode and never for the graph mode.  A memset is a
//      kernel launch -- i.e. part of the 2.17 us was the very toll the megakernel
//      exists to remove.  (parts 1 vs 2)
//   3. THE WRONG REGIME.  Every stage was a large, balanced, homogeneous GEMV.
//      That is the regime where a kernel boundary is CHEAPEST and a barrier is
//      HARDEST (all blocks must arrive).  The megakernel's actual claim is about
//      the 158 SMALL nodes of the 343, which were never measured.  (parts 4, 5)
//
// The instrument is arranged so each factor moves alone:
//
//   1  shootout, counters reset OUTSIDE the timing   -> the real barrier cost
//   2  shootout, counters reset INSIDE the timing    -> reproduces the old number
//   3  shootout at a shape where half the blocks idle-> barrier's worst case
//   4  cadence sweep: do we need a barrier per stage?
//   5  SMALL-stage sweep: where does persistent actually start to win?
//   6  bit-identity for every variant (memcmp)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "megakernel.cuh"
#include "mkbar.cuh"

// exit(), not return: this macro is used inside void helpers, and a `return 1`
// there compiles to a silent early exit that would look like a result.
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(3); } } while (0)

static uint32_t g_seed = 12345u;
static uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }

#define REPS 5

// Refuse to run a persistent variant unless every block is resident: the barrier
// deadlocks otherwise, and a deadlock looks like a very slow kernel.
static void check_resident(int nb, int per_sm, const char * what) {
    if (per_sm <= 0) { printf("FATAL: %s not resident, refusing to run\n", what); exit(2); }
}

// ===========================================================================
// THE STAGE.  One real engine stage: the Q4_0 x Q8_1 GEMV ported in
// src/megakernel.cuh, run `nstages` times, optionally with a barrier between.
//
//   cad = barrier every `cad` stages.  cad > 1 makes the output WRONG by
//   construction (stages inside a group race), exactly like NOBAR; it is a cost
//   decomposition and is never a correctness path.  The traffic is unchanged,
//   which is why the timing stays meaningful.
// ===========================================================================
template<int V, int NOBAR>
__global__ void k_probe_b(const uint4 * __restrict__ qs, const __half * __restrict__ d,
                          const block_q8_1 * __restrict__ xq,
                          int out_rows, int bpr, float * __restrict__ y,
                          int nstages, int * __restrict__ bar,
                          lk_bp_t * __restrict__ bp, unsigned * __restrict__ leaf,
                          size_t stage_stride, int cad, unsigned gen0) {
    const unsigned nb = gridDim.x;
    unsigned gen = gen0;
    for (int s = 0; s < nstages; ++s) {
        mk_stage_gemv_q4_0(qs + (size_t) s * stage_stride,
                           d  + (size_t) s * stage_stride,
                           xq, out_rows, bpr, y, s > 0, nullptr, 0.0f);
        if (!NOBAR && (cad <= 1 || ((s + 1) % cad) == 0 || s == nstages - 1)) {
            ++gen;
            lk_bar_any<V>(bar, bp, leaf, nb, gen);
        }
    }
}

struct Ctx {
    const uint4 * qs; const __half * d; const block_q8_1 * xq;
    int rows, bpr, nstages, nb;
    float * y;
    int * bar; lk_bp_t * bp; unsigned * leaf;
    size_t stride; int cad;
    cudaStream_t st;
    unsigned gbase;         // running generation offset: lets the counter scheme
                            // run forever with NO reset, which is the whole point
};

// barriers performed by one pass, matching the kernel's own condition
static int nbar_of(int nstages, int cad) {
    int n = 0;
    for (int s = 0; s < nstages; ++s)
        if (cad <= 1 || ((s + 1) % cad) == 0 || s == nstages - 1) ++n;
    return n;
}

#define GEMV_V(V) k_probe_b<V,0><<<c.nb,256,0,c.st>>>(c.qs,c.d,c.xq,c.rows,c.bpr,c.y, \
                          c.nstages,c.bar,c.bp,c.leaf,c.stride,c.cad,gg)

// One case per variant.  A `default:` that silently launched the wrong barrier
// would make the table a lie, so an out-of-range variant is fatal instead.
#define DISPATCH(VAL, CALL) do { switch (VAL) { \
    case 0: CALL(0); break; case 1: CALL(1); break; case 2: CALL(2); break; \
    case 3: CALL(3); break; case 4: CALL(4); break; case 5: CALL(5); break; \
    case 6: CALL(6); break; case 7: CALL(7); break; case 8: CALL(8); break; \
    case 9: CALL(9); break; case 10: CALL(10); break; case 11: CALL(11); break; \
    case 12: CALL(12); break; case 13: CALL(13); break; \
    default: printf("FATAL: variant %d has no dispatch case\n", VAL); exit(2); } } while (0)

// mode 0 = separate launches (the graph equivalent), 1 = persistent + barrier,
// 2 = persistent no barrier, 3 = persistent with a REAL cudaGraph of the stage
static void one_pass(Ctx & c, int mode, int V, bool reset_inside,
                     const float * fbuf, float * fsink, size_t fstage, int THR) {
    unsigned gg = c.gbase;
    if (reset_inside) {
        CK(cudaMemsetAsync(c.bar, 0, 128, c.st));
        CK(cudaMemsetAsync(c.bp, 0, sizeof(lk_bp_t), c.st));
        CK(cudaMemsetAsync(c.leaf, 0, LK_LEAF_U32 * sizeof(unsigned), c.st));
        gg = 0;
    }
    if (mode == 0) {
        for (int s = 0; s < c.nstages; ++s)
            k_gemv_q4_0<<<LK_GEMV_BLOCKS(c.rows), 256, 0, c.st>>>(
                c.qs + (size_t) s * c.stride, c.d + (size_t) s * c.stride,
                c.xq, c.rows, c.bpr, c.y, s > 0, nullptr, 0.0f);
    } else if (mode == 2) {
        k_probe_b<0,1><<<c.nb,256,0,c.st>>>(c.qs,c.d,c.xq,c.rows,c.bpr,c.y,
                                            c.nstages,c.bar,c.bp,c.leaf,c.stride,c.cad,gg);
    } else {
        DISPATCH(V, GEMV_V);
    }
    if (!reset_inside) c.gbase = gg + (unsigned) nbar_of(c.nstages, c.cad);
}

// Zero the counters and rewind the generation offset.  THE MONOTONIC SCHEME
// REQUIRES THIS: `arrived` must be exactly (gen-1)*nb when generation `gen`
// begins, so a counter left over from a run with a different nb or a different
// pass count never satisfies the release equality and the barrier DEADLOCKS.
// This is called OUTSIDE every timed region, which is the entire point -- doing
// it inside is the confound part (2) exists to expose.
static void reset_bar(Ctx & c) {
    CK(cudaMemsetAsync(c.bar, 0, 128, c.st));
    CK(cudaMemsetAsync(c.bp, 0, sizeof(lk_bp_t), c.st));
    CK(cudaMemsetAsync(c.leaf, 0, LK_LEAF_U32 * sizeof(unsigned), c.st));
    c.gbase = 0;
}

static float time_mode(Ctx & c, int mode, int V, bool reset_inside,
                       const float * fbuf, float * fsink, size_t fstage, int THR) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    if (!reset_inside) reset_bar(c);
    for (int w = 0; w < 2; ++w) one_pass(c, mode, V, reset_inside, fbuf, fsink, fstage, THR);
    CK(cudaStreamSynchronize(c.st));
    CK(cudaEventRecord(a, c.st));
    for (int r = 0; r < REPS; ++r) one_pass(c, mode, V, reset_inside, fbuf, fsink, fstage, THR);
    CK(cudaEventRecord(b, c.st));
    CK(cudaStreamSynchronize(c.st));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms * 1000.0f / REPS / c.nstages;
}

// ===========================================================================
// SMALL STAGES.  The regime the first megakernel test never visited.  One stage
// here is a few hundred KB read by the whole grid, so the transfer is a fraction
// of a launch toll -- which is exactly the shape of 158 of the engine's 343
// nodes, and the only place a persistent kernel can plausibly win.
// ===========================================================================
__global__ void k_small_stage(const float * __restrict__ buf, float * __restrict__ sink,
                              size_t stage_floats, int s, int nblocks) {
    const int b = blockIdx.x;
    const size_t per = stage_floats / (size_t) nblocks;
    const float * p = buf + (size_t) s * stage_floats + (size_t) b * per;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < (int) per; i += blockDim.x) acc += p[i];
    if (acc == 1.2345e30f) sink[b] = acc;
}

template<int V>
__global__ void k_small_persist(const float * __restrict__ buf, float * __restrict__ sink,
                                size_t stage_floats, int nstages, int * __restrict__ bar,
                                lk_bp_t * __restrict__ bp, unsigned * __restrict__ leaf,
                                unsigned gen0) {
    const unsigned nb = gridDim.x;
    const size_t per = stage_floats / (size_t) nb;
    const float * base = buf + (size_t) blockIdx.x * per;
    unsigned gen = gen0;
    for (int s = 0; s < nstages; ++s) {
        const float * p = base + (size_t) s * stage_floats;
        float acc = 0.0f;
        for (int i = threadIdx.x; i < (int) per; i += blockDim.x) acc += p[i];
        if (acc == 1.2345e30f) sink[blockIdx.x] = acc;
        ++gen;
        lk_bar_any<V>(bar, bp, leaf, nb, gen);
    }
}

int main(int argc, char ** argv) {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s  SMs=%d  maxThreadsPerSM=%d  L2=%d MB\n\n",
           prop.name, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor,
           (int) (prop.l2CacheSize >> 20));

    cudaStream_t st; CK(cudaStreamCreate(&st));

    int per_sm = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_probe_b<0,0>, 256, 0));
    check_resident(per_sm, per_sm, "k_probe_b");
    const int NB = per_sm * prop.multiProcessorCount;
    printf("k_probe_b occupancy %d blocks/SM -> NB = %d resident blocks "
           "(%d threads/SM, %.0f%% of the 1536 ceiling)\n\n",
           per_sm, NB, per_sm * 256, 100.0 * per_sm * 256 / 1536.0);

    // ---- shapes -----------------------------------------------------------
    // ROWS_BIG is the shape the first verdict used, kept for continuity.  BPR is
    // the input width in Q8_1 blocks: 64 for a 2048-wide hidden.
    const int ROWS_BIG = (argc > 1) ? atoi(argv[1]) : 12288;
    const int ROWS_SML = LK_HIDDEN;              // 2048 rows: half the grid idles
    const int BPR      = LK_HIDDEN / 32;

    const size_t slab_big = (size_t) ROWS_BIG * BPR;
    const size_t slab_sml = (size_t) ROWS_SML * BPR;

    uint4  * h_qs_b = new uint4[slab_big];  __half * h_d_b = new __half[slab_big];
    uint4  * h_qs_s = new uint4[slab_sml];  __half * h_d_s = new __half[slab_sml];
    block_q8_1 * h_xq = new block_q8_1[BPR];

    for (size_t i = 0; i < slab_big; ++i) {
        h_qs_b[i].x = rnd(); h_qs_b[i].y = rnd(); h_qs_b[i].z = rnd(); h_qs_b[i].w = rnd();
        h_d_b[i] = __float2half(0.005f + 0.001f * (rnd() % 7));
    }
    for (size_t i = 0; i < slab_sml; ++i) {
        h_qs_s[i].x = rnd(); h_qs_s[i].y = rnd(); h_qs_s[i].z = rnd(); h_qs_s[i].w = rnd();
        h_d_s[i] = __float2half(0.005f + 0.001f * (rnd() % 7));
    }
    for (int k = 0; k < BPR; ++k) {
        h_xq[k].ds = __floats2half2_rn(0.01f, 0.5f);
        for (int j = 0; j < 32; ++j) h_xq[k].qs[j] = (int8_t) ((int) (rnd() % 255) - 127);
    }

    // Stage count chosen so each shape streams ~450 MB in total, same as the
    // first test, so the numbers are comparable to the published ones.
    int NST_BIG = (int) (450e6 / ((double) slab_big * (sizeof(uint4) + sizeof(__half))));
    if (NST_BIG > 130) NST_BIG = 130; if (NST_BIG < 8) NST_BIG = 8;
    int NST_SML = (int) (450e6 / ((double) slab_sml * (sizeof(uint4) + sizeof(__half))));
    if (NST_SML > 130) NST_SML = 130; if (NST_SML < 8) NST_SML = 8;

    uint4 * d_qs_b; __half * d_d_b; uint4 * d_qs_s; __half * d_d_s;
    block_q8_1 * d_xq; float * d_y; int * d_bar; lk_bp_t * d_bp; unsigned * d_leaf;
    CK(cudaMalloc(&d_qs_b, slab_big * NST_BIG * sizeof(uint4)));
    CK(cudaMalloc(&d_d_b,  slab_big * NST_BIG * sizeof(__half)));
    CK(cudaMalloc(&d_qs_s, slab_sml * NST_SML * sizeof(uint4)));
    CK(cudaMalloc(&d_d_s,  slab_sml * NST_SML * sizeof(__half)));
    CK(cudaMalloc(&d_xq, BPR * sizeof(block_q8_1)));
    CK(cudaMalloc(&d_y, (size_t) ROWS_BIG * sizeof(float)));
    CK(cudaMalloc(&d_bar, 128));
    CK(cudaMalloc(&d_bp, sizeof(lk_bp_t)));
    CK(cudaMalloc(&d_leaf, LK_LEAF_U32 * sizeof(unsigned)));
    // A tree node with zero blocks in its group never fires, so the root never
    // reaches gen*G and the kernel deadlocks.  G <= nb is a hard requirement.
    if (NB < 84) { printf("FATAL: NB=%d < 84, the widest tree node would deadlock\n", NB); return 2; }
    for (int s = 0; s < NST_BIG; ++s) {
        CK(cudaMemcpy(d_qs_b + (size_t) s * slab_big, h_qs_b, slab_big * sizeof(uint4), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_d_b  + (size_t) s * slab_big, h_d_b,  slab_big * sizeof(__half), cudaMemcpyHostToDevice));
    }
    for (int s = 0; s < NST_SML; ++s) {
        CK(cudaMemcpy(d_qs_s + (size_t) s * slab_sml, h_qs_s, slab_sml * sizeof(uint4), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_d_s  + (size_t) s * slab_sml, h_d_s,  slab_sml * sizeof(__half), cudaMemcpyHostToDevice));
    }
    CK(cudaMemcpy(d_xq, h_xq, BPR * sizeof(block_q8_1), cudaMemcpyHostToDevice));

    printf("shape BIG: %d rows x %d     %d stages x %.2f MB = %.0f MB\n",
           ROWS_BIG, BPR, NST_BIG, (double) slab_big * 18 / 1e6,
           (double) slab_big * NST_BIG * 18 / 1e6);
    printf("shape SML: %d rows x %d     %d stages x %.2f MB = %.0f MB\n\n",
           ROWS_SML, BPR, NST_SML, (double) slab_sml * 18 / 1e6,
           (double) slab_sml * NST_SML * 18 / 1e6);

    Ctx c;
    c.qs = d_qs_b; c.d = d_d_b; c.xq = d_xq; c.rows = ROWS_BIG; c.bpr = BPR;
    c.nstages = NST_BIG; c.nb = NB; c.y = d_y; c.bar = d_bar; c.bp = d_bp; c.leaf = d_leaf;
    c.stride = slab_big; c.cad = 1; c.st = st; c.gbase = 4096;   // room for warmups

    // =====================================================================
    // (6) BIT-IDENTITY, first, because a fast wrong barrier is worthless.
    // =====================================================================
    printf("(6) BIT-IDENTITY, every variant, 8 stages, memcmp\n");
    {
        const int NST = 8;
        float * h_ref = new float[ROWS_BIG];
        float * h_prt = new float[ROWS_BIG];
        float * d_ref; float * d_prt;
        CK(cudaMalloc(&d_ref, (size_t) ROWS_BIG * sizeof(float)));
        CK(cudaMalloc(&d_prt, (size_t) ROWS_BIG * sizeof(float)));
        CK(cudaMemset(d_ref, 0, (size_t) ROWS_BIG * sizeof(float)));
        CK(cudaMemset(d_prt, 0, (size_t) ROWS_BIG * sizeof(float)));
        for (int s = 0; s < NST; ++s)
            k_gemv_q4_0<<<LK_GEMV_BLOCKS(ROWS_BIG), 256, 0, st>>>(
                d_qs_b, d_d_b, d_xq, ROWS_BIG, BPR, d_ref, s > 0, nullptr, 0.0f);
        CK(cudaStreamSynchronize(st));
        CK(cudaMemcpy(h_ref, d_ref, (size_t) ROWS_BIG * sizeof(float), cudaMemcpyDeviceToHost));

        int nfail = 0;
        for (int V = 0; V < LK_NBAR_VARIANTS; ++V) {
            CK(cudaMemset(d_prt, 0, (size_t) ROWS_BIG * sizeof(float)));
            CK(cudaMemset(d_bar, 0, 128));
            CK(cudaMemset(d_bp, 0, sizeof(lk_bp_t)));
            CK(cudaMemset(d_leaf, 0, LK_LEAF_U32 * sizeof(unsigned)));
            unsigned gg = 0;
            // Every variant must produce the same bits, including the tree
            // barriers -- if a variant is missing here it is never gated.
            #define BIT_V(VV) k_probe_b<VV,0><<<NB,256,0,st>>>( \
                d_qs_b,d_d_b,d_xq,ROWS_BIG,BPR,d_prt,NST,d_bar,d_bp,d_leaf,slab_big,1,gg)
            DISPATCH(V, BIT_V);
            #undef BIT_V
            CK(cudaStreamSynchronize(st));
            CK(cudaGetLastError());
            CK(cudaMemcpy(h_prt, d_prt, (size_t) ROWS_BIG * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = -1;
            for (int i = 0; i < ROWS_BIG; ++i)
                if (memcmp(&h_ref[i], &h_prt[i], sizeof(float)) != 0) { bad = i; break; }
            if (bad < 0) printf("    V%d %-30s BIT-IDENTICAL\n", V, lk_bar_name(V));
            else { printf("    V%d %-30s *** MISMATCH at row %d: %.9g vs %.9g ***\n",
                          V, lk_bar_name(V), bad, h_ref[bad], h_prt[bad]); ++nfail; }
        }
        printf("    -> %d/%d variants bit-identical\n\n", LK_NBAR_VARIANTS - nfail, LK_NBAR_VARIANTS);
        delete[] h_ref; delete[] h_prt;
        CK(cudaFree(d_ref)); CK(cudaFree(d_prt));
    }

    // =====================================================================
    // (1) THE SHOOTOUT.  Counters reset OUTSIDE the timed region, via a running
    // generation offset -- so the barrier is charged only for itself.
    // =====================================================================
    printf("(1) BARRIER SHOOTOUT, counters reset OUTSIDE the timing\n");
    printf("    shape BIG, %d stages of %.2f MB\n", NST_BIG, (double) slab_big * 18 / 1e6);
    printf("    %-30s %10s %10s\n", "variant", "us/stage", "bar cost");
    for (int i = 0; i < 66; ++i) putchar('-'); putchar('\n');

    c.gbase = 4096;
    const bool RI = false;
    const float t_ref  = time_mode(c, 0, 0, RI, nullptr, nullptr, 0, 256);
    const float t_nb   = time_mode(c, 2, 0, RI, nullptr, nullptr, 0, 256);
    float t_var[LK_NBAR_VARIANTS];
    for (int V = 0; V < LK_NBAR_VARIANTS; ++V)
        t_var[V] = time_mode(c, 1, V, RI, nullptr, nullptr, 0, 256);

    printf("    %-30s %10.3f %10s\n", "separate launches (graph)", t_ref, "-");
    printf("    %-30s %10.3f %10.3f\n", "NO barrier (cost control)", t_nb, "-");
    printf("    %-30s %10.3f %10.3f\n", "LAUNCH BOUNDARY (ref-nobar)", t_ref - t_nb, t_ref - t_nb);
    for (int V = 0; V < LK_NBAR_VARIANTS; ++V)
        printf("    %-30s %10.3f %10.3f   %+.3f vs boundary\n",
               lk_bar_name(V), t_var[V], t_var[V] - t_nb, (t_var[V] - t_nb) - (t_ref - t_nb));
    printf("\n");

    // =====================================================================
    // (2) THE SAME, WITH THE RESET INSIDE THE TIMING -- reproducing the
    // published instrument.  The delta against part (1) is the hidden memset
    // launch, i.e. free speed for the megakernel that the first verdict charged
    // to the barrier.
    // =====================================================================
    printf("(2) SAME SHOOTOUT, counters reset INSIDE the timing (the published\n");
    printf("    instrument).  Delta vs part (1) = the hidden memset launch.\n");
    printf("    %-30s %10s %10s %10s\n", "variant", "reset in", "reset out", "hidden");
    for (int i = 0; i < 70; ++i) putchar('-'); putchar('\n');
    for (int V = 0; V < LK_NBAR_VARIANTS; ++V) {
        c.gbase = 4096;
        const float ti = time_mode(c, 1, V, true, nullptr, nullptr, 0, 256);
        printf("    %-30s %10.3f %10.3f %10.3f\n", lk_bar_name(V), ti, t_var[V], ti - t_var[V]);
    }
    printf("\n");

    // =====================================================================
    // (3) THE BARRIER'S WORST CASE.  2048 rows means 2048 warps of work against
    // 4032-5040 resident warps, so most blocks do nothing and still have to
    // arrive.  This is not a synthetic worry: the engine is full of small stages.
    // =====================================================================
    printf("(3) SHOOTOUT at a shape where MOST BLOCKS IDLE (rows=%d)\n", ROWS_SML);
    {
        Ctx cs = c;
        cs.qs = d_qs_s; cs.d = d_d_s; cs.rows = ROWS_SML; cs.nstages = NST_SML;
        cs.stride = slab_sml; cs.gbase = 4096;
        const float r = time_mode(cs, 0, 0, false, nullptr, nullptr, 0, 256);
        const float n = time_mode(cs, 2, 0, false, nullptr, nullptr, 0, 256);
        printf("    %-30s %10.3f\n", "separate launches (graph)", r);
        printf("    %-30s %10.3f\n", "NO barrier (cost control)", n);
        printf("    %-30s %10.3f %10.3f\n", "LAUNCH BOUNDARY", r - n, r - n);
        for (int V = 0; V < LK_NBAR_VARIANTS; ++V) {
            cs.gbase = 4096;
            const float t = time_mode(cs, 1, V, false, nullptr, nullptr, 0, 256);
            printf("    %-30s %10.3f %10.3f\n", lk_bar_name(V), t, t - n);
        }
    }
    printf("\n");

    // =====================================================================
    // (4) CADENCE.  The megakernel does not need a barrier between stages that
    // have no cross-block dependency.  If the cost is per-barrier, cadence sets
    // the budget; if cadence does not help, the barrier is not the problem.
    // =====================================================================
    printf("(4) CADENCE SWEEP, flat (V%d) vs tree (V11), shape BIG\n", LK_BAR_BEST);
    printf("    MEASURED TWICE, round-robin.  A cadence sweep that is only measured\n");
    printf("    once cannot tell a cadence effect from drift: the first version of\n");
    printf("    this table read 23.9 us at cad=1 where parts (1) and the published\n");
    printf("    test both read 18.5, and there was no way to see which was wrong.\n");
    printf("    `per node` is the column that decides it: it is the barrier's cost\n");
    printf("    divided over the stages it covers, and it must beat the boundary.\n");
    printf("    %4s | %9s %9s %9s | %9s %9s %9s\n",
           "cad", "flat r1", "flat r2", "per nod", "tree r1", "tree r2", "per nod");
    for (int i = 0; i < 76; ++i) putchar('-'); putchar('\n');
    {
        const int cads[] = {1, 2, 4, 8, 16, 32};
        const int vars[] = {LK_BAR_BEST, 11};
        float t[2][6][2];
        // Round-robin over BOTH variants and BOTH rounds, so a drift in the
        // machine shows up as an inconsistent delta rather than as a "result".
        for (int round = 0; round < 2; ++round)
            for (int vi = 0; vi < 2; ++vi)
                for (int i = 0; i < 6; ++i) {
                    c.cad = cads[i];
                    t[vi][i][round] =
                        time_mode(c, 1, vars[vi], false, nullptr, nullptr, 0, 256);
                }
        for (int i = 0; i < 6; ++i) {
            // t is us/STAGE with nbar_of() barriers per c.nstages stages, so the
            // per-NODE cost is (t - t_nb) * nstages / nbar_of.
            const int nbn = nbar_of(c.nstages, cads[i]);
            double per[2];
            for (int vi = 0; vi < 2; ++vi)
                per[vi] = (t[vi][i][1] - t_nb) * c.nstages / (nbn > 0 ? nbn : 1);
            printf("    %4d | %9.3f %9.3f %9.3f | %9.3f %9.3f %9.3f\n",
                   cads[i], t[0][i][0], t[0][i][1], per[0],
                   t[1][i][0], t[1][i][1], per[1]);
        }
        c.cad = 1;
    }
    printf("    boundary for reference: %.3f us PER NODE\n\n", t_ref - t_nb);

    // =====================================================================
    // (5) THE REGIME THAT MATTERS.  Small stages: one kernel launch each in the
    // graph, one barrier each in a persistent kernel.  This is what 158 of the
    // engine's 343 nodes look like, and the first verdict never looked here.
    // =====================================================================
    printf("(5) SMALL-STAGE SWEEP.  graph of N launches vs 1 persistent + N barriers\n");
    printf("    %9s %7s %8s %10s %10s %9s\n",
           "stage", "KB", "NST", "graph us/s", "persist us/s", "ratio");
    for (int i = 0; i < 62; ++i) putchar('-'); putchar('\n');
    {
        const size_t KBs[] = {8, 32, 128, 512, 2048, 8192};
        for (int i = 0; i < 6; ++i) {
            const size_t bytes = KBs[i] * 1024;
            const size_t sf = bytes / sizeof(float);
            const int NST = 400;
            // The persistent layout lets block b's LAST read land at
            // b*per + (NST-1)*sf + per, so the buffer must cover (NST+2)*sf --
            // not NST*sf.  Getting this wrong reads off the end silently.
            const size_t total = sf * (NST + 2);
            float * dbuf; float * dsink;
            CK(cudaMalloc(&dbuf, total * sizeof(float)));
            CK(cudaMemset(dbuf, 0, total * sizeof(float)));
            CK(cudaMalloc(&dsink, NB * sizeof(float)));

            // --- graph mode
            cudaGraph_t g; cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            for (int s = 0; s < NST; ++s)
                k_small_stage<<<NB, 256, 0, st>>>(dbuf, dsink, sf, s, NB);
            CK(cudaStreamEndCapture(st, &g));
            CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
            for (int w = 0; w < 2; ++w) CK(cudaGraphLaunch(ge, st));
            CK(cudaStreamSynchronize(st));
            cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
            CK(cudaEventRecord(a, st));
            for (int r = 0; r < REPS; ++r) CK(cudaGraphLaunch(ge, st));
            CK(cudaEventRecord(b, st));
            CK(cudaStreamSynchronize(st));
            float ms; CK(cudaEventElapsedTime(&ms, a, b));
            const double tg = ms * 1000.0 / REPS / NST;

            // --- persistent mode.  reset_bar FIRST: the counter must line up
            //     with generation 1 or the barrier deadlocks.
            Ctx cs = c; cs.nb = NB; cs.bar = d_bar; cs.bp = d_bp;
            reset_bar(cs);
            unsigned gg = 0;
            for (int w = 0; w < 2; ++w) {
                k_small_persist<LK_BAR_BEST><<<NB,256,0,st>>>(dbuf, dsink, sf, NST, d_bar, d_bp, d_leaf, gg);
                gg += NST;
            }
            CK(cudaStreamSynchronize(st));
            CK(cudaEventRecord(a, st));
            for (int r = 0; r < REPS; ++r) {
                k_small_persist<LK_BAR_BEST><<<NB,256,0,st>>>(dbuf, dsink, sf, NST, d_bar, d_bp, d_leaf, gg);
                gg += NST;
            }
            CK(cudaEventRecord(b, st));
            CK(cudaStreamSynchronize(st));
            CK(cudaGetLastError());
            CK(cudaEventElapsedTime(&ms, a, b));
            const double tp = ms * 1000.0 / REPS / NST;

            printf("    %8zuK %7.1f %8d %10.3f %10.3f %8.3fx\n",
                   KBs[i], (double) bytes / 1024,
                   NST, tg, tp, tg / tp);
            CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
            CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
            CK(cudaFree(dbuf)); CK(cudaFree(dsink));
        }
    }

    printf("\nVERDICT LOGIC\n");
    printf("  part 1 says what a handoff really costs; part 2 says how much of the\n");
    printf("  published 2.17 us was a memset; part 4 says whether fewer barriers are\n");
    printf("  cheaper; part 5 says whether there is ANY stage size where a persistent\n");
    printf("  kernel beats a graph.  If part 5 shows a crossover, the megakernel is\n");
    printf("  not refuted -- it is simply not applicable at batch 1's large stages.\n");

    delete[] h_qs_b; delete[] h_d_b; delete[] h_qs_s; delete[] h_d_s; delete[] h_xq;
    CK(cudaFree(d_qs_b)); CK(cudaFree(d_d_b)); CK(cudaFree(d_qs_s)); CK(cudaFree(d_d_s));
    CK(cudaFree(d_xq)); CK(cudaFree(d_y)); CK(cudaFree(d_bar)); CK(cudaFree(d_bp));
    CK(cudaFree(d_leaf));
    CK(cudaStreamDestroy(st));
    return 0;
}
