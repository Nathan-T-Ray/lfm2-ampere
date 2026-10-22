// bench_runahead.cu - the run-ahead kill-test (REVIEW-2026-10-21.md, section 1).
//
// THE QUESTION.  ARCHITECTURE.md 13.14 measured 14 software grid barriers at
// ~2.5 us per stage against a 1.18 us launch boundary and closed the
// megakernel.  All 14 have the same shape: finish stage s, rendezvous, and only
// THEN issue the loads for stage s+1.  So DRAM drains during every rendezvous
// and has to refill after it, and the barrier is charged for both.
//
// But a stage's WEIGHTS do not depend on the stage before it; only its
// activations do.  If each warp issues its next-stage weight loads before the
// rendezvous completes, the rendezvous overlaps the transfer, and the barrier
// should cost only what the transfer cannot hide.
//
// PREDICTION, written before the first run: at the published 12288-row shape,
// a barrier with run-ahead costs <= 0.5 us per stage over the no-barrier floor.
// FALSIFIED if it does not beat the launch boundary.
//
// RUN 1 (bench/runahead_r1_2026-10-21.txt) issued the run-ahead BEFORE the
// arrival and it did not help: +1.87 us against +1.96 published at 12288 rows,
// and smem run-ahead was WORSE (+2.83).  At 2048 rows the whole next stage was
// in flight during the barrier and the stage time did not move -- which says
// the arrival itself waited for the run-ahead.  The arrival begins with
// __threadfence(), and a fence waits for the issuing thread's outstanding
// memory operations.  Run 2 therefore adds:
//   * a SPLIT barrier: fence + arrive, THEN issue the run-ahead, then wait on
//     the release with an acquire load (no trailing fence).  Nothing the fence
//     could wait for exists when it executes.
//   * a fence-free barrier, DIAGNOSTIC ONLY, to price the fence directly.  It is
//     valid here only because each stage's output is read back by the warp that
//     wrote it; a real cross-block handoff needs the fence.
// Run 1's "xq via L2" variants loaded every activation word with ld.global.cg
// from every lane, which hot-spots 2.3 KB on a few L2 slices (50x slower).  That
// was an instrument bug.  The honest variant now stages xq into shared memory
// once per block per stage with 144 coalesced ld.global.cg -- what a real
// persistent kernel must do, since L1 is not coherent across blocks.
//
// RESULT (runs 2 and 3, bench/runahead_r3_2026-10-21.txt).  FALSIFIED.  The
// best correct run-ahead variant (split barrier, smem run-ahead, xq staged) is
// +1.74 / +2.16 / +2.08 us per stage over the floor in shapes B / C / D,
// against +0.83 / +1.51 / +1.07 for a CUDA graph.  Even the fence-free
// diagnostic barrier only matches a launch boundary, and at 2048 rows a full
// next stage in flight during it did not shorten the stage (4.112 vs 4.116 us):
// the barrier's atomics and polls queue behind the bulk traffic.
// What DOES work is the graph-side tail prefetch: the last warps of each GEMV,
// after their own row, prefetch ~1 MB (768-1024 rows) of the next big GEMV.
// -0.85 us/stage at 12288 rows (the boundary disappears), -0.63 at 2048 rows,
// -4.0% per engine-like layer.  Issued before the row, or larger, it loses.
//
// THE INSTRUMENT is bench_mkbar's, with one factor moved: the same Q4_0 x Q8_1
// stage, the same 504 resident blocks of 256 threads, counters reset outside
// the timing, and every variant gated by memcmp against the shipped
// k_gemv_q4_0 before it is timed.
//
// Shapes:
//   (B) 12288 rows x 31 stages     the published shape
//   (C)  2048 rows x 130 stages    most warps idle, barrier-dominated
//   (D) engine-like layer          9 stages per layer, 12 layers:
//       6144 / 64 / 2048 / 64 / 64 / 64 / 12288 / 64 / 8192 rows
//       ~ in_proj, conv, out_proj, norm, router, top4, gate_up, swiglu, down:
//       33.4 MB per layer against the engine's ~36.7 MB, 9 nodes against ~8.6.
//       The 64-row stages are 8-block stand-ins for the small nodes: run-ahead
//       has to carry the next big stage's weights across them.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "megakernel.cuh"
#include "mkbar.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(3); } } while (0)

#define RA_BPR      64                                // LK_HIDDEN / 32 Q8_1 blocks per row
#define RA_SLOT_U4  (RA_BPR + RA_BPR * 2 / 16)        // 64 uint4 of qs + 8 uint4 (128 B) of d
#define RA_XQ_U4    (RA_BPR * 36 / 16)                // 144 uint4 = one row's activations
#define RA_THREADS  256
#define RA_WARPS    (RA_THREADS / 32)
#define RA_SMEM     (RA_WARPS * RA_SLOT_U4 * 16)      // 9216 B per block
#define RA_PER_SM   6                                 // the published 504-block grid
#define RA_BIG_ROWS 1024                              // a stage this large gets run-ahead
#define RA_MAX_ST   512
#define REPS        5
static_assert(RA_BPR == LK_HIDDEN / 32, "row width");
static_assert(RA_XQ_U4 <= RA_THREADS, "one uint4 per thread stages xq");

// Barrier selectors.  0..13 are mkbar.cuh's.
#define BAR_NONE    (-1)
#define BAR_NOFENCE 100      // diagnostic only, see header
#define BAR_SPLIT   101      // fence + arrive / run-ahead / acquire-wait

static uint32_t g_seed = 12345u;
static uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }

// ---------------------------------------------------------------------------
// The stage program.  One entry per stage, read with a warp-uniform index.
// nxt = the next stage with rows >= RA_BIG_ROWS, or -1: the run-ahead target.
// ---------------------------------------------------------------------------
struct Stage {
    const uint4 * qs; const __half * d;
    int rows; int big; int nxt; int pad;
};
__constant__ Stage c_prog[RA_MAX_ST];

struct ProgArgs {
    const block_q8_1 * xq; float * y; int nst; int first_big;
    int * bar; lk_bp_t * bp; unsigned * leaf; unsigned gen0;
    int pf_rows; int pf_q;
};

// ---- PTX ------------------------------------------------------------------
__device__ __forceinline__ void ra_cp16(void * smem, const void * gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void ra_commit()   { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void ra_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }
__device__ __forceinline__ void ra_pf_l2(const void * p) {
    asm volatile("prefetch.global.L2::evict_last [%0];\n" :: "l"(p));
}

// ---------------------------------------------------------------------------
// Barriers added by this test.
// ---------------------------------------------------------------------------
// DIAGNOSTIC ONLY: V2 with both fences removed.
__device__ __forceinline__ void ra_bar_nofence(lk_bp_t * bp, unsigned nb, unsigned gen) {
    __syncthreads();
    if (threadIdx.x == 0) {
        if (atomicAdd(&bp->arrived, 1u) + 1u == gen * nb)
            *(volatile unsigned *) &bp->released = gen;
        while (*(volatile unsigned *) &bp->released < gen) __nanosleep(64);
    }
    __syncthreads();
}

// SPLIT, phase 1: the stage's stores are fenced and the block is counted.  The
// trailing __syncthreads is what lets the other warps issue run-ahead safely:
// nothing is outstanding when thread 0's fence executes.
__device__ __forceinline__ void ra_arrive(lk_bp_t * bp, unsigned nb, unsigned gen) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(&bp->arrived, 1u) + 1u == gen * nb) {
            __threadfence();
            *(volatile unsigned *) &bp->released = gen;
        }
    }
    __syncthreads();
}
// SPLIT, phase 2: acquire-load spin, no trailing fence.  bar.sync carries the
// acquire to the rest of the block (PTX causality order is transitive).
__device__ __forceinline__ void ra_wait(lk_bp_t * bp, unsigned gen) {
    if (threadIdx.x == 0)
        while (lk_ld_acq(&bp->released) < gen) __nanosleep(64);
    __syncthreads();
}

template<int BARV>
__device__ __forceinline__ void ra_bar(const ProgArgs & a, unsigned nb, unsigned & gen) {
    if constexpr (BARV == BAR_NOFENCE) { ++gen; ra_bar_nofence(a.bp, nb, gen); }
    else if constexpr (BARV >= 0 && BARV < LK_NBAR_VARIANTS) { ++gen; lk_bar_any<BARV>(a.bar, a.bp, a.leaf, nb, gen); }
}

// ---------------------------------------------------------------------------
// The dot.  Transcribed from dot_q4_0_q8_1 (quant.cuh) with the weight load
// hoisted out, so the same arithmetic can run on a row that came from smem.
// `x` may point at global or shared memory.  Bit-identity to the shipped
// kernel is not argued; the memcmp gate checks it.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float ra_dot(const uint4 q, const float d4, const block_q8_1 * x) {
    const int32_t * xq = reinterpret_cast<const int32_t *>(x->qs);
    const uint32_t * qw = reinterpret_cast<const uint32_t *>(&q);
    int sumi = 0;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const uint32_t v = qw[w];
        sumi = dp4a((int) (v & 0x0F0F0F0Fu),        xq[w],     sumi);
        sumi = dp4a((int) ((v >> 4) & 0x0F0F0F0Fu), xq[4 + w], sumi);
    }
    const float d8  = __low2float(x->ds);
    const float sum = __high2float(x->ds);
    return d4 * (d8 * (float) sumi - 8.0f * sum);
}

__device__ __forceinline__ float ra_row_global(const uint4 * __restrict__ qs,
                                               const __half * __restrict__ d,
                                               const block_q8_1 * xq, int row, int lane) {
    float acc = 0.0f;
    for (int k = lane; k < RA_BPR; k += 32) {
        const size_t i = (size_t) row * RA_BPR + k;
        acc += ra_dot(__ldcs(qs + i), __half2float(d[i]), xq + k);
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float ra_row_smem(const uint4 * sq, const block_q8_1 * xq, int lane) {
    const __half * sd = reinterpret_cast<const __half *>(sq + RA_BPR);
    float acc = 0.0f;
    for (int k = lane; k < RA_BPR; k += 32)
        acc += ra_dot(sq[k], __half2float(sd[k]), xq + k);
    return warp_sum(acc);
}

// One row (1024 B of qs + 128 B of d) into this warp's slot: 72 x 16 B copies,
// each instruction covering 512 contiguous bytes across the warp.
__device__ __forceinline__ void ra_issue_row(uint4 * sq, const uint4 * qs, const __half * d,
                                             int row, int lane) {
    const uint4 * gq = qs + (size_t) row * RA_BPR;
    ra_cp16(sq + lane,      gq + lane);
    ra_cp16(sq + lane + 32, gq + lane + 32);
    if (lane < 8)
        ra_cp16(sq + RA_BPR + lane, reinterpret_cast<const uint4 *>(d + (size_t) row * RA_BPR) + lane);
}

// L2 prefetch of up to npf of this warp's rows, the first 32*pfq bytes of qs
// and all 128 B of d.  One 32 B sector per lane per instruction.
__device__ __forceinline__ void ra_pf_rows(const uint4 * qs, const __half * d, int rows,
                                           int w0, int tw, int npf, int pfq, int lane) {
    for (int j = 0; j < npf; ++j) {
        const int r = w0 + j * tw;
        if (r >= rows) break;
        if (lane < pfq) ra_pf_l2(reinterpret_cast<const char *>(qs + (size_t) r * RA_BPR) + 32 * lane);
        if (lane < 4)   ra_pf_l2(reinterpret_cast<const char *>(d + (size_t) r * RA_BPR) + 32 * lane);
    }
}

// ---------------------------------------------------------------------------
// THE PERSISTENT KERNEL.
//   MODE 0  wait-then-load (the published shape)
//   MODE 1  L2 run-ahead, issued before the arrival
//   MODE 2  smem run-ahead, issued before the arrival
//   MODE 4  SPLIT barrier, smem run-ahead issued between arrive and wait
//   MODE 5  SPLIT barrier, L2 run-ahead issued between arrive and wait
//   MODE 6  SPLIT barrier, no run-ahead (prices the split barrier itself)
// XS=1 stages xq into shared memory once per block per stage via ld.global.cg.
// Row ownership is bench_mkbar's grid-stride map, so a warp owns the same rows
// in every stage and the per-row arithmetic is the shipped kernel's.
// ---------------------------------------------------------------------------
template<int MODE, int BARV, int XS>
__global__ void __launch_bounds__(RA_THREADS, RA_PER_SM) k_prog(const ProgArgs a) {
    constexpr bool SLOT  = (MODE == 2 || MODE == 4);
    constexpr bool L2PF  = (MODE == 1 || MODE == 5);
    constexpr bool SPLIT = (MODE >= 4);
    static_assert(!SPLIT || BARV == BAR_SPLIT, "split modes take BAR_SPLIT");
    extern __shared__ uint4 ra_smem[];
    __shared__ uint4 s_xq4[XS ? RA_XQ_U4 : 1];

    const int lane = threadIdx.x & 31;
    const int tw   = (int) ((gridDim.x * blockDim.x) >> 5);
    const int w0   = (int) ((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    uint4 * sq = ra_smem + (threadIdx.x >> 5) * RA_SLOT_U4;
    const unsigned nb = gridDim.x;
    unsigned gen = a.gen0;
    (void) nb; (void) s_xq4;

    int held = -1;                                    // stage whose first row is in the slot
    if (SLOT && a.first_big >= 0 && a.first_big < a.nst && w0 < c_prog[a.first_big].rows) {
        ra_issue_row(sq, c_prog[a.first_big].qs, c_prog[a.first_big].d, w0, lane);
        ra_commit();
        held = a.first_big;
    }
    for (int s = 0; s < a.nst; ++s) {
        const uint4 *  qs   = c_prog[s].qs;
        const __half * d    = c_prog[s].d;
        const int      rows = c_prog[s].rows;
        const int      nxt  = (c_prog[s].nxt < a.nst) ? c_prog[s].nxt : -1;

        const block_q8_1 * xq = a.xq;
        if constexpr (XS) {
            // Only blocks with rows in this stage stage xq (block-uniform, so the
            // __syncthreads stay convergent).  Run 2 staged it in all 504 blocks,
            // which charged idle blocks for an L2 hot spot the design need not have.
            if ((int) (blockIdx.x * RA_WARPS) < rows) {
                if constexpr (BARV == BAR_NONE) __syncthreads();   // no barrier guards s_xq reuse
                if (threadIdx.x < RA_XQ_U4)
                    s_xq4[threadIdx.x] = __ldcg(reinterpret_cast<const uint4 *>(a.xq) + threadIdx.x);
                __syncthreads();
            }
            xq = reinterpret_cast<const block_q8_1 *>(s_xq4);
        }

        int row = w0;
        if (SLOT && held == s) {
            ra_wait_all();
            __syncwarp();                             // lanes read each other's d copies
            const float v = ra_row_smem(sq, xq, lane);
            if (lane == 0) a.y[row] = (s > 0) ? (a.y[row] + v) : v;
            held = -1;
            row += tw;
        }
        for (; row < rows; row += tw) {
            const float v = ra_row_global(qs, d, xq, row, lane);
            if (lane == 0) a.y[row] = (s > 0) ? (a.y[row] + v) : v;
        }

        if constexpr (SPLIT) {
            ++gen;
            ra_arrive(a.bp, nb, gen);
        }
        if (SLOT && held < 0 && nxt >= 0 && w0 < c_prog[nxt].rows) {
            __syncwarp();                             // the slot was read above
            ra_issue_row(sq, c_prog[nxt].qs, c_prog[nxt].d, w0, lane);
            ra_commit();
            held = nxt;
        }
        if (L2PF && c_prog[s].big && nxt >= 0)
            ra_pf_rows(c_prog[nxt].qs, c_prog[nxt].d, c_prog[nxt].rows, w0, tw,
                       a.pf_rows, a.pf_q, lane);
        if constexpr (SPLIT) ra_wait(a.bp, gen);
        else                 ra_bar<BARV>(a, nb, gen);
    }
    if (SLOT) ra_wait_all();
}

// ---------------------------------------------------------------------------
// Graph-compatible run-ahead: the shipped one-row-per-warp GEMV, plus an L2
// prefetch of the NEXT big GEMV's first pf_count rows issued by THIS GEMV's
// last pf_count warps -- the last-dispatched ones.  pf_after=1 issues it after
// the warp's own row (its loads are done); 0 issues it before (competing with
// them).  qn == nullptr: plain GEMV.
// ---------------------------------------------------------------------------
// rpw = rows prefetched per issuing warp, so the issuers are the last
// ceil(pf_count / rpw) warps: fewer, later issuers for the same byte count.
__global__ void __launch_bounds__(RA_THREADS) k_gemv_tailpf(
        const uint4 * __restrict__ qs, const __half * __restrict__ d,
        const block_q8_1 * __restrict__ xq, int rows, float * __restrict__ y, int add_residual,
        const uint4 * __restrict__ qn, const __half * __restrict__ dn, int pf_count, int pf_after,
        int rpw) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const int nis = (pf_count + rpw - 1) / rpw;        // issuing warps
    const int i   = warp - (rows - nis);
    const bool pf = (qn != nullptr) && (i >= 0);
    // rows i*rpw .. i*rpw+rpw-1 of the next GEMV, clipped to pf_count
    if (pf && !pf_after) ra_pf_rows(qn, dn, pf_count, i * rpw, 1, rpw, 32, lane);
    const float v = ra_row_global(qs, d, xq, warp, lane);
    if (lane == 0) y[warp] = add_residual ? (y[warp] + v) : v;
    if (pf && pf_after) ra_pf_rows(qn, dn, pf_count, i * rpw, 1, rpw, 32, lane);
}

// ===========================================================================
// HOST
// ===========================================================================
typedef void (*ProgKern)(const ProgArgs);
enum { KIND_REF = 0, KIND_OWN = 1, KIND_PERSIST = 2 };
enum { TAG_NONE = 0, TAG_FLOOR, TAG_GRAPH, TAG_PUB, TAG_RA, TAG_RA_HONEST, TAG_TAIL };

struct Cfg {
    const char * name;
    int kind, graph;
    ProgKern pk; int smem; int barv; int xs;
    int pf_rows, pf_q;            // persistent L2 run-ahead
    int pf_count, pf_after;       // graph tail prefetch
    int tag;
};

#define PERS(name, M, B, X, smem, pfr, pfq, tag) \
    {name, KIND_PERSIST, 0, k_prog<M, B, X>, smem, B, X, pfr, pfq, 0, 0, tag}
// pf_rows doubles as rows-per-issuing-warp for the tail configs
#define TAIL(name, cnt, after, rpw, tag) \
    {name, KIND_OWN, 1, nullptr, 0, BAR_NONE, 0, rpw, 0, cnt, after, tag}

static Cfg g_cfgs[] = {
  {"launches, stream (published ref)", KIND_REF, 0, nullptr, 0, BAR_NONE, 0, 0, 0, 0, 0, TAG_NONE},
  {"launches, CUDA graph",             KIND_REF, 1, nullptr, 0, BAR_NONE, 0, 0, 0, 0, 0, TAG_GRAPH},
  TAIL("graph, own GEMV, no prefetch",          0, 0, 1, TAG_NONE),
  TAIL("graph + tail pf 2048, early",        2048, 0, 1, TAG_NONE),
  TAIL("graph + tail pf  256 late",           256, 1, 1, TAG_TAIL),
  TAIL("graph + tail pf  512 late",           512, 1, 1, TAG_TAIL),
  TAIL("graph + tail pf  768 late",           768, 1, 1, TAG_TAIL),
  TAIL("graph + tail pf 1024 late",          1024, 1, 1, TAG_TAIL),
  TAIL("graph + tail pf 1536 late",          1536, 1, 1, TAG_TAIL),
  TAIL("graph + tail pf 2048 late",          2048, 1, 1, TAG_TAIL),
  TAIL("graph + tail pf 4096 late",          4096, 1, 1, TAG_TAIL),
  TAIL("graph + tail pf 1024 late, 512 x2",  1024, 1, 2, TAG_TAIL),
  TAIL("graph + tail pf 1024 late, 256 x4",  1024, 1, 4, TAG_TAIL),
  TAIL("graph + tail pf 2048 late, 512 x4",  2048, 1, 4, TAG_TAIL),
  PERS("persistent, NO barrier (floor)",     0, BAR_NONE,    0, 0,       0, 0,  TAG_FLOOR),
  PERS("V2 wait-then-load (published)",      0, 2,           0, 0,       0, 0,  TAG_PUB),
  PERS("V2 + L2 512 B, before arrive",       1, 2,           0, 0,       1, 16, TAG_NONE),
  PERS("V2 + smem, before arrive",           2, 2,           0, RA_SMEM, 0, 0,  TAG_NONE),
  PERS("no-fence bar (diag), wait-then-load",0, BAR_NOFENCE, 0, 0,       0, 0,  TAG_NONE),
  PERS("no-fence bar (diag) + smem",         2, BAR_NOFENCE, 0, RA_SMEM, 0, 0,  TAG_NONE),
  PERS("SPLIT bar, no run-ahead",            6, BAR_SPLIT,   0, 0,       0, 0,  TAG_NONE),
  PERS("SPLIT bar + smem run-ahead",         4, BAR_SPLIT,   0, RA_SMEM, 0, 0,  TAG_RA),
  PERS("SPLIT bar + L2 512 B run-ahead",     5, BAR_SPLIT,   0, 0,       1, 16, TAG_RA),
  PERS("SPLIT bar + L2 1 KB run-ahead",      5, BAR_SPLIT,   0, 0,       1, 32, TAG_RA),
  PERS("xq staged: NO barrier",              0, BAR_NONE,    1, 0,       0, 0,  TAG_NONE),
  PERS("xq staged: V2 wait-then-load",       0, 2,           1, 0,       0, 0,  TAG_NONE),
  PERS("xq staged: SPLIT, no run-ahead",     6, BAR_SPLIT,   1, 0,       0, 0,  TAG_NONE),
  PERS("xq staged: SPLIT + smem run-ahead",  4, BAR_SPLIT,   1, RA_SMEM, 0, 0,  TAG_RA_HONEST),
  PERS("xq staged: SPLIT + L2 512 B",        5, BAR_SPLIT,   1, 0,       1, 16, TAG_RA_HONEST),
};
static const int NCFG = (int) (sizeof(g_cfgs) / sizeof(g_cfgs[0]));

struct Shape {
    const char * name;
    std::vector<Stage> st;
    int first_big = -1;
    int nchk = 0;                 // stages run by the bit-identity gate
    int layer = 0;                // stages per layer, for the per-layer column (0 = chain)
    double bytes = 0;             // weight bytes per pass
};

static void link_shape(Shape & sh) {
    const int n = (int) sh.st.size();
    if (n > RA_MAX_ST) { printf("FATAL: %d stages > %d\n", n, RA_MAX_ST); exit(2); }
    int next = -1;
    for (int s = n - 1; s >= 0; --s) {
        sh.st[s].big = sh.st[s].rows >= RA_BIG_ROWS;
        sh.st[s].nxt = next;
        if (sh.st[s].big) next = s;
    }
    sh.first_big = next;
    sh.bytes = 0;
    for (const Stage & s : sh.st) sh.bytes += (double) s.rows * RA_BPR * 18.0;
}

struct Dev {
    const block_q8_1 * xq;
    int * bar; lk_bp_t * bp; unsigned * leaf;
    int nb;
    cudaStream_t st;
};

static void reset_bar(const Dev & dv) {
    CK(cudaMemsetAsync(dv.bar, 0, 128, dv.st));
    CK(cudaMemsetAsync(dv.bp, 0, sizeof(lk_bp_t), dv.st));
    CK(cudaMemsetAsync(dv.leaf, 0, LK_LEAF_U32 * sizeof(unsigned), dv.st));
}

// Run `nst` stages of the shape once.  Launch kinds go straight to the stream
// (the caller captures them into a graph when it wants one).
static void run_once(const Shape & sh, const Cfg & c, int nst, float * y, unsigned & gbase,
                     const Dev & dv) {
    if (c.kind == KIND_REF) {
        for (int s = 0; s < nst; ++s) {
            const Stage & t = sh.st[s];
            k_gemv_q4_0<<<LK_GEMV_BLOCKS(t.rows), 256, 0, dv.st>>>(
                t.qs, t.d, dv.xq, t.rows, RA_BPR, y, s > 0, nullptr, 0.0f);
        }
    } else if (c.kind == KIND_OWN) {
        for (int s = 0; s < nst; ++s) {
            const Stage & t = sh.st[s];
            const uint4 * qn = nullptr; const __half * dn = nullptr; int pc = 0;
            if (c.pf_count > 0 && t.big && t.nxt >= 0 && t.nxt < nst) {
                const Stage & u = sh.st[t.nxt];
                qn = u.qs; dn = u.d;
                pc = std::min(c.pf_count, std::min(t.rows, u.rows));
            }
            k_gemv_tailpf<<<LK_GEMV_BLOCKS(t.rows), RA_THREADS, 0, dv.st>>>(
                t.qs, t.d, dv.xq, t.rows, y, s > 0, qn, dn, pc, c.pf_after,
                c.pf_rows > 0 ? c.pf_rows : 1);
        }
    } else {
        ProgArgs a;
        a.xq = dv.xq; a.y = y; a.nst = nst; a.first_big = sh.first_big;
        a.bar = dv.bar; a.bp = dv.bp; a.leaf = dv.leaf; a.gen0 = gbase;
        a.pf_rows = c.pf_rows; a.pf_q = c.pf_q;
        c.pk<<<dv.nb, RA_THREADS, c.smem, dv.st>>>(a);
        if (c.barv != BAR_NONE) gbase += (unsigned) nst;
    }
}

static float time_cfg(const Shape & sh, const Cfg & c, cudaGraphExec_t ge, float * y,
                      const Dev & dv) {
    const int nst = (int) sh.st.size();
    unsigned g = 0;
    if (c.kind == KIND_PERSIST) reset_bar(dv);       // outside the timing, as in bench_mkbar
    auto pass = [&]() {
        if (ge) CK(cudaGraphLaunch(ge, dv.st));
        else    run_once(sh, c, nst, y, g, dv);
    };
    for (int w = 0; w < 2; ++w) pass();
    CK(cudaStreamSynchronize(dv.st));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, dv.st));
    for (int r = 0; r < REPS; ++r) pass();
    CK(cudaEventRecord(e1, dv.st));
    CK(cudaStreamSynchronize(dv.st));
    CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    return ms * 1000.0f / REPS / nst;                // us per stage (node)
}

static float median_of(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n & 1) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

struct ShapeResult { float floor_, graph, pub, best_ra, best_honest, best_tail; int mism; };

static ShapeResult run_shape(Shape & sh, const Dev & dv, float * d_y, float * d_yref,
                             float * h_y, float * h_ref, int rounds) {
    ShapeResult R = {0, 0, 0, 1e30f, 1e30f, 1e30f, 0};
    link_shape(sh);
    const int nst = (int) sh.st.size();
    CK(cudaMemcpyToSymbol(c_prog, sh.st.data(), nst * sizeof(Stage)));
    const size_t ybytes = (size_t) 12288 * sizeof(float);

    printf("=====================================================================\n");
    printf("%s\n", sh.name);
    printf("    %d stages, %.1f MB per pass, %.2f MB per stage on average\n",
           nst, sh.bytes / 1e6, sh.bytes / 1e6 / nst);

    // ---- bit-identity gate, first: a fast wrong kernel is worthless ------
    {
        unsigned g = 0;
        CK(cudaMemset(d_yref, 0, ybytes));
        run_once(sh, g_cfgs[0], sh.nchk, d_yref, g, dv);
        CK(cudaStreamSynchronize(dv.st));
        CK(cudaMemcpy(h_ref, d_yref, ybytes, cudaMemcpyDeviceToHost));
        int nok = 0;
        for (int i = 0; i < NCFG; ++i) {
            const Cfg & c = g_cfgs[i];
            CK(cudaMemset(d_y, 0, ybytes));
            reset_bar(dv);
            unsigned gg = 0;
            run_once(sh, c, sh.nchk, d_y, gg, dv);
            CK(cudaStreamSynchronize(dv.st));
            CK(cudaGetLastError());
            CK(cudaMemcpy(h_y, d_y, ybytes, cudaMemcpyDeviceToHost));
            int bad = -1;
            for (int r = 0; r < 12288; ++r)
                if (memcmp(&h_ref[r], &h_y[r], sizeof(float)) != 0) { bad = r; break; }
            if (bad < 0) ++nok;
            else {
                printf("    *** %s: MISMATCH at row %d: %.9g vs %.9g\n",
                       c.name, bad, h_ref[bad], h_y[bad]);
                ++R.mism;
            }
        }
        printf("    bit-identity, %d stages, memcmp vs k_gemv_q4_0: %d/%d configs identical\n\n",
               sh.nchk, nok, NCFG);
    }

    // ---- graphs for the launch configs ------------------------------------
    std::vector<cudaGraphExec_t> ge(NCFG, nullptr);
    std::vector<cudaGraph_t> gr(NCFG, nullptr);
    for (int i = 0; i < NCFG; ++i) {
        if (!g_cfgs[i].graph) continue;
        unsigned g = 0;
        CK(cudaStreamBeginCapture(dv.st, cudaStreamCaptureModeThreadLocal));
        run_once(sh, g_cfgs[i], nst, d_y, g, dv);
        CK(cudaStreamEndCapture(dv.st, &gr[i]));
        CK(cudaGraphInstantiate(&ge[i], gr[i], nullptr, nullptr, 0));
    }

    // ---- round-robin timing ------------------------------------------------
    std::vector<std::vector<float>> t(NCFG);
    for (int r = 0; r < rounds; ++r)
        for (int i = 0; i < NCFG; ++i)
            t[i].push_back(time_cfg(sh, g_cfgs[i], ge[i], d_y, dv));

    std::vector<float> med(NCFG);
    for (int i = 0; i < NCFG; ++i) med[i] = median_of(t[i]);
    for (int i = 0; i < NCFG; ++i) {
        if (g_cfgs[i].tag == TAG_FLOOR) R.floor_ = med[i];
        if (g_cfgs[i].tag == TAG_GRAPH) R.graph  = med[i];
        if (g_cfgs[i].tag == TAG_PUB)   R.pub    = med[i];
    }
    const double bps = sh.bytes / nst;
    printf("    us per stage, median of %d round-robin rounds of %d passes\n", rounds, REPS);
    printf("    %-38s %8s %8s %8s %9s %9s %7s%s\n", "config", "median", "min", "max",
           "vs floor", "vs graph", "GB/s", sh.layer ? "  us/layer" : "");
    for (int i = 0; i < 110; ++i) putchar('-'); putchar('\n');
    for (int i = 0; i < NCFG; ++i) {
        const float mn = *std::min_element(t[i].begin(), t[i].end());
        const float mx = *std::max_element(t[i].begin(), t[i].end());
        printf("    %-38s %8.3f %8.3f %8.3f %+9.3f %+9.3f %7.1f", g_cfgs[i].name, med[i], mn, mx,
               med[i] - R.floor_, med[i] - R.graph, bps / (med[i] * 1e-6) / 1e9);
        if (sh.layer) printf("  %8.2f", med[i] * sh.layer);
        printf("\n");
        if (g_cfgs[i].tag == TAG_RA)        R.best_ra     = std::min(R.best_ra, med[i]);
        if (g_cfgs[i].tag == TAG_RA_HONEST) R.best_honest = std::min(R.best_honest, med[i]);
        if (g_cfgs[i].tag == TAG_TAIL)      R.best_tail   = std::min(R.best_tail, med[i]);
    }
    for (int i = 0; i < NCFG; ++i) {
        if (ge[i]) CK(cudaGraphExecDestroy(ge[i]));
        if (gr[i]) CK(cudaGraphDestroy(gr[i]));
    }

    const float bnd = R.graph - R.floor_, pub = R.pub - R.floor_;
    const float ra = R.best_ra - R.floor_, hon = R.best_honest - R.floor_, tail = R.best_tail - R.floor_;
    printf("\n    per-stage overhead above the no-barrier floor (%.3f us):\n", R.floor_);
    printf("      launch boundary (CUDA graph)            %+7.3f\n", bnd);
    printf("      barrier, wait-then-load (published)     %+7.3f\n", pub);
    printf("      split barrier + run-ahead, best         %+7.3f\n", ra);
    printf("      same, xq staged per stage, best         %+7.3f   <- the honest one\n", hon);
    printf("      graph + tail prefetch (late), best      %+7.3f\n", tail);
    const char * v = (hon <= 0.5f) ? "PREDICTION HELD (<= 0.5 us)"
                   : (hon < bnd)   ? "PARTIAL: beats the launch boundary, misses 0.5 us"
                                   : "FALSIFIED: does not beat the launch boundary";
    printf("    verdict on the honest run-ahead variant: %s\n", v);
    if (R.mism) printf("    WARNING: %d configs failed bit-identity; their timings are not valid\n", R.mism);
    printf("\n");
    return R;
}

int main(int argc, char ** argv) {
    const int rounds = (argc > 1) ? atoi(argv[1]) : 5;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s  SMs=%d  L2=%d MB  smem/SM=%zu KB\n",
           prop.name, prop.multiProcessorCount, (int) (prop.l2CacheSize >> 20),
           prop.sharedMemPerMultiprocessor >> 10);

    Dev dv;
    CK(cudaStreamCreate(&dv.st));
    dv.nb = RA_PER_SM * prop.multiProcessorCount;

    // Residency: the barrier deadlocks unless every block is resident, and a
    // config with lower occupancy would not be the same experiment anyway.
    printf("residency (need %d blocks/SM for the %d-block grid):\n", RA_PER_SM, dv.nb);
    for (int i = 0; i < NCFG; ++i) {
        if (g_cfgs[i].kind != KIND_PERSIST) continue;
        // 6 x (9216 + 2304 + 1024 reserved) B needs more than the 64 KB
        // carveout.  Pin the largest, so the driver cannot pick a smaller one at
        // launch and leave blocks unscheduled behind a spinning barrier.
        if (g_cfgs[i].smem || g_cfgs[i].xs)
            CK(cudaFuncSetAttribute((const void *) g_cfgs[i].pk,
                                    cudaFuncAttributePreferredSharedMemoryCarveout,
                                    cudaSharedmemCarveoutMaxShared));
        int n = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void *) g_cfgs[i].pk,
                                                         RA_THREADS, g_cfgs[i].smem));
        cudaFuncAttributes fa;
        CK(cudaFuncGetAttributes(&fa, (const void *) g_cfgs[i].pk));
        printf("    %-38s %d blocks/SM  %3d regs  %5zu B smem  %4zu B local\n", g_cfgs[i].name, n,
               fa.numRegs, g_cfgs[i].smem + fa.sharedSizeBytes, fa.localSizeBytes);
        if (n < RA_PER_SM) { printf("FATAL: %s is not fully resident\n", g_cfgs[i].name); return 2; }
    }
    if (dv.nb < 84) { printf("FATAL: grid narrower than the widest tree\n"); return 2; }
    printf("\n");

    // ---- weights: one pool of random Q4_0 rows, sized for the largest shape
    const int SLAB = 12288, NSLAB = 31;
    const size_t slab_n = (size_t) SLAB * RA_BPR;
    const size_t pool_rows = (size_t) SLAB * NSLAB;
    std::vector<uint4> h_q(slab_n); std::vector<__half> h_d(slab_n);
    for (size_t i = 0; i < slab_n; ++i) {
        h_q[i].x = rnd(); h_q[i].y = rnd(); h_q[i].z = rnd(); h_q[i].w = rnd();
        h_d[i] = __float2half(0.005f + 0.001f * (rnd() % 7));
    }
    uint4 * d_q; __half * d_d;
    CK(cudaMalloc(&d_q, pool_rows * RA_BPR * sizeof(uint4)));
    CK(cudaMalloc(&d_d, pool_rows * RA_BPR * sizeof(__half)));
    for (int s = 0; s < NSLAB; ++s) {
        CK(cudaMemcpy(d_q + s * slab_n, h_q.data(), slab_n * sizeof(uint4), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_d + s * slab_n, h_d.data(), slab_n * sizeof(__half), cudaMemcpyHostToDevice));
    }
    std::vector<block_q8_1> h_xq(RA_BPR);
    for (int k = 0; k < RA_BPR; ++k) {
        h_xq[k].ds = __floats2half2_rn(0.01f, 0.5f);
        for (int j = 0; j < 32; ++j) h_xq[k].qs[j] = (int8_t) ((int) (rnd() % 255) - 127);
    }
    block_q8_1 * d_xq; CK(cudaMalloc(&d_xq, RA_BPR * sizeof(block_q8_1)));
    CK(cudaMemcpy(d_xq, h_xq.data(), RA_BPR * sizeof(block_q8_1), cudaMemcpyHostToDevice));
    dv.xq = d_xq;
    CK(cudaMalloc(&dv.bar, 128));
    CK(cudaMalloc(&dv.bp, sizeof(lk_bp_t)));
    CK(cudaMalloc(&dv.leaf, LK_LEAF_U32 * sizeof(unsigned)));
    float * d_y; float * d_yref;
    CK(cudaMalloc(&d_y, 12288 * sizeof(float)));
    CK(cudaMalloc(&d_yref, 12288 * sizeof(float)));
    std::vector<float> h_y(12288), h_ref(12288);

    auto at = [&](size_t row0, int rows) {
        if (row0 + rows > pool_rows) { printf("FATAL: pool overflow\n"); exit(2); }
        Stage s; s.qs = d_q + row0 * RA_BPR; s.d = d_d + row0 * RA_BPR;
        s.rows = rows; s.big = 0; s.nxt = -1; s.pad = 0;
        return s;
    };

    Shape big; big.name = "(B) CHAIN, 12288 rows x 31 stages -- the published shape";
    for (int s = 0; s < 31; ++s) big.st.push_back(at((size_t) s * 12288, 12288));
    big.nchk = 8;

    Shape sml; sml.name = "(C) CHAIN, 2048 rows x 130 stages -- most warps idle";
    for (int s = 0; s < 130; ++s) sml.st.push_back(at((size_t) s * 2048, 2048));
    sml.nchk = 8;

    Shape eng; eng.name = "(D) ENGINE-LIKE LAYER x 12: 6144/64/2048/64/64/64/12288/64/8192 rows";
    {
        const int pat[9] = {6144, 64, 2048, 64, 64, 64, 12288, 64, 8192};
        size_t off = 0;
        for (int l = 0; l < 12; ++l)
            for (int k = 0; k < 9; ++k) { eng.st.push_back(at(off, pat[k])); off += pat[k]; }
        eng.nchk = 18;
        eng.layer = 9;
    }

    const ShapeResult rb = run_shape(big, dv, d_y, d_yref, h_y.data(), h_ref.data(), rounds);
    const ShapeResult rs = run_shape(sml, dv, d_y, d_yref, h_y.data(), h_ref.data(), rounds);
    const ShapeResult re = run_shape(eng, dv, d_y, d_yref, h_y.data(), h_ref.data(), rounds);

    printf("=====================================================================\n");
    printf("SUMMARY: per-stage overhead above the no-barrier floor, us\n");
    printf("    %-10s %10s %10s %10s %10s %10s\n", "shape", "graph", "published", "split RA",
           "RA honest", "tail pf");
    const ShapeResult * rr[3] = {&rb, &rs, &re};
    const char * nm[3] = {"(B) 12288", "(C) 2048", "(D) engine"};
    for (int i = 0; i < 3; ++i)
        printf("    %-10s %+10.3f %+10.3f %+10.3f %+10.3f %+10.3f\n", nm[i],
               rr[i]->graph - rr[i]->floor_, rr[i]->pub - rr[i]->floor_,
               rr[i]->best_ra - rr[i]->floor_, rr[i]->best_honest - rr[i]->floor_,
               rr[i]->best_tail - rr[i]->floor_);
    printf("    (D) us per layer: floor %.2f, graph %.2f, graph+tail pf %.2f, honest run-ahead %.2f\n",
           re.floor_ * 9, re.graph * 9, re.best_tail * 9, re.best_honest * 9);

    CK(cudaFree(d_q)); CK(cudaFree(d_d)); CK(cudaFree(d_xq));
    CK(cudaFree(dv.bar)); CK(cudaFree(dv.bp)); CK(cudaFree(dv.leaf));
    CK(cudaFree(d_y)); CK(cudaFree(d_yref));
    CK(cudaStreamDestroy(dv.st));
    return (rb.mism + rs.mism + re.mism) ? 1 : 0;
}
