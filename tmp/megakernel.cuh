// megakernel.cuh - the persistent-kernel scaffold (Phase 5).
//
// WHY.  ARCHITECTURE.md 13.9 measured a kernel boundary at 1.54-2.03 us and an
// intra-kernel grid barrier at 0.06-0.10 us; 13.10 showed the advantage survives
// an 8x spread in per-block work; 13.12 showed a fused kernel fits -- 420 to 504
// blocks resident once the two register outliers are excluded.  Those three
// measurements are the whole case for replacing 343 graph nodes with one
// persistent kernel, and this file is the beginning of that replacement.
//
// THE STRUCTURE.  Every intra-layer boundary in this engine is ALL-TO-ALL, not
// band-local: every output row of every GEMV dots its weight row against the
// WHOLE input vector, so a consumer cannot start until its producers finish
// globally.  band-local producer/consumer flags -- the obvious design -- do not
// apply.  A grid barrier per stage does, and it is the cheaper of the two
// measured options anyway.
//
// THE PORTING RULE, and the only thing that makes this safe.  From
// k_moe_down_accum and k_gemv_q4_0_t, the entire reduction for one output row
// lives inside one warp:
//
//     for (int k = lane; k < bpr; k += 32) acc += dot_q4_0_q8_1(...);
//     warp_sum(acc);            // the same tree, in the same order
//
// So the port is exactly:
//
//     const int warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
// ->
//     for (int warp = w0; warp < nrows; warp += total_warps) { ... }
//
// WHICH WARP AND WHICH BLOCK COMPUTES A ROW IS FREE TO CHANGE.  The lane loop,
// the warp_sum tree, the expert order and the per-token Q8_1 block ownership are
// not.  tests/test_megakernel.cu proves this on a real stage by comparing the
// ported output against the original kernel with memcmp, not a tolerance.
#ifndef LK_MEGAKERNEL_CUH
#define LK_MEGAKERNEL_CUH

#include "kernels.cuh"

// ---------------------------------------------------------------------------
// The grid barrier.
//
// Monotonic and RESET-FREE: `arrived` counts to gen*nb and the last arriver
// releases nb at a time on `released`.  Nothing is ever written back to zero, so
// no fast block can race the reset of a slow one -- which is the usual way a
// hand-rolled grid barrier deadlocks.
//
// Requires EVERY block resident.  ARCHITECTURE.md 13.12 is the budget: 420
// blocks at 48 registers, 504 with __launch_bounds__(256,6).
//
// CAVEAT for the real integration: this counter must be reset between graph
// replays, and a memset node per token would reintroduce a launch.  The intended
// fix is an epoch counter in device memory that the host bumps once per replay,
// with the barrier testing `epoch * nstages + gen`; the kernel then needs no
// reset at all and stays graph-capturable.
// ---------------------------------------------------------------------------
// THE SPIN MUST NOT BE AN ATOMIC RMW, and this cost a measurement to find.  The
// first version spun on `atomicAdd(&bar[1], 0)`, so every waiting block issued a
// read-modify-write to ONE L2 address, hundreds of them continuously.  That
// saturates the slot and delays the release for everyone, including the block
// trying to publish it: on a real engine stage the barrier came out at ~2.3 us
// against the 0.06-0.34 us the synthetic test measured, and the persistent
// version ended up 10% SLOWER than 130 separate launches.  The synthetic test
// never saw it because its blocks were perfectly balanced and arrived together,
// so almost nobody was spinning.
//
// Fix: arrive with an atomic RMW (unavoidable, once), release with a plain
// store, and wait on a volatile LOAD with a backoff.  Reads do not contend for
// ownership the way RMWs do.
__device__ __forceinline__ void lk_gbar(int * __restrict__ bar, int nb, int gen) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();                       // make this stage's stores visible
        if (atomicAdd(&bar[0], 1) + 1 == gen * nb) {
            __threadfence();
            *(volatile int *) &bar[1] = gen;   // release: plain store, monotonic
        }
        while (*(volatile int *) &bar[1] < gen) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(64);                   // back off; do not hammer the line
#endif
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// STAGE: Q4_0 x Q8_1 GEMV, one output row per warp, grid-stride over rows.
//
// This is k_gemv_q4_0_t<NORM_EPILOGUE=false> with the warp index turned into a
// loop.  Every line of the per-row arithmetic is the original's, including the
// position of the ninv multiply, because bit-identity is the acceptance test.
// LK_GEMV_ROWS is 1 and static_asserted elsewhere; this port assumes that.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mk_stage_gemv_q4_0(
        const uint4 * __restrict__ qs, const __half * __restrict__ d,
        const block_q8_1 * __restrict__ xq,
        int out_rows, int bpr, float * __restrict__ y, int add_residual,
        const float * __restrict__ ssq, float eps) {
    static_assert(LK_GEMV_ROWS == 1, "mk_stage_gemv_q4_0 assumes one row per warp");
    const int lane = threadIdx.x & 31;
    // warp-uniform: every lane of a warp evaluates the same loop bounds
    const int total_warps = (int) (((size_t) gridDim.x * blockDim.x) >> 5);
    const int w0          = (int) ((blockIdx.x * blockDim.x + threadIdx.x) >> 5);

    float ninv = 1.0f;
    if (ssq) ninv = lk_norm_inv(ssq, lane, eps);

    for (int row = w0; row < out_rows; row += total_warps) {
        float acc = 0.0f;
        for (int k = lane; k < bpr; k += 32)
            acc += dot_q4_0_q8_1(qs + (size_t) row * bpr + k,
                                 d  + (size_t) row * bpr + k, 0, xq + k);
        const float a = warp_sum(acc) * ninv;
        if (lane == 0) y[row] = add_residual ? (y[row] + a) : a;
    }
}

// ---------------------------------------------------------------------------
// PROBE KERNEL -- not part of the engine.
//
// Runs the same stage `nstages` times with a grid barrier between each, so that
// (a) the barrier is exercised under real load and (b) the result can be
// compared bit-for-bit against the same number of real kernel launches.  This
// is what turns "the port should be bit-identical" into a test that can fail.
// ---------------------------------------------------------------------------
__global__ void k_mk_probe(const uint4 * __restrict__ qs, const __half * __restrict__ d,
                           const block_q8_1 * __restrict__ xq,
                           int out_rows, int bpr, float * __restrict__ y,
                           int nstages, int * __restrict__ bar,
                           const float * __restrict__ ssq, float eps) {
    for (int s = 0; s < nstages; ++s) {
        mk_stage_gemv_q4_0(qs, d, xq, out_rows, bpr, y, s > 0, ssq, eps);
        lk_gbar(bar, gridDim.x, s + 1);
    }
}

// The same loop with NO barrier.  The results are wrong by construction (every
// stage races), so this is a COST DECOMPOSITION only, never a correctness path.
// It exists so the barrier can be priced directly instead of inferred:
//
//     launch boundary = (separate_launches - no_barrier) / nstages
//     barrier cost    = (with_barrier     - no_barrier) / nstages
//
// Without this the two effects are confounded, which is how the first version of
// this test produced a 10% regression that could not be attributed.
__global__ void k_mk_probe_nobar(const uint4 * __restrict__ qs, const __half * __restrict__ d,
                                 const block_q8_1 * __restrict__ xq,
                                 int out_rows, int bpr, float * __restrict__ y,
                                 int nstages, const float * __restrict__ ssq, float eps) {
    for (int s = 0; s < nstages; ++s)
        mk_stage_gemv_q4_0(qs, d, xq, out_rows, bpr, y, s > 0, ssq, eps);
}

#endif  // LK_MEGAKERNEL_CUH
