// test_normfuse.cu - Gate G7: the Phase 4x fused RMSNorm epilogue.
//
// k_gemv_q4_0_norm runs the FFN norm inside the projection that produces its
// input, with the last block out doing the reduction.  Two things can go wrong
// and neither is visible in a benchmark:
//
//   1. the epilogue computes a DIFFERENT norm from k_rmsnorm_q8_1.  The
//      reduction tree genuinely differs (the 2048 elements are partitioned
//      across threads differently), so this is a tolerance check, not an
//      equality check -- but it must be a tolerance check against the real
//      kernel, not against a hope.
//
//   2. the done-counter is not reset, so the epilogue is SKIPPED on the next
//      replay.  This is the dangerous one: the kernel still runs, the GEMV
//      still produces correct rows, and the only symptom is that xq and y32
//      hold the PREVIOUS step's values.  It produces plausible output.  The
//      oracle parity gate moved 191/825 -> 334/825 when this path was enabled,
//      which is exactly the kind of "improvement" that would have hidden it.
//
// So the test drives three consecutive launches with DIFFERENT inputs and
// checks each against the reference kernel independently.  Repeating the same
// input would not catch a skipped epilogue.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/kernels.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static uint64_t g_seed = 0x243f6a8885a308d3ull;
static double urand() {
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17;
    return (double)((g_seed >> 11) & ((1ull << 53) - 1)) / (double)(1ull << 53);
}
static double nrand() { return urand() * 2.0 - 1.0; }

static int g_fail = 0;

// One round: y[] is preloaded with `x`, the GEMV weights are all zero so the
// projection contributes exactly 0, and the epilogue must therefore normalise
// `x` -- the same job k_rmsnorm_q8_1 does in its own kernel.
static void run_round(const char * label, uint4 * d_qs, __half * d_d, block_q8_1 * d_xq,
                      float * d_y, float * d_w, float * d_ref32, block_q8_1 * d_refq8,
                      float * d_out32, block_q8_1 * d_outq8,
                      float * d_partial, unsigned int * d_ctr,
                      const std::vector<float> & x, int round) {
    const int n = LK_HIDDEN;
    CK(cudaMemcpy(d_y, x.data(), n * 4, cudaMemcpyHostToDevice));
    // Poison the outputs so a SKIPPED epilogue is visible rather than silently
    // matching a stale-but-correct-looking buffer.
    CK(cudaMemset(d_out32, 0xFF, n * 4));
    CK(cudaMemset(d_outq8, 0xFF, 64 * sizeof(block_q8_1)));
    CK(cudaMemset(d_ref32, 0, n * 4));
    CK(cudaMemset(d_refq8, 0, 64 * sizeof(block_q8_1)));

    k_rmsnorm_q8_1<<<1, 256, LK_HIDDEN * 4>>>(d_y, d_w, n, LK_EPS, d_ref32, nullptr, d_refq8);

    LKNormEpilogue ne;
    ne.w = d_w; ne.eps = LK_EPS; ne.y32 = d_out32; ne.yq8 = d_outq8;
    ne.partial = d_partial; ne.ctr = d_ctr;
    k_gemv_q4_0_norm<<<LK_GEMV_BLOCKS(LK_HIDDEN), 256>>>(
        d_qs, d_d, d_xq, LK_HIDDEN, 64, d_y, /*add_residual=*/1, ne);
    CK(cudaDeviceSynchronize());

    std::vector<float> ref32(n), out32(n);
    std::vector<block_q8_1> refq8(64), outq8(64);
    CK(cudaMemcpy(ref32.data(), d_ref32, n * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(out32.data(), d_out32, n * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(refq8.data(), d_refq8, 64 * sizeof(block_q8_1), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(outq8.data(), d_outq8, 64 * sizeof(block_q8_1), cudaMemcpyDeviceToHost));

    double d32 = 0.0;
    for (int i = 0; i < n; ++i) d32 = fmax(d32, fabs((double) ref32[i] - out32[i]));

    // Compare the Q8_1 payload, not just the shared scales: a metric that only
    // looks at ds would pass while every activation byte was stale.
    double dq = 0.0;
    for (int b = 0; b < 64; ++b) {
        for (int i = 0; i < 32; ++i)
            dq = fmax(dq, fabs((double) refq8[b].qs[i] - (double) outq8[b].qs[i]));
        dq = fmax(dq, fabs((double) __low2float(refq8[b].ds) - (double) __low2float(outq8[b].ds)));
    }

    unsigned int ctr = 12345;
    CK(cudaMemcpy(&ctr, d_ctr, 4, cudaMemcpyDeviceToHost));

    const double tol32 = 1e-5;     // different reduction tree, same mathematics
    const int    tolq  = 1;        // one int8 step at most
    const bool ok = (d32 <= tol32) && (dq <= tolq) && (ctr == 0);
    if (!ok) ++g_fail;
    printf("  round %d  %-18s  |y32 err|max=%.3e (tol %.0e)  |q8 err|max=%.0f (tol %d)  ctr=%u  %s\n",
           round, label, d32, tol32, dq, tolq, ctr, ok ? "OK" : "FAIL");
}

int main() {
    printf("fused RMSNorm epilogue: LK_FUSE_NORM=%d LK_NORM_PARTIALS=%d\n",
           LK_FUSE_NORM, LK_NORM_PARTIALS);

    const int n = LK_HIDDEN, bpr = 64;
    const size_t blk = (size_t) n * bpr;

    uint4 * d_qs; __half * d_d; block_q8_1 * d_xq;
    float * d_y, * d_w, * d_ref32, * d_out32, * d_partial;
    block_q8_1 * d_refq8, * d_outq8;
    unsigned int * d_ctr;
    CK(cudaMalloc(&d_qs, blk * sizeof(uint4)));
    CK(cudaMalloc(&d_d, blk * sizeof(__half)));
    CK(cudaMalloc(&d_xq, 64 * sizeof(block_q8_1)));
    CK(cudaMalloc(&d_y, n * 4));
    CK(cudaMalloc(&d_w, n * 4));
    CK(cudaMalloc(&d_ref32, n * 4));
    CK(cudaMalloc(&d_out32, n * 4));
    CK(cudaMalloc(&d_refq8, 64 * sizeof(block_q8_1)));
    CK(cudaMalloc(&d_outq8, 64 * sizeof(block_q8_1)));
    CK(cudaMalloc(&d_partial, LK_NORM_PARTIALS * 4));
    CK(cudaMalloc(&d_ctr, 4));

    // Zero GEMV weights: d == 0 makes every dot product exactly 0, so the
    // projection is a no-op and the epilogue sees `x` itself.
    CK(cudaMemset(d_qs, 0, blk * sizeof(uint4)));
    CK(cudaMemset(d_d, 0, blk * sizeof(__half)));
    CK(cudaMemset(d_xq, 0, 64 * sizeof(block_q8_1)));
    CK(cudaMemset(d_ctr, 0, 4));      // as engine_create does

    std::vector<float> w(n);
    for (auto & v : w) v = (float) nrand() * 0.5f + 1.0f;
    CK(cudaMemcpy(d_w, w.data(), n * 4, cudaMemcpyHostToDevice));

    // Three DIFFERENT inputs, back to back.  A fused epilogue that fails to
    // reset its counter skips rounds 2 and 3 and leaves the poisoned output
    // buffers in place, which is a very loud failure here and a silent one in
    // the engine.
    for (int r = 0; r < 3; ++r) {
        std::vector<float> x(n);
        for (auto & v : x) v = (float) nrand() * 3.0f;
        char lbl[32];
        snprintf(lbl, sizeof lbl, "input %d", r);
        run_round(lbl, d_qs, d_d, d_xq, d_y, d_w, d_ref32, d_refq8, d_out32, d_outq8,
                  d_partial, d_ctr, x, r + 1);
    }

    // And once more with an all-zero input, where every activation quantises to
    // zero and a stale buffer would be maximally visible.
    {
        std::vector<float> x(n, 0.0f);
        run_round("all-zero input", d_qs, d_d, d_xq, d_y, d_w, d_ref32, d_refq8,
                  d_out32, d_outq8, d_partial, d_ctr, x, 4);
    }

    printf("\nGATE G7 (norm fusion): %s\n",
           g_fail ? "FAIL" : "PASS - epilogue matches the standalone norm across replays");
    return g_fail ? 1 : 0;
}
