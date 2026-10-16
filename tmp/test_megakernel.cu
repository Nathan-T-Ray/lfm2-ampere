// test_megakernel.cu - does the persistent-kernel port change a single bit?
//
// This is the acceptance test for Phase 5, and it is deliberately narrow.  It
// takes ONE real engine stage -- the Q4_0 x Q8_1 GEMV, the single most-executed
// kernel shape in the model -- and runs it two ways:
//
//   reference : N separate kernel launches of the shipped k_gemv_q4_0
//   port      : ONE k_mk_probe launch running the stage N times, with a grid
//               barrier between each, exactly as the megakernel would
//
// and compares the two output vectors with **memcmp**.  Not a tolerance, not a
// correlation: bit-for-bit, because the whole economic case for the megakernel
// rests on it being a pure scheduling change.  A port that moves one bit is a
// port that changes the model, and it does not ship.
//
// The porting rule it verifies (src/megakernel.cuh): which warp computes a row
// is free to change; the lane loop, the warp_sum tree and the ninv multiply are
// not.  The reference has one warp per row and the port has each warp walking
// several rows, so if the rule is wrong this test fails loudly.
//
// The same run also prices the change, so a passing test says both "correct" and
// "and here is what it is worth".
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include "megakernel.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

static uint32_t g_seed = 12345u;
static uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }

// mode 0 = separate launches, 1 = persistent WITH barrier, 2 = persistent no barrier
static float timed_launches(const uint4 * qs, const __half * d, const block_q8_1 * xq,
                            int rows, int bpr, float * y, int nstages,
                            int mode, int nb, int * bar,
                            cudaStream_t st, cudaEvent_t a, cudaEvent_t b) {
    for (int w = 0; w < 2; ++w) {
        if (mode == 2) {
            k_mk_probe_nobar<<<nb, 256, 0, st>>>(qs, d, xq, rows, bpr, y, nstages,
                                                 nullptr, 0.0f);
        } else if (mode == 1) {
            CK(cudaMemsetAsync(bar, 0, 2 * sizeof(int), st));
            k_mk_probe<<<nb, 256, 0, st>>>(qs, d, xq, rows, bpr, y, nstages, bar,
                                           nullptr, 0.0f);
        } else {
            for (int s = 0; s < nstages; ++s)
                k_gemv_q4_0<<<LK_GEMV_BLOCKS(rows), 256, 0, st>>>(
                    qs, d, xq, rows, bpr, y, s > 0, nullptr, 0.0f);
        }
    }
    CK(cudaStreamSynchronize(st));
    CK(cudaEventRecord(a, st));
    const int reps = 5;
    for (int r = 0; r < reps; ++r) {
        if (mode == 2) {
            k_mk_probe_nobar<<<nb, 256, 0, st>>>(qs, d, xq, rows, bpr, y, nstages,
                                                 nullptr, 0.0f);
        } else if (mode == 1) {
            CK(cudaMemsetAsync(bar, 0, 2 * sizeof(int), st));
            k_mk_probe<<<nb, 256, 0, st>>>(qs, d, xq, rows, bpr, y, nstages, bar,
                                           nullptr, 0.0f);
        } else {
            for (int s = 0; s < nstages; ++s)
                k_gemv_q4_0<<<LK_GEMV_BLOCKS(rows), 256, 0, st>>>(
                    qs, d, xq, rows, bpr, y, s > 0, nullptr, 0.0f);
        }
    }
    CK(cudaEventRecord(b, st));
    CK(cudaStreamSynchronize(st));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    return ms * 1000.0f / reps;
}

int main() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));

    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    cudaEvent_t ev_a, ev_b;
    CK(cudaEventCreate(&ev_a)); CK(cudaEventCreate(&ev_b));

    // ---- the shape: a 2048x2048 Q4_0 matrix against a 64-block Q8_1 vector,
    //      which is exactly shortconv_out / attn_out in the real layer.
    const int ROWS = LK_HIDDEN;         // 2048 output rows
    const int BPR  = LK_HIDDEN / 32;    // 64 Q8_1 blocks along the input dim

    uint4 * h_qs = new uint4[(size_t) ROWS * BPR];
    __half * h_d = new __half[(size_t) ROWS * BPR];
    block_q8_1 * h_xq = new block_q8_1[BPR];
    float * h_ref = new float[ROWS];
    float * h_port = new float[ROWS];

    for (size_t i = 0; i < (size_t) ROWS * BPR; ++i) {
        h_qs[i].x = rnd(); h_qs[i].y = rnd(); h_qs[i].z = rnd(); h_qs[i].w = rnd();
        // a real Q4_0 d plane is positive and small; keep it that way so the dot
        // is finite and a mismatch would be a real mismatch, not an inf
        h_d[i] = __float2half(0.005f + 0.001f * (rnd() % 7));
    }
    for (int k = 0; k < BPR; ++k) {
        h_xq[k].ds = __floats2half2_rn(0.01f, 0.5f);
        for (int j = 0; j < 32; ++j) h_xq[k].qs[j] = (int8_t) ((int) (rnd() % 255) - 127);
    }

    uint4 * d_qs; __half * d_d; block_q8_1 * d_xq; float * d_ref; float * d_port;
    int * d_bar;
    CK(cudaMalloc(&d_qs, (size_t) ROWS * BPR * sizeof(uint4)));
    CK(cudaMalloc(&d_d,  (size_t) ROWS * BPR * sizeof(__half)));
    CK(cudaMalloc(&d_xq, BPR * sizeof(block_q8_1)));
    CK(cudaMalloc(&d_ref, ROWS * sizeof(float)));
    CK(cudaMalloc(&d_port, ROWS * sizeof(float)));
    CK(cudaMalloc(&d_bar, 2 * sizeof(int)));
    CK(cudaMemcpy(d_qs, h_qs, (size_t) ROWS * BPR * sizeof(uint4), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_d,  h_d,  (size_t) ROWS * BPR * sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_xq, h_xq, BPR * sizeof(block_q8_1), cudaMemcpyHostToDevice));

    int per_sm = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_mk_probe, 256, 0));
    const int NB = per_sm * prop.multiProcessorCount;
    printf("shape: %d rows x %d Q8_1 blocks   Q4_0 weights %.2f MB\n",
           ROWS, BPR, (double) ROWS * BPR * (sizeof(uint4) + sizeof(__half)) / 1e6);
    printf("k_mk_probe occupancy: %d blocks/SM -> NB = %d resident blocks "
           "(%d threads/SM, %.0f%% of the 1536 ceiling)\n\n",
           per_sm, NB, per_sm * 256, 100.0 * per_sm * 256 / 1536.0);

    // =====================================================================
    // (1) BIT-IDENTITY, 8 stages
    // =====================================================================
    const int NST = 8;
    CK(cudaMemset(d_ref, 0, ROWS * sizeof(float)));
    CK(cudaMemset(d_port, 0, ROWS * sizeof(float)));

    for (int s = 0; s < NST; ++s)
        k_gemv_q4_0<<<LK_GEMV_BLOCKS(ROWS), 256, 0, st>>>(
            d_qs, d_d, d_xq, ROWS, BPR, d_ref, s > 0, nullptr, 0.0f);
    CK(cudaMemset(d_bar, 0, 2 * sizeof(int)));
    k_mk_probe<<<NB, 256, 0, st>>>(d_qs, d_d, d_xq, ROWS, BPR, d_port, NST, d_bar,
                                   nullptr, 0.0f);
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());

    CK(cudaMemcpy(h_ref, d_ref, ROWS * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_port, d_port, ROWS * sizeof(float), cudaMemcpyDeviceToHost));

    int first_bad = -1;
    for (int i = 0; i < ROWS; ++i)
        if (memcmp(&h_ref[i], &h_port[i], sizeof(float)) != 0) { first_bad = i; break; }
    const int ident = (first_bad < 0);
    printf("(1) BIT-IDENTITY after %d stages (reference = %d separate launches, "
           "port = 1 launch + %d barriers)\n", NST, NST, NST);
    if (ident) {
        printf("    ALL %d ROWS BIT-IDENTICAL\n", ROWS);
    } else {
        printf("    *** MISMATCH at row %d: ref %.9g  port %.9g ***\n",
               first_bad, h_ref[first_bad], h_port[first_bad]);
    }
    printf("    sample: ref[0]=%.9g port[0]=%.9g   ref[%d]=%.9g port[%d]=%.9g\n\n",
           h_ref[0], h_port[0], ROWS - 1, h_ref[ROWS - 1], ROWS - 1, h_port[ROWS - 1]);

    // =====================================================================
    // (2) THE SAME, WITH THE PHASE-4y SCALE PATH LIVE (ssq non-null)
    //
    // Both paths call the identical lk_norm_inv, so any sane partials array
    // exercises it; what is being tested is that the port applies the scale in
    // the same place, once per row, and not once per block.
    // =====================================================================
    float * d_ssq;
    CK(cudaMalloc(&d_ssq, LK_NORM_PARTIALS * sizeof(float)));
    {
        float h_ssq[LK_NORM_PARTIALS];
        for (int i = 0; i < LK_NORM_PARTIALS; ++i) h_ssq[i] = 8.0f;
        CK(cudaMemcpy(d_ssq, h_ssq, sizeof(h_ssq), cudaMemcpyHostToDevice));
    }
    CK(cudaMemset(d_ref, 0, ROWS * sizeof(float)));
    CK(cudaMemset(d_port, 0, ROWS * sizeof(float)));
    for (int s = 0; s < NST; ++s)
        k_gemv_q4_0<<<LK_GEMV_BLOCKS(ROWS), 256, 0, st>>>(
            d_qs, d_d, d_xq, ROWS, BPR, d_ref, s > 0, d_ssq, LK_EPS);
    CK(cudaMemset(d_bar, 0, 2 * sizeof(int)));
    k_mk_probe<<<NB, 256, 0, st>>>(d_qs, d_d, d_xq, ROWS, BPR, d_port, NST, d_bar,
                                   d_ssq, LK_EPS);
    CK(cudaStreamSynchronize(st));
    CK(cudaMemcpy(h_ref, d_ref, ROWS * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_port, d_port, ROWS * sizeof(float), cudaMemcpyDeviceToHost));
    int bad2 = -1;
    for (int i = 0; i < ROWS; ++i)
        if (memcmp(&h_ref[i], &h_port[i], sizeof(float)) != 0) { bad2 = i; break; }
    printf("(2) SCALE PATH (ssq non-null) after %d stages\n", NST);
    if (bad2 < 0) printf("    ALL %d ROWS BIT-IDENTICAL\n", ROWS);
    else printf("    *** MISMATCH at row %d: ref %.9g  port %.9g ***\n",
                bad2, h_ref[bad2], h_port[bad2]);
    printf("    sample: ref[0]=%.9g port[0]=%.9g\n\n", h_ref[0], h_port[0]);
    const int ident2 = (bad2 < 0);

    // =====================================================================
    // (3) PRICE IT
    // =====================================================================
    const int NST_T = 130;
    const double t_ref  = timed_launches(d_qs, d_d, d_xq, ROWS, BPR, d_ref, NST_T,
                                         0, NB, d_bar, st, ev_a, ev_b);
    const double t_port = timed_launches(d_qs, d_d, d_xq, ROWS, BPR, d_port, NST_T,
                                         1, NB, d_bar, st, ev_a, ev_b);
    const double t_nb   = timed_launches(d_qs, d_d, d_xq, ROWS, BPR, d_port, NST_T,
                                         2, NB, d_bar, st, ev_a, ev_b);
    printf("(3) PRICE, %d stages of the real stage shape:\n", NST_T);
    printf("    %3d separate launches        : %8.1f us  (%.3f us/stage)\n",
           NST_T, t_ref, t_ref / NST_T);
    printf("    1 launch, %d grid barriers    : %8.1f us  (%.3f us/stage)\n",
           NST_T, t_port, t_port / NST_T);
    printf("    1 launch, NO barrier (cost ctl): %8.1f us  (%.3f us/stage)\n",
           t_nb, t_nb / NST_T);
    printf("    ------------------------------------------------------------\n");
    printf("    LAUNCH BOUNDARY  (ref - nobar)/N : %8.3f us\n", (t_ref - t_nb) / NST_T);
    printf("    GRID BARRIER     (bar - nobar)/N : %8.3f us\n", (t_port - t_nb) / NST_T);
    printf("    NET              (ref - bar)/N   : %8.3f us\n", (t_ref - t_port) / NST_T);
    printf("    --> ratio                        : %8.3fx\n\n", t_ref / t_port);

    const int ok = ident && ident2;
    printf("VERDICT: %s\n", ok
           ? "THE PORT IS BIT-IDENTICAL, both scale paths.  The rule holds on a real stage."
           : "*** THE PORT CHANGES BITS.  DO NOT PROCEED. ***");

    delete[] h_qs; delete[] h_d; delete[] h_xq; delete[] h_ref; delete[] h_port;
    CK(cudaFree(d_qs)); CK(cudaFree(d_d)); CK(cudaFree(d_xq));
    CK(cudaFree(d_ref)); CK(cudaFree(d_port)); CK(cudaFree(d_bar)); CK(cudaFree(d_ssq));
    CK(cudaEventDestroy(ev_a)); CK(cudaEventDestroy(ev_b));
    CK(cudaStreamDestroy(st));
    return ok ? 0 : 1;
}
