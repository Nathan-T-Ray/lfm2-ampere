// test_attn.cu - Gate G6: attention equivalence.
//
// Phase 4v changed k_attn_split's grid from (32 query heads x SPLIT) to
// (8 kv_heads x SPLIT) so that the four query heads sharing a kv_head read K
// and V ONCE instead of four times.  That is asserted to be a pure
// data-movement change: same tokens, same dims, same accumulation order, so
// the partials must be BIT-IDENTICAL to the pre-4v kernel.
//
// That assertion was made BEFORE this test existed, and the token trajectory
// A/B immediately falsified it.  This test says where the difference is and
// how big it is, which the token A/B cannot: a flipped argmax looks the same
// whether the underlying error is 1e-7 or 1e-1.
//
// Three references are computed:
//   serial  - k_attention_decode, the pre-existing single-block kernel and the
//             one that scores 66/66 on the oracle parity gate
//   split   - k_attn_split + k_attn_combine, the kernel under test
//   float64 - host reference, exact to ~1e-16, so it measures ACCURACY rather
//             than agreement
//
// Passing this test does not by itself prove the kernel matches the oracle.
// Failing it proves the kernel is wrong, which is the useful direction.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/kernels.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static uint64_t g_seed = 0x9e3779b97f4a7c15ull;
static double urand() {
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17;
    return (double)((g_seed >> 11) & ((1ull << 53) - 1)) / (double)(1ull << 53);
}
static double nrand() { return urand() * 2.0 - 1.0; }

static int g_fail = 0;

// Exact float64 reference.  This is the definition of the function the two
// kernels are two different roundings of.
static void host_ref(const std::vector<float> & q, const std::vector<__half> & kc,
                     const std::vector<__half> & vc, int seq, std::vector<double> & out) {
    out.assign(LK_N_HEADS * LK_HEAD_DIM, 0.0);
    const int G = LK_N_HEADS / LK_N_KV_HEADS;
    for (int head = 0; head < LK_N_HEADS; ++head) {
        const int kv = head / G;
        std::vector<double> s(seq);
        double m = -1e300;
        for (int t = 0; t < seq; ++t) {
            double a = 0.0;
            const __half * k = &kc[((size_t) t * LK_N_KV_HEADS + kv) * LK_HEAD_DIM];
            for (int d = 0; d < LK_HEAD_DIM; ++d)
                a += (double) q[head * LK_HEAD_DIM + d] * (double) __half2float(k[d]);
            a *= (double) LK_ATTN_SCALE;
            s[t] = a;
            if (a > m) m = a;
        }
        double l = 0.0;
        for (int t = 0; t < seq; ++t) { s[t] = exp(s[t] - m); l += s[t]; }
        for (int d = 0; d < LK_HEAD_DIM; ++d) {
            double a = 0.0;
            for (int t = 0; t < seq; ++t) {
                const __half * v = &vc[((size_t) t * LK_N_KV_HEADS + kv) * LK_HEAD_DIM];
                a += s[t] * (double) __half2float(v[d]);
            }
            out[head * LK_HEAD_DIM + d] = a / l;
        }
    }
}

static double maxdev(const std::vector<float> & a, const std::vector<double> & b, int n) {
    double mx = 0.0;
    for (int i = 0; i < n; ++i) mx = fmax(mx, fabs((double) a[i] - b[i]));
    return mx;
}
static double maxdiff(const std::vector<float> & a, const std::vector<float> & b, int n) {
    double mx = 0.0;
    for (int i = 0; i < n; ++i) mx = fmax(mx, fabs((double) a[i] - b[i]));
    return mx;
}

static void run_case(int seq, bool verbose) {
    printf("== seq_len %d ==\n", seq);
    std::vector<float> q(LK_N_HEADS * LK_HEAD_DIM);
    for (auto & x : q) x = (float) nrand() * 1.5f;
    // KV cache is allocated at LK_ATTN_MAX_SEQ so the split kernel's
    // LK_ATTN_CHUNK-sized shared arrays are exercised at full width.
    std::vector<__half> kh(LK_ATTN_MAX_SEQ * LK_N_KV_HEADS * LK_HEAD_DIM);
    std::vector<__half> vh(kh.size());
    for (size_t i = 0; i < kh.size(); ++i) {
        kh[i] = __float2half((float) nrand());
        vh[i] = __float2half((float) nrand());
    }

    float *d_q, *d_part, *d_out_s, *d_out_p;
    __half *d_k, *d_v;
    int *d_pos;
    CK(cudaMalloc(&d_q, q.size() * 4));
    CK(cudaMalloc(&d_k, kh.size() * 2));
    CK(cudaMalloc(&d_v, vh.size() * 2));
    CK(cudaMalloc(&d_part, (size_t) LK_N_HEADS * LK_ATTN_SPLIT * LK_ATTN_PARTW * 4));
    CK(cudaMalloc(&d_out_s, (size_t) LK_N_HEADS * LK_HEAD_DIM * 4));
    CK(cudaMalloc(&d_out_p, (size_t) LK_N_HEADS * LK_HEAD_DIM * 4));
    CK(cudaMalloc(&d_pos, 4));
    CK(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_k, kh.data(), kh.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_v, vh.data(), vh.size() * 2, cudaMemcpyHostToDevice));
    int pos = seq - 1;
    CK(cudaMemcpy(d_pos, &pos, 4, cudaMemcpyHostToDevice));

    // serial reference (the kernel that passes the oracle gate 66/66)
    // The serial kernel is a SECOND OPINION, not the definition of correct --
    // host_ref is the definition.  It is only usable up to
    // LK_ATTN_DECODE_MAX_SEQ, because it keeps its whole score row in a
    // __shared__ float s_score[LK_ATTN_DECODE_MAX_SEQ]; past that it prints
    // FATAL and returns early, leaving d_out_s untouched.  Comparing against an
    // output that was never written is how this test reported a 6.2e-03
    // "divergence" at 131072 from a kernel whose error against the float64
    // reference was 3.6e-09 -- the failure was entirely in the reference.
    const bool do_serial = (seq <= LK_ATTN_DECODE_MAX_SEQ);
    if (do_serial) {
        k_attention_decode<<<LK_N_HEADS, LK_ATTN_BLOCK>>>(d_q, d_k, d_v, d_pos, d_out_s, nullptr);
    }
    // kernel under test
    k_attn_split<<<dim3(LK_N_KV_HEADS, LK_ATTN_SPLIT), 256>>>(d_q, d_k, d_v, d_pos, d_part);
    k_attn_combine<<<LK_N_HEADS, LK_HEAD_DIM>>>(d_part, d_pos, d_out_p, nullptr);
    CK(cudaDeviceSynchronize());

    std::vector<float> os(LK_N_HEADS * LK_HEAD_DIM), op(LK_N_HEADS * LK_HEAD_DIM);
    if (do_serial) CK(cudaMemcpy(os.data(), d_out_s, os.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(op.data(), d_out_p, op.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<double> ref;
    host_ref(q, kh, vh, seq, ref);

    const int n = LK_N_HEADS * LK_HEAD_DIM;
    const double es = do_serial ? maxdev(os, ref, n) : 0.0;   // serial vs exact
    const double ep = maxdev(op, ref, n);                     // split  vs exact
    const double d  = do_serial ? maxdiff(os, op, n) : 0.0;   // split  vs serial

    if (do_serial) {
        printf("  serial vs float64   max abs err = %.3e\n", es);
    } else {
        printf("  serial vs float64   SKIPPED (seq > LK_ATTN_DECODE_MAX_SEQ %d;\n"
               "                      the serial kernel cannot run here at all)\n",
               LK_ATTN_DECODE_MAX_SEQ);
    }
    printf("  split  vs float64   max abs err = %.3e\n", ep);
    if (do_serial) printf("  split  vs serial    max abs diff= %.3e\n", d);

    // Tolerances.
    //
    // split-vs-serial is the SHORT-CONTEXT check (it can only run at seq <=
    // LK_ATTN_DECODE_MAX_SEQ): both are float32.  Phase 4z.8 deliberately broke
    // the accumulation order they used to share -- a shuffle tree over eight
    // 8-dim partials instead of one 64-long serial chain -- so this is now a
    // comparison between two different roundings, and it stays at 1e-4 because
    // the measured value is 7.5e-08, a thousand times inside it.  It is a
    // smoke check that the two kernels agree about what they are computing.
    //
    // split-vs-float64 is the REAL gate, and it covers every length.  It is
    // also the stricter of the two: the coalesced K loop is a TREE, so it is
    // more accurate than the serial chain it replaced, and its measured error
    // is ~1e-8 where the serial kernel's own is ~7e-8.
    const double tol_split_serial = 1e-4;
    const double tol_vs_exact     = 1e-3;
    if (do_serial && d > tol_split_serial) {
        printf("  FAIL: split diverges from serial by %.3e (tol %.1e)\n", d, tol_split_serial);
        ++g_fail;
    }
    if (ep > tol_vs_exact || es > tol_vs_exact) {
        printf("  FAIL: error vs float64 exceeds %.1e\n", tol_vs_exact);
        ++g_fail;
    }
    if (verbose && do_serial && d > 1e-5) {
        // locate the worst head so the bug has an address
        for (int h = 0; h < LK_N_HEADS; ++h) {
            double hd = 0.0;
            for (int dd = 0; dd < LK_HEAD_DIM; ++dd)
                hd = fmax(hd, fabs((double) os[h * LK_HEAD_DIM + dd] - op[h * LK_HEAD_DIM + dd]));
            if (hd > 1e-5) printf("    head %2d (kv %d): %.3e\n", h, h / (LK_N_HEADS / LK_N_KV_HEADS), hd);
        }
    }
    CK(cudaFree(d_q)); CK(cudaFree(d_k)); CK(cudaFree(d_v));
    CK(cudaFree(d_part)); CK(cudaFree(d_out_s)); CK(cudaFree(d_out_p)); CK(cudaFree(d_pos));
}

int main() {
    printf("attention equivalence: LK_ATTN_SPLIT=%d LK_ATTN_GQA=%d LK_ATTN_CHUNK=%d\n",
           LK_ATTN_SPLIT, LK_N_HEADS / LK_N_KV_HEADS, LK_ATTN_CHUNK);
    // SHORT AND UNEVEN.  4096 -> even slices, all non-empty.  100 -> uneven.
    // 5 -> SPLIT-1 empty slices, which is where the early-out path and the
    // combine's -1e30 sentinel interact.  1 -> the very first decode step.
    run_case(4096, true);
    run_case(8192, true);      // the cap: slices == LK_ATTN_SPLIT
    run_case(513, true);       // just above the SPLIT_MIN * TARGET_N knee
    run_case(512, true);       // exactly at it
    run_case(100, true);
    run_case(5, true);
    run_case(1, true);
    // ---------------------------------------------------------------------
    // LONG CONTEXT.  Added for Phase 4z.8, and they run last because they are
    // the expensive cases.
    //
    // This test used to stop at 8192, which is exactly where LK_ATTN_SPLIT
    // stops mattering: at 8192 the slice rule saturates (nslices == SPLIT ==
    // 128, chunk == TARGET_N == 64) and EVERY LONGER SEQUENCE IS THE SAME CODE
    // PATH WITH A BIGGER NUMBER.  So the shipped test validated the one regime
    // in which a long-context addressing bug cannot appear.
    //
    // The 4z.8 K loop is warp-coalesced and NOT bit-identical -- each lane sums
    // 8 dims and a shuffle tree combines the eight partials, where the kernel it
    // replaces summed all 64 in one thread -- so tools/ab.py cannot gate it and
    // this test is the only thing that can.  It is the gate, not a smoke test,
    // and it has to cover the sizes the engine actually runs at.
    //
    // 100003 is here on purpose: it is prime, so it is not a multiple of the
    // warp's 4-token stride, of 32, of TARGET_N or of SPLIT, and it exercises
    // the dead-lane tail (n % 32 != 0) that the interior cases never reach.
    run_case(16384, true);     // chunk 128
    run_case(32768, true);     // chunk 256
    run_case(65536, true);     // chunk 512
    run_case(100003, true);    // chunk 782, and a ragged tail
    run_case(131072, true);    // chunk 1024 == LK_ATTN_CHUNK, the maximum
    printf("\n%s\n", g_fail ? "ATTENTION TEST FAILED" : "ATTENTION TEST PASSED");
    return g_fail ? 1 : 0;
}
