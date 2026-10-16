// test_normscale.cu - Gate G8: the Phase 4y scale-invariant Q8_1 norm split.
//
// An RMSNorm followed by a Q8_1 quantisation is split across a kernel boundary
// with no synchronisation: the producer (k_moe_accum4, or k_gemv_q4_0_prod)
// quantises the RAW x*w and emits 64 sum-of-squares partials; the consumer
// applies ONE multiply per output row.  See kernels.cuh for why that is exact
// up to fp32 reassociation.
//
// Three things can go wrong, and NONE of them is visible in a benchmark:
//
//   1. the scale is never applied, so every activation is off by one factor of
//      inv_rms.  This is the loud one in a loss curve and invisible in a
//      throughput number.
//
//   2. the partials are stale or incomplete, so inv_rms is wrong by a little.
//      This is the dangerous one: it produces plausible output.
//
//   3. the int8 payload silently diverges.  The producer rounds
//      round(u*127/amax_u) where the reference rounds round((u*inv)*127/amax),
//      which reassociate at ~1e-7 and flip a rint occasionally.  That is
//      expected and must be BOUNDED, not zero -- asserting zero would be
//      asserting something false.
//
// The test therefore carries an explicit negative control: it re-runs the
// consumer with ssq = nullptr (i.e. inv = 1) on the same raw activations and
// requires that result to be WRONG.  Without that, a tolerance check on a
// mechanism that did nothing at all would pass.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/kernels.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static uint64_t g_seed = 0x13198a2e03707344ull;
static double urand() {
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17;
    return (double)((g_seed >> 11) & ((1ull << 53) - 1)) / (double)(1ull << 53);
}
static double nrand() { return urand() * 2.0 - 1.0; }

static int g_fail = 0;

// Relative error of a vector against a reference, in the max norm.
static double rel_err(const std::vector<float> & ref, const std::vector<float> & got) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        num = fmax(num, fabs((double) ref[i] - got[i]));
        den = fmax(den, fabs((double) ref[i]));
    }
    return den > 0.0 ? num / den : num;
}

struct Bufs {
    uint4 * qs; __half * d;              // the Q4_0 weight matrix
    block_q8_1 * xq_ref, * xq_raw;
    float * x, * w, * down4, * dw, * ssq;
    float * y_ref, * y_fused, * y_noscale;
    int rows, bpr;
};

// One round at a given input.  Everything the mechanism touches is written
// fresh, and the outputs are poisoned first so a skipped step is loud.
static void run_round(Bufs & b, const std::vector<float> & xin, int round, bool zero_input) {
    const int n = LK_HIDDEN;

    CK(cudaMemcpy(b.x, xin.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(b.xq_raw, 0x5A, 64 * sizeof(block_q8_1)));
    CK(cudaMemset(b.ssq, 0xFF, LK_NORM_BLOCKS * 4));
    CK(cudaMemset(b.y_ref, 0, b.rows * 4));
    CK(cudaMemset(b.y_fused, 0, b.rows * 4));
    CK(cudaMemset(b.y_noscale, 0, b.rows * 4));

    // ---- reference: the norm as its own kernel, then a plain consumer -------
    k_rmsnorm_q8_1<<<1, 256, n * 4>>>(b.x, b.w, n, LK_EPS, nullptr, nullptr, b.xq_ref);
    k_gemv_q4_0<<<LK_GEMV_BLOCKS(b.rows), 256>>>(b.qs, b.d, b.xq_ref, b.rows, b.bpr,
                                                 b.y_ref, 0, nullptr, 0.0f);

    // ---- fused: producer quantises raw x*w and emits partials ---------------
    // d_w is all zero, so the MoE accumulation contributes exactly 0 and the
    // kernel normalises `x` itself -- the same job k_rmsnorm_q8_1 does.
    LKNormProd np;
    np.w = b.w; np.h32 = nullptr; np.yq8 = b.xq_raw; np.ssq = b.ssq;
    k_moe_accum4<<<8, 256>>>(b.x, b.down4, b.dw, n, np);
    k_gemv_q4_0<<<LK_GEMV_BLOCKS(b.rows), 256>>>(b.qs, b.d, b.xq_raw, b.rows, b.bpr,
                                                 b.y_fused, 0, b.ssq, LK_EPS);

    // ---- negative control: same raw activations, scale NOT applied ----------
    k_gemv_q4_0<<<LK_GEMV_BLOCKS(b.rows), 256>>>(b.qs, b.d, b.xq_raw, b.rows, b.bpr,
                                                 b.y_noscale, 0, nullptr, 0.0f);
    CK(cudaDeviceSynchronize());

    std::vector<float> yr(b.rows), yf(b.rows), yn(b.rows);
    std::vector<block_q8_1> qr(64), qw(64);
    std::vector<float> ssq(LK_NORM_BLOCKS);
    CK(cudaMemcpy(yr.data(), b.y_ref, b.rows * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(yf.data(), b.y_fused, b.rows * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(yn.data(), b.y_noscale, b.rows * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(qr.data(), b.xq_ref, 64 * sizeof(block_q8_1), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(qw.data(), b.xq_raw, 64 * sizeof(block_q8_1), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ssq.data(), b.ssq, LK_NORM_BLOCKS * 4, cudaMemcpyDeviceToHost));

    // How far did the raw int8 payload drift from the reference's?
    int flips = 0, maxstep = 0;
    for (int k = 0; k < 64; ++k)
        for (int i = 0; i < 32; ++i) {
            const int dv = abs((int) qr[k].qs[i] - (int) qw[k].qs[i]);
            if (dv) ++flips;
            maxstep = maxstep > dv ? maxstep : dv;
        }

    // The residual error must be EXPLAINED, not merely bounded.  The only
    // difference left after the int8 payload agrees is that the reference
    // rounds fp16(inv*u) while we compute inv*fp16(u).  So reconstruct the
    // reference's two scale fields from ours and count how many land on the
    // same half.  If the theory is right this is nearly all 64 blocks, and the
    // observed relative error is ~1 ulp of fp16.
    double inv_host = 0.0;
    {
        double ss = 0.0;
        for (int k = 0; k < LK_NORM_BLOCKS; ++k) ss += ssq[k];
        inv_host = 1.0 / sqrt(ss / (double) n + (double) LK_EPS);
    }
    int ds_exact = 0, sum_exact = 0, ds_near = 0, sum_near = 0;
    for (int k = 0; k < 64; ++k) {
        const float d_raw = __low2float(qw[k].ds), s_raw = __high2float(qw[k].ds);
        const float d_ref = __low2float(qr[k].ds), s_ref = __high2float(qr[k].ds);
        // What the reference would have stored had it rounded AFTER scaling,
        // i.e. what our path would produce if the two roundings commuted.
        const float d_got = __half2float(__float2half((float)(inv_host * d_raw)));
        const float s_got = __half2float(__float2half((float)(inv_host * s_raw)));
        if (d_got == d_ref) ++ds_exact;
        if (s_got == s_ref) ++sum_exact;
        // fp16's spacing near v is |v|*2^-10, so one ulp is the right yardstick.
        if (fabsf(d_got - d_ref) <= 1.05e-3f * fabsf(d_ref)) ++ds_near;
        if (fabsf(s_got - s_ref) <= 1.05e-3f * fabsf(s_ref)) ++sum_near;
    }

    // Is the partial array complete and finite?  A hole here is failure mode 2.
    bool ssq_ok = true;
    double ssq_tot = 0.0;
    for (int k = 0; k < LK_NORM_BLOCKS; ++k) {
        if (!std::isfinite(ssq[k])) ssq_ok = false;
        ssq_tot += ssq[k];
    }
    // Independent recomputation of the same sum, on the host, in double.
    double ssq_host = 0.0;
    for (int i = 0; i < n; ++i) ssq_host += (double) xin[i] * (double) xin[i];

    const double e_fused   = rel_err(yr, yf);
    const double e_noscale = rel_err(yr, yn);

    // 2^-11 is fp16's own relative precision.  d and SUM are each rounded once
    // by the format in both paths, and the two roundings do not commute, so the
    // fused path cannot do better than ~1 ulp -- and does not need to.  An
    // earlier draft of this test used 2e-4, which was a guess made before any
    // measurement and is BELOW the format's own precision, i.e. unachievable by
    // construction.  The bound is set from the format, not from the reading.
    const double FP16_ULP = 1.0 / 2048.0;
    const double tol_fused  = zero_input ? 1e-6 : 2.5 * FP16_ULP;
    const double need_wrong = zero_input ? 0.0 : 0.05;   // the control must be visibly wrong
    const bool ok_ssq  = zero_input ? ssq_ok : (ssq_ok && fabs(ssq_tot - ssq_host) <= 1e-4 * ssq_host);

    // Teeth: with the scale omitted the answer MUST be wrong.  For an all-zero
    // input inv is undefined-ish and both paths are zero, so the control is
    // only meaningful for a live input.
    const bool ok_teeth = zero_input ? true : (e_noscale > need_wrong);
    // Everything must be accounted for by ONE mechanism: the two fp16 roundings
    // of the scale fields do not commute.  The reference rounds fp16(inv*u)
    // directly; we round fp16(u) and scale afterwards, which is a double
    // rounding and lands one ulp away on roughly a quarter of blocks (the
    // intermediate error is ~0.25 ulp, so ~25% cross a rounding boundary).
    // That is why the assertion is "within one ulp", not "identical" -- an
    // identical-scale assertion would be asserting something false, and the
    // first draft of this test did exactly that.
    const bool ok_explained = zero_input ? true : (flips == 0 && ds_near == 64 && sum_near == 64);
    const bool ok = (e_fused <= tol_fused) && ok_teeth && ok_ssq && ok_explained;

    if (!ok) ++g_fail;
    printf("  round %d  relerr=%.3e = %.1f fp16-ulp (tol %.1f)  control=%.3e (need >%.2f)  "
           "ssq=%.1f/%.1f  int8_flips=%d  scale_to_1ulp ds=%d/64 sum=%d/64 "
           "(exact %d/%d)  %s\n",
           round, e_fused, e_fused / FP16_ULP, tol_fused / FP16_ULP, e_noscale, need_wrong,
           ssq_tot, ssq_host, flips, ds_near, sum_near, ds_exact, sum_exact, ok ? "OK" : "FAIL");
    if (!ok_teeth)
        printf("      ^ CONTROL DID NOT FAIL: the scale multiply is not being exercised\n");
    if (!ok_ssq)
        printf("      ^ PARTIALS WRONG: sum %.6f vs host %.6f (finite=%d)\n",
               ssq_tot, ssq_host, (int) ssq_ok);
    if (!ok_explained)
        printf("      ^ UNEXPLAINED: int8 flips=%d, scale within 1 ulp %d/%d ds %d/%d sum "
               "-- the residual is NOT just fp16 scale rounding\n",
               flips, ds_near, 64, sum_near, 64);
}

int main() {
    printf("scale-invariant Q8_1 norm: LK_NORM_PROD_ACCUM=%d LK_NORM_PROD_GEMV=%d "
           "LK_NORM_BLOCKS=%d\n", LK_NORM_PROD_ACCUM, LK_NORM_PROD_GEMV, LK_NORM_BLOCKS);

    Bufs b;
    b.rows = LK_HIDDEN; b.bpr = 64;
    const int n = LK_HIDDEN;
    const size_t blk = (size_t) b.rows * b.bpr;

    CK(cudaMalloc(&b.qs, blk * sizeof(uint4)));
    CK(cudaMalloc(&b.d, blk * sizeof(__half)));
    CK(cudaMalloc(&b.xq_ref, 64 * sizeof(block_q8_1)));
    CK(cudaMalloc(&b.xq_raw, 64 * sizeof(block_q8_1)));
    CK(cudaMalloc(&b.x, n * 4));
    CK(cudaMalloc(&b.w, n * 4));
    CK(cudaMalloc(&b.down4, 4 * n * 4));
    CK(cudaMalloc(&b.dw, 4 * 4));
    CK(cudaMalloc(&b.ssq, LK_NORM_BLOCKS * 4));
    CK(cudaMalloc(&b.y_ref, b.rows * 4));
    CK(cudaMalloc(&b.y_fused, b.rows * 4));
    CK(cudaMalloc(&b.y_noscale, b.rows * 4));

    // A real Q4_0 matrix, so the dot product actually runs.  Random nibbles and
    // small positive scales; the values need not be a real model, only enough
    // to make the reduction non-degenerate.
    {
        std::vector<uint4> qs(blk);
        std::vector<__half> d(blk);
        for (auto & v : qs) {
            uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;
            for (int t = 0; t < 8; ++t) {
                w0 |= (uint32_t)(urand() * 16.0) << (4 * t);
                w1 |= (uint32_t)(urand() * 16.0) << (4 * t);
                w2 |= (uint32_t)(urand() * 16.0) << (4 * t);
                w3 |= (uint32_t)(urand() * 16.0) << (4 * t);
            }
            v = make_uint4(w0, w1, w2, w3);
        }
        for (auto & v : d) v = __float2half((float)(urand() * 0.02 + 0.001));
        CK(cudaMemcpy(b.qs, qs.data(), blk * sizeof(uint4), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b.d, d.data(), blk * sizeof(__half), cudaMemcpyHostToDevice));
    }
    CK(cudaMemset(b.down4, 0, 4 * n * 4));
    CK(cudaMemset(b.dw, 0, 4 * 4));

    std::vector<float> w(n);
    for (auto & v : w) v = (float) nrand() * 0.5f + 1.0f;
    CK(cudaMemcpy(b.w, w.data(), n * 4, cudaMemcpyHostToDevice));

    // Three DIFFERENT inputs back to back: a stale or partially-written partial
    // array cannot survive a change of input, so this is the replay hazard.
    for (int r = 0; r < 3; ++r) {
        std::vector<float> x(n);
        for (auto & v : x) v = (float) nrand() * 3.0f;
        run_round(b, x, r + 1, false);
    }

    // All-zero input: every activation quantises to zero, a stale buffer would
    // be maximally visible, and d == 0 must not produce NaN through the
    // amax==0 guard.
    {
        std::vector<float> x(n, 0.0f);
        run_round(b, x, 4, true);
    }

    printf("\nGATE G8 (scale-invariant norm): %s\n", g_fail
           ? "FAIL"
           : "PASS - fused path matches the standalone norm, and the no-scale "
             "control is measurably wrong");
    return g_fail ? 1 : 0;
}
