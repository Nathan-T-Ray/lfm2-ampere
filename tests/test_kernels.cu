// test_kernels.cu - Gate G2: verify the semantic kernels against host-side
// double-precision references.  Uses the REAL shortconv tap weights and the
// REAL MoE router matrix/bias from the pack, because those are the two places
// where layout/semantics errors hide silently.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/kernels.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static uint64_t g_seed = 88172645463325252ull;
static double urand() {                       // xorshift -> [0,1)
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17;
    return (double)((g_seed >> 11) & ((1ull << 53) - 1)) / (double)(1ull << 53);
}
static double nrand() { return (urand() * 2.0 - 1.0); }


// ---- device-side helpers we need to exercise directly ---------------------
__global__ void t_rope(float * x, int n_heads, float pos) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < n_heads) rope_neox_head(x + (size_t)h * LK_HEAD_DIM, pos);
}

static int g_fail = 0;
// Absolute-error check: used whenever the reference is a difference-from-zero
// (where a relative metric is meaningless).
static void check_abs(const char * what, double got, double ref, double tol) {
    const double err = fabs(got - ref);
    const bool ok = err <= tol;
    if (!ok) ++g_fail;
    printf("  %-38s abs=%.3e  tol=%.1e  %s\n", what, err, tol, ok ? "OK" : "FAIL");
}
static void check_rel(const char * what, double got, double ref, double tol) {
    const double err = fabs(got - ref) / (fabs(ref) + 1e-9);
    const bool ok = err <= tol;
    if (!ok) ++g_fail;
    printf("  %-38s got=%+.8f ref=%+.8f rel=%.2e  %s\n", what, got, ref, err, ok ? "OK" : "FAIL");
}

int main(int argc, char ** argv) {
    const char * path = (argc > 1) ? argv[1] : "ref/kernel_fixture.bin";
    FILE * f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    char magic[4]; if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "LKKF", 4)) { printf("bad magic\n"); return 1; }
    auto readv = [&](std::vector<float> & v) {
        uint32_t n = 0; if (fread(&n, 4, 1, f) != 1) exit(1);
        v.resize(n); if (fread(v.data(), 4, n, f) != n) exit(1);
    };
    std::vector<float> conv_w, gate_inp, bias;
    readv(conv_w); readv(gate_inp); readv(bias);
    fclose(f);
    printf("loaded real weights: conv_w=%zu gate_inp=%zu bias=%zu\n",
           conv_w.size(), gate_inp.size(), bias.size());

    // ================= RMSNorm =================
    printf("\n[RMSNorm] eps=%g\n", LK_EPS);
    {
        const int n = LK_HIDDEN;
        std::vector<float> x(n), w(n);
        for (int i = 0; i < n; ++i) { x[i] = (float)nrand(); w[i] = (float)(1.0 + 0.3 * nrand()); }
        double sq = 0; for (int i = 0; i < n; ++i) sq += (double)x[i] * x[i];
        const double inv = 1.0 / sqrt(sq / n + LK_EPS);
        float *dx, *dw, *dy; __half *dy16;
        CK(cudaMalloc(&dx, n * 4)); CK(cudaMalloc(&dw, n * 4));
        CK(cudaMalloc(&dy, n * 4)); CK(cudaMalloc(&dy16, n * 2));
        CK(cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw, w.data(), n * 4, cudaMemcpyHostToDevice));
        k_rmsnorm<<<1, 256, 256 * 4>>>(dx, dw, n, LK_EPS, dy, dy16);
        CK(cudaDeviceSynchronize());
        std::vector<float> y(n); CK(cudaMemcpy(y.data(), dy, n * 4, cudaMemcpyDeviceToHost));
        std::vector<__half> y16(n); CK(cudaMemcpy(y16.data(), dy16, n * 2, cudaMemcpyDeviceToHost));
        double mx = 0, mx16 = 0;
        for (int i = 0; i < n; ++i) {
            mx = std::max(mx, fabs((double)y[i] - (double)x[i] * inv * w[i]));
            mx16 = std::max(mx16, fabs((double)__half2float(y16[i]) - (double)x[i] * inv * w[i]));
        }
        check_abs("max |y32 - ref| (fp32 roundoff)", mx, 0.0, 2e-6);
        check_abs("max |y16 - ref| (fp16 precision)", mx16, 0.0, 2e-3);
        cudaFree(dx); cudaFree(dw); cudaFree(dy); cudaFree(dy16);
    }

    // ================= NEOX RoPE =================
    printf("\n[NEOX RoPE] theta=1e7 head_dim=64 (half-split, NOT adjacent pairs)\n");
    {
        const int nh = 4; const float pos = 12345.0f;
        std::vector<float> x(nh * LK_HEAD_DIM);
        for (auto & v : x) v = (float)nrand();
        std::vector<float> ref = x;
        for (int h = 0; h < nh; ++h) {
            float * r = ref.data() + h * LK_HEAD_DIM;
            double tmp[LK_HEAD_DIM];
            for (int i = 0; i < LK_HEAD_DIM / 2; ++i) {
                const double invf = pow((double)LK_ROPE_THETA, -2.0 * i / (double)LK_HEAD_DIM);
                const double ang = (double)pos * invf;
                const double c = cos(ang), s = sin(ang);
                const double x0 = x[h * LK_HEAD_DIM + i];
                const double x1 = x[h * LK_HEAD_DIM + i + LK_HEAD_DIM / 2];
                tmp[i] = x0 * c - x1 * s;
                tmp[i + LK_HEAD_DIM / 2] = x0 * s + x1 * c;
            }
            for (int i = 0; i < LK_HEAD_DIM; ++i) r[i] = (float)tmp[i];
        }
        float * dx; CK(cudaMalloc(&dx, ref.size() * 4));
        CK(cudaMemcpy(dx, x.data(), ref.size() * 4, cudaMemcpyHostToDevice));
        t_rope<<<1, nh>>>(dx, nh, pos);
        CK(cudaDeviceSynchronize());
        std::vector<float> got(ref.size()); CK(cudaMemcpy(got.data(), dx, ref.size() * 4, cudaMemcpyDeviceToHost));
        double mx = 0; int arg = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            if (fabs((double)got[i] - ref[i]) > mx) { mx = fabs((double)got[i] - ref[i]); arg = (int)i; }
        char lbl[96]; snprintf(lbl, sizeof lbl, "max |got - ref| (elem %d)", arg);
        // Tolerance rationale: the reference evaluates the angle in DOUBLE,
        // while the kernel (like llama.cpp's rope) forms pos*theta^(-2i/d) in
        // fp32.  At pos=12345 the angle reaches ~1.2e4, where one fp32 ulp is
        // ~7e-4, so sin/cos inherit up to ~1.5e-4 of absolute error.  That is a
        // representation limit shared with the oracle, not a formula error --
        // a wrong rotation scheme (e.g. adjacent pairs) would be O(1) here.
        check_abs(lbl, mx, 0.0, 2e-3);
        cudaFree(dx);
    }

    // ================= ShortConv =================
    printf("\n[ShortConv] real taps; out = C*(h0*w0 + h1*w1 + bx*w2)\n");
    {
        const int n = LK_HIDDEN;
        std::vector<float> bcx(3 * n), hist(2 * n), y(n);
        for (int i = 0; i < 3 * n; ++i) bcx[i] = (float)nrand();
        for (int i = 0; i < 2 * n; ++i) hist[i] = (float)(0.5 * nrand());
        std::vector<float> ref(n), refhist = hist;
        for (int ch = 0; ch < n; ++ch) {
            const double B = bcx[ch], C = bcx[n + ch], X = bcx[2 * n + ch];
            const double bx = B * X;
            const double h0 = hist[ch * 2 + 0], h1 = hist[ch * 2 + 1];
            const double w0 = conv_w[0 + ch * 3], w1 = conv_w[1 + ch * 3], w2 = conv_w[2 + ch * 3];
            ref[ch] = (float)(C * (h0 * w0 + h1 * w1 + bx * w2));
            refhist[ch * 2 + 0] = (float)h1;
            refhist[ch * 2 + 1] = (float)bx;
        }
        float *dbcx, *dhist, *dy, *dcw;
        CK(cudaMalloc(&dbcx, 3 * n * 4)); CK(cudaMalloc(&dhist, 2 * n * 4));
        CK(cudaMalloc(&dy, n * 4)); CK(cudaMalloc(&dcw, conv_w.size() * 4));
        CK(cudaMemcpy(dbcx, bcx.data(), 3 * n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dhist, hist.data(), 2 * n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dcw, conv_w.data(), conv_w.size() * 4, cudaMemcpyHostToDevice));
        k_shortconv<<<8, 256>>>(dbcx, 0, dcw, dhist, dy, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<float> gy(n), gh(2 * n);
        CK(cudaMemcpy(gy.data(), dy, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gh.data(), dhist, 2 * n * 4, cudaMemcpyDeviceToHost));
        double my = 0, mh = 0;
        for (int i = 0; i < n; ++i) my = std::max(my, fabs((double)gy[i] - ref[i]));
        for (int i = 0; i < 2 * n; ++i) mh = std::max(mh, fabs((double)gh[i] - refhist[i]));
        check_abs("max |y - ref|", my, 0.0, 1e-5);
        check_abs("max |hist' - ref| (state update)", mh, 0.0, 0.0);   // exact copy
        cudaFree(dbcx); cudaFree(dhist); cudaFree(dy); cudaFree(dcw);
    }

    // ================= MoE router =================
    printf("\n[MoE router] sigmoid; top-4 by (sigmoid+bias); weights UNBIASED + renorm\n");
    {
        std::vector<float> h(LK_HIDDEN);
        for (auto & v : h) v = (float)nrand();
        double lg[LK_N_EXPERTS];
        for (int j = 0; j < LK_N_EXPERTS; ++j) {
            double a = 0; for (int i = 0; i < LK_HIDDEN; ++i) a += (double)gate_inp[j * LK_HIDDEN + i] * h[i];
            lg[j] = a;
        }
        double pr[LK_N_EXPERTS], sel[LK_N_EXPERTS];
        for (int j = 0; j < LK_N_EXPERTS; ++j) { pr[j] = 1.0 / (1.0 + exp(-lg[j])); sel[j] = pr[j] + bias[j]; }
        std::vector<int> ord(LK_N_EXPERTS); std::iota(ord.begin(), ord.end(), 0);
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return sel[a] > sel[b]; });
        double ws = 0; for (int k = 0; k < LK_TOP_K; ++k) ws += pr[ord[k]];
        if (ws < LK_TOPK_GUARD) ws = LK_TOPK_GUARD;

        float *dh, *dgi, *dlogits, *db; int *didx; float *dw;
        CK(cudaMalloc(&dh, LK_HIDDEN * 4));
        CK(cudaMalloc(&dgi, gate_inp.size() * 4));
        CK(cudaMalloc(&dlogits, LK_N_EXPERTS * 4));
        CK(cudaMalloc(&db, LK_N_EXPERTS * 4));
        CK(cudaMalloc(&didx, LK_TOP_K * 4)); CK(cudaMalloc(&dw, LK_TOP_K * 4));
        CK(cudaMemcpy(dh, h.data(), LK_HIDDEN * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dgi, gate_inp.data(), gate_inp.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(db, bias.data(), LK_N_EXPERTS * 4, cudaMemcpyHostToDevice));
        // The router is two kernels now, and the test exercises both: dot into
        // a logits tensor, then select.  Calling only the second one (as an
        // earlier revision of this test did after the split) silently tested
        // nothing at all.
        //
        // k_router_dot gained bias/sel/sig for T4r.  That path is off by
        // default, so sel and sig are null here and the sigmoid stays in
        // k_router_top4 where this test can see it.
        k_router_dot<<<LK_N_EXPERTS, 32>>>(dh, dgi, dlogits, db, nullptr, nullptr, nullptr, 0.0f);
        k_router_top4<<<1, LK_N_EXPERTS>>>(dlogits, db, didx, dw);
        CK(cudaDeviceSynchronize());
        int gi[LK_TOP_K]; float gw[LK_TOP_K];
        CK(cudaMemcpy(gi, didx, LK_TOP_K * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gw, dw, LK_TOP_K * 4, cudaMemcpyDeviceToHost));
        printf("    ref idx:");
        for (int k = 0; k < LK_TOP_K; ++k) printf(" %2d(sel=%+.4f)", ord[k], sel[ord[k]]);
        printf("\n    got idx:");
        for (int k = 0; k < LK_TOP_K; ++k) printf(" %2d(sel=%+.4f)", gi[k], sel[gi[k]]);
        printf("\n");
        for (int k = 0; k < LK_TOP_K; ++k) {
            char lbl[64]; snprintf(lbl, sizeof lbl, "top%d expert id", k);
            check_abs(lbl, gi[k], ord[k], 0.0);
        }
        double mw = 0; for (int k = 0; k < LK_TOP_K; ++k) mw = std::max(mw, fabs((double)gw[k] - pr[ord[k]] / ws));
        check_abs("max |weight - ref|", mw, 0.0, 1e-5);
        double sum = 0; for (int k = 0; k < LK_TOP_K; ++k) sum += gw[k];
        check_rel("sum(weights) == 1", sum, 1.0, 1e-6);
        cudaFree(dh); cudaFree(dgi); cudaFree(dlogits); cudaFree(db); cudaFree(didx); cudaFree(dw);
    }

    printf("\nGATE G2 (kernels): %s\n", g_fail ? "FAIL" : "PASS - all semantic kernels match");
    return g_fail ? 1 : 0;
}
