// test_prefill.cu - gates for the batched prefill kernels (src/prefill.cu).
//
// Three kinds of check, because the kernels make three kinds of claim:
//   BIT-IDENTICAL  the per-token kernels (norm, conv, qk-norm+RoPE, router
//                  top-4) carry the decode kernels' arithmetic, so given the
//                  same input they must produce the same bytes.  memcmp.
//   REASSOCIATION  the tensor-core GEMM sums the same per-block terms in a
//                  different order from the GEMV, so it is held to a float64
//                  reference of those exact terms and to the GEMV itself.
//   FP16 ATTENTION the flash attention is held to a float64 softmax over the
//                  same fp16 K/V/Q values.
#include "../src/prefill.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static int g_fail = 0;
static void report(const char * name, bool ok, const char * fmt, double a = 0, double b = 0) {
    printf("  %-44s %s  ", name, ok ? "PASS" : "FAIL");
    printf(fmt, a, b);
    printf("\n");
    if (!ok) ++g_fail;
}

template <class T> static T * dalloc(size_t n) { T * p; CK(cudaMalloc(&p, n * sizeof(T))); return p; }
template <class T> static void h2d(T * d, const std::vector<T> & h) { CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); }
template <class T> static std::vector<T> d2h(const T * d, size_t n) { std::vector<T> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }

static std::mt19937 rng(12345);
static float urand(float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }

// Activation floats with a realistic spread: mostly O(1), a few outliers.
static std::vector<float> act_floats(size_t n) {
    std::vector<float> v(n);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & x : v) { x = nd(rng); if ((rng() & 63) == 0) x *= 8.0f; }
    return v;
}

struct Q4 { std::vector<uint4> qs; std::vector<__half> d; };
static Q4 rand_q4(size_t rows, int bpr) {
    Q4 w; w.qs.resize(rows * bpr); w.d.resize(rows * bpr);
    for (auto & q : w.qs) { q.x = rng(); q.y = rng(); q.z = rng(); q.w = rng(); }
    for (auto & d : w.d) d = __float2half(urand(0.002f, 0.03f));
    return w;
}

// float64 reference of out[t][n] = sum_b d4 * (d8*sumi - m8), from the planar
// activation the GPU produced (so the reference sees exactly the same quants).
// slack (optional): the rounding PF_EPI2 is allowed, sum over blocks of
// |d4| * half an ulp of c = fma(d8, 1.5*2^23, 8*sum) -- the folded constant the
// GEMM subtracts in place of the exact 8*sum.  Equivalent to half a unit of
// sumi per block, the size of Q8_1's own fp16 rounding of 8*sum.
static double ref_dot(const Q4 & w, size_t wrow, int bpr, const int8_t * aq, const float2 * as, double * mag,
                      double * slack = nullptr) {
    double acc = 0.0, am = 0.0, sl = 0.0;
    for (int b = 0; b < bpr; ++b) {
        const uint4 q = w.qs[wrow * bpr + b];
        const uint32_t qw[4] = { q.x, q.y, q.z, q.w };
        long sumi = 0;
        for (int e = 0; e < 32; ++e) {
            const int byte = e & 15;
            const int nib = (e < 16) ? ((qw[byte >> 2] >> (8 * (byte & 3))) & 15)
                                     : ((qw[byte >> 2] >> (8 * (byte & 3) + 4)) & 15);
            sumi += (long) nib * aq[b * 32 + e];
        }
        const double d4 = (double) __half2float(w.d[wrow * bpr + b]);
        const double term = d4 * ((double) as[b].x * (double) sumi - (double) as[b].y);
        acc += term; am += fabs(term);
        const float c = fabsf(fmaf(as[b].x, 12582912.0f, as[b].y));
        sl += fabs(d4) * 0.5 * (double) (nextafterf(c, INFINITY) - c);
    }
    if (mag) *mag = am;
    if (slack) *slack = sl;
    return acc;
}

// ---------------------------------------------------------------------------
static void test_gemm_dense(int N, int bpr, int M, bool add, bool scale, bool three_seg) {
    char name[96];
    snprintf(name, sizeof name, "gemm dense N=%d K=%d M=%d%s%s%s", N, bpr * 32, M,
             add ? " +res" : "", scale ? " *scale" : "", three_seg ? " 3seg" : "");
    Q4 w = rand_q4(N, bpr);
    uint4 * dq = dalloc<uint4>(w.qs.size()); h2d(dq, w.qs);
    __half * dd = dalloc<__half>(w.d.size()); h2d(dd, w.d);
    std::vector<float> xf = act_floats((size_t) M * bpr * 32);
    float * dx = dalloc<float>(xf.size()); h2d(dx, xf);
    PfAct a{ dalloc<int8_t>((size_t) M * bpr * 32), dalloc<float2>((size_t) M * bpr) };
    k_pf_quant_rows<<<dim3(M, (bpr + 7) / 8), 256>>>(dx, bpr * 32, bpr, a);
    CK(cudaGetLastError());
    std::vector<float> sc(M); for (auto & s : sc) s = urand(0.5f, 2.0f);
    float * dsc = dalloc<float>(M); h2d(dsc, sc);
    std::vector<float> o0((size_t) M * N); for (auto & v : o0) v = urand(-1.0f, 1.0f);
    float * dout = dalloc<float>(o0.size()); h2d(dout, o0);

    PfGemm g{};
    if (three_seg) {
        const int s0 = N / 3 / 128 * 128, s1 = s0 + 128;
        for (int i = 0; i < 3; ++i) { g.qs[i] = dq; g.d[i] = dd; }
        // segment i starts where it says; reuse the same buffer at offsets so the
        // reference stays simple: segment 1 rows map to weight rows s0.., etc.
        g.qs[1] = dq + (size_t) s0 * bpr; g.d[1] = dd + (size_t) s0 * bpr;
        g.qs[2] = dq + (size_t) s1 * bpr; g.d[2] = dd + (size_t) s1 * bpr;
        g.seg_end[0] = s0; g.seg_end[1] = s1; g.seg_end[2] = N;
    } else {
        for (int i = 0; i < 3; ++i) { g.qs[i] = dq; g.d[i] = dd; g.seg_end[i] = N; }
    }
    g.N = N; g.bpr = bpr; g.aqs = a.qs; g.asc = a.sc; g.ntok = M;
    g.scale = scale ? dsc : nullptr; g.out = dout; g.out_ld = N; g.add = add;
    pf_gemm_dense(g, 0);
    CK(cudaDeviceSynchronize());
    std::vector<float> out = d2h(dout, o0.size());
    std::vector<int8_t> aq = d2h(a.qs, (size_t) M * bpr * 32);
    std::vector<float2> as = d2h(a.sc, (size_t) M * bpr);
    double worst = 0.0;
    for (int t = 0; t < M; ++t)
        for (int n = 0; n < N; ++n) {
            double mag, sl;
            double r = ref_dot(w, n, bpr, aq.data() + (size_t) t * bpr * 32, as.data() + (size_t) t * bpr, &mag, &sl);
            const double s = scale ? sc[t] : 1.0;
            r = r * s + (add ? o0[(size_t) t * N + n] : 0.0);
            const double err = std::max(0.0, fabs(out[(size_t) t * N + n] - r) - sl * s) / (mag * s + 1e-30);
            worst = std::max(worst, err);
        }
    // Beyond the allowed EPI2 rounding (slack): fp32 accumulation of bpr terms,
    // ~bpr * 2^-24 of the summed magnitudes.  A layout or fragment bug is O(1).
    report(name, worst < 2e-6 * std::max(1, bpr / 16), "max err / sum|terms| = %.3e", worst);

    // Against the shipped GEMV, for a handful of tokens (block_q8_1 input built
    // by the shipped quantiser from the same floats).
    if (!add && !three_seg && bpr % 32 == 0) {
        block_q8_1 * xq = dalloc<block_q8_1>(bpr);
        float * y = dalloc<float>(N);
        double wg = 0.0;
        for (int t = 0; t < std::min(M, 8); ++t) {
            k_quantize_q8_1<<<bpr, 32>>>(dx + (size_t) t * bpr * 32, xq, bpr);
            k_gemv_q4_0<<<LK_GEMV_BLOCKS(N), 256>>>(dq, dd, xq, N, bpr, y, 0, nullptr, 0.0f);
            CK(cudaDeviceSynchronize());
            std::vector<float> yv = d2h(y, N);
            for (int n = 0; n < N; ++n) {
                double mag, sl;
                ref_dot(w, n, bpr, aq.data() + (size_t) t * bpr * 32, as.data() + (size_t) t * bpr, &mag, &sl);
                const double s = scale ? sc[t] : 1.0;
                wg = std::max(wg, std::max(0.0, fabs(out[(size_t) t * N + n] - yv[n] * s) - sl * s) / (mag * s + 1e-30));
            }
        }
        char n2[96]; snprintf(n2, sizeof n2, "  vs k_gemv_q4_0 (same quants)");
        report(n2, wg < 2e-6 * std::max(1, bpr / 16), "max diff / sum|terms| = %.3e", wg);
        CK(cudaFree(xq)); CK(cudaFree(y));
    }
    CK(cudaFree(dq)); CK(cudaFree(dd)); CK(cudaFree(dx)); CK(cudaFree(a.qs)); CK(cudaFree(a.sc));
    CK(cudaFree(dsc)); CK(cudaFree(dout));
}

// MoE: E experts, M tokens x 4 slots routed at random, gate|up-style 2 segments
// for a_shift = 2, and a down-style single segment over slot rows for a_shift 0.
static void test_gemm_moe(int E, int rows_per_expert, int bpr, int M, int a_shift) {
    char name[96];
    snprintf(name, sizeof name, "gemm moe E=%d rows=%d K=%d M=%d shift=%d", E, rows_per_expert, bpr * 32, M, a_shift);
    const int NS = M * 4;
    const int arows = a_shift ? M : NS;
    Q4 w = rand_q4((size_t) E * rows_per_expert, bpr);
    uint4 * dq = dalloc<uint4>(w.qs.size()); h2d(dq, w.qs);
    __half * dd = dalloc<__half>(w.d.size()); h2d(dd, w.d);
    std::vector<float> xf = act_floats((size_t) arows * bpr * 32);
    float * dx = dalloc<float>(xf.size()); h2d(dx, xf);
    PfAct a{ dalloc<int8_t>((size_t) arows * bpr * 32), dalloc<float2>((size_t) arows * bpr) };
    k_pf_quant_rows<<<dim3(arows, (bpr + 7) / 8), 256>>>(dx, bpr * 32, bpr, a);
    // Routing: 4 distinct experts per token, skewed so some experts get many.
    std::vector<int> idx(NS);
    for (int t = 0; t < M; ++t) {
        int got = 0;
        while (got < 4) {
            int e = (rng() % 3 == 0) ? (int) (rng() % E) : (int) (rng() % std::max(1, E / 4));
            bool dup = false; for (int k = 0; k < got; ++k) dup |= (idx[t * 4 + k] == e);
            if (!dup) idx[t * 4 + got++] = e;
        }
    }
    int * didx = dalloc<int>(NS); h2d(didx, idx);
    int * cnt = dalloc<int>(64); int * off = dalloc<int>(64); int * perm = dalloc<int>(NS);
    const int max_tiles = NS / PF_MOE_BN + 64;
    int4 * tiles = dalloc<int4>(max_tiles); int * ntiles = dalloc<int>(1);
    CK(cudaMemset(cnt, 0, 64 * 4));
    k_pf_moe_count<<<(NS + 255) / 256, 256>>>(didx, NS, cnt);
    k_pf_moe_plan<<<1, 32>>>(cnt, off, tiles, ntiles);
    k_pf_moe_scatter<<<64, 1024>>>(didx, NS, off, perm);
    std::vector<float> sc(M); for (auto & s : sc) s = urand(0.5f, 2.0f);
    float * dsc = dalloc<float>(M); h2d(dsc, sc);
    float * dout = dalloc<float>((size_t) NS * rows_per_expert);
    CK(cudaMemset(dout, 0xFF, (size_t) NS * rows_per_expert * 4));   // NaN: every slot must be written

    PfGemm g{};
    const int half = rows_per_expert / 2;
    if (a_shift) {   // two segments, like gate|up, both offset by the expert
        g.qs[0] = dq; g.d[0] = dd;
        g.qs[1] = g.qs[2] = dq + (size_t) half * bpr; g.d[1] = g.d[2] = dd + (size_t) half * bpr;
        g.seg_end[0] = half; g.seg_end[1] = g.seg_end[2] = rows_per_expert;
    } else {
        for (int i = 0; i < 3; ++i) { g.qs[i] = dq; g.d[i] = dd; g.seg_end[i] = rows_per_expert; }
    }
    g.N = rows_per_expert; g.bpr = bpr; g.exp_stride = (long long) rows_per_expert * bpr;
    g.aqs = a.qs; g.asc = a.sc; g.perm = perm; g.tiles = tiles; g.ntiles = ntiles; g.a_shift = a_shift;
    g.scale = a_shift ? dsc : nullptr; g.out = dout; g.out_ld = rows_per_expert; g.add = 0;
    pf_gemm_moe(g, max_tiles, 0);
    CK(cudaDeviceSynchronize());
    std::vector<float> out = d2h(dout, (size_t) NS * rows_per_expert);
    std::vector<int8_t> aq = d2h(a.qs, (size_t) arows * bpr * 32);
    std::vector<float2> as = d2h(a.sc, (size_t) arows * bpr);
    std::vector<int> pv = d2h(perm, NS);
    // perm must be a permutation, grouped by expert, ascending within a group
    bool perm_ok = true;
    { std::vector<int> s = pv; std::sort(s.begin(), s.end()); for (int i = 0; i < NS; ++i) perm_ok &= (s[i] == i); }
    for (int i = 1; i < NS; ++i) {
        const int e0 = idx[pv[i - 1]], e1 = idx[pv[i]];
        perm_ok &= (e0 < e1) || (e0 == e1 && pv[i - 1] < pv[i]);
    }
    double worst = 0.0;
    for (int s = 0; s < NS; ++s) {
        const int ar = s >> a_shift, e = idx[s];
        for (int n = 0; n < rows_per_expert; ++n) {
            double mag, sl;
            double r = ref_dot(w, (size_t) e * rows_per_expert + n, bpr, aq.data() + (size_t) ar * bpr * 32,
                               as.data() + (size_t) ar * bpr, &mag, &sl);
            const double sv = a_shift ? sc[ar] : 1.0;
            const double err = std::max(0.0, fabs(out[(size_t) s * rows_per_expert + n] - r * sv) - sl * sv) / (mag * sv + 1e-30);
            if (!(err == err)) { worst = 1e30; } else worst = std::max(worst, err);
        }
    }
    report(name, perm_ok && worst < 2e-6 * std::max(1, bpr / 16), "max err / sum|terms| = %.3e", worst);
    if (!perm_ok) printf("    permutation invalid\n");
    CK(cudaFree(dq)); CK(cudaFree(dd)); CK(cudaFree(dx)); CK(cudaFree(a.qs)); CK(cudaFree(a.sc));
    CK(cudaFree(didx)); CK(cudaFree(cnt)); CK(cudaFree(off)); CK(cudaFree(perm)); CK(cudaFree(tiles));
    CK(cudaFree(ntiles)); CK(cudaFree(dsc)); CK(cudaFree(dout));
}

// ---------------------------------------------------------------------------
static void test_attn(int m, int P) {
    char name[96]; snprintf(name, sizeof name, "flash attention m=%d P=%d", m, P);
    const int S = P + m;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<__half> K((size_t) S * 8 * 64), V((size_t) S * 8 * 64), Q((size_t) m * 2048);
    // Q/K are RMS-normed per head in the model; give them that scale (and a
    // few large dims) so the scores span a realistic range.
    for (auto & x : K) x = __float2half(nd(rng) * 1.5f);
    for (auto & x : V) x = __float2half(nd(rng));
    for (auto & x : Q) x = __float2half(nd(rng) * 1.5f * 0.125f);
    __half * dk = dalloc<__half>(K.size()); h2d(dk, K);
    __half * dv = dalloc<__half>(V.size()); h2d(dv, V);
    __half * dqh = dalloc<__half>(Q.size()); h2d(dqh, Q);
    float * dout = dalloc<float>((size_t) m * 2048);
    k_pf_attn<<<dim3((m + 31) / 32, 8), 256>>>(dqh, m, P, dk, dv, dout);
    CK(cudaDeviceSynchronize());
    std::vector<float> o = d2h(dout, (size_t) m * 2048);
    double worst = 0.0;
    std::vector<double> sc(S);
    for (int t = 0; t < m; ++t)
        for (int h = 0; h < 32; ++h) {
            const int kvh = h / 4, qp = P + t;
            double mx = -1e300;
            for (int j = 0; j <= qp; ++j) {
                double s = 0.0;
                for (int d = 0; d < 64; ++d)
                    s += (double) __half2float(Q[(size_t) t * 2048 + h * 64 + d]) *
                         (double) __half2float(K[((size_t) j * 8 + kvh) * 64 + d]);
                sc[j] = s; mx = std::max(mx, s);
            }
            double l = 0.0;
            for (int j = 0; j <= qp; ++j) { sc[j] = exp(sc[j] - mx); l += sc[j]; }
            for (int d = 0; d < 64; ++d) {
                double r = 0.0;
                for (int j = 0; j <= qp; ++j) r += sc[j] * (double) __half2float(V[((size_t) j * 8 + kvh) * 64 + d]);
                r /= l;
                worst = std::max(worst, fabs(o[(size_t) t * 2048 + h * 64 + d] - r));
            }
        }
    // P is rounded to fp16 (2^-11 relative) before the PV product; with |V| ~ 1
    // that bounds the output error near 1e-3.
    report(name, worst < 4e-3, "max |err| = %.3e (V ~ N(0,1))", worst);
    CK(cudaFree(dk)); CK(cudaFree(dv)); CK(cudaFree(dqh)); CK(cudaFree(dout));
}

// INT8-score attention, against the same float64 softmax over the fp16 inputs
// (so the int8 quantisation error is IN the measured error).  koff adds a
// common per-channel offset to every key, the case smoothing exists for.
static double attn8_err(int m, int P, float koff, bool smooth, int near, double * err16) {
    const int S = P + m;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> off(8 * 64);
    for (auto & x : off) x = nd(rng) * koff;
    std::vector<__half> K((size_t) S * 8 * 64), V((size_t) S * 8 * 64), Q((size_t) m * 2048);
    for (size_t i = 0; i < K.size(); ++i) K[i] = __float2half(nd(rng) * 1.5f + off[i % 512]);
    for (auto & x : V) x = __float2half(nd(rng));
    for (auto & x : Q) x = __float2half(nd(rng) * 1.5f * 0.125f);
    if (koff < 0.0f) {
        // Exact int8 grid: every 64-row holds integers with |max| = 127 times a
        // power of two, so quantisation is lossless and only kernel error remains.
        std::uniform_int_distribution<int> ui(-127, 127);
        for (size_t r = 0; r < K.size() / 64; ++r)
            for (int d = 0; d < 64; ++d) K[r * 64 + d] = __float2half((d == 0 ? 127 : ui(rng)) / 32.0f);
        for (size_t r = 0; r < Q.size() / 64; ++r)
            for (int d = 0; d < 64; ++d) Q[r * 64 + d] = __float2half((d == 0 ? 127 : ui(rng)) / 1024.0f);
    }
    __half * dk = dalloc<__half>(K.size()); h2d(dk, K);
    __half * dv = dalloc<__half>(V.size()); h2d(dv, V);
    __half * dqh = dalloc<__half>(Q.size()); h2d(dqh, Q);
    int8_t * q8 = dalloc<int8_t>(Q.size()); float * qsc = dalloc<float>((size_t) m * 32);
    float * qcs = dalloc<float>((size_t) m * 32);
    const int KS = (S + 3) & ~3;   // kstride: 16-byte aligned rows
    int8_t * k8 = dalloc<int8_t>(K.size()); float * ksc = dalloc<float>((size_t) 8 * KS + 4);
    float * cm = dalloc<float>(512);
    float * dout = dalloc<float>((size_t) m * 2048), * dout16 = dalloc<float>((size_t) m * 2048);
    if (smooth) k_pf_kmean<<<8, 1024>>>(dk, std::min(S, 8192), cm);
    k_pf_q8<<<(m * 32 + 7) / 8, 256>>>(dqh, m, smooth ? cm : nullptr, q8, qsc, qcs);
    k_pf_k8<<<(S * 8 + 7) / 8, 256>>>(dk, 0, S, smooth ? cm : nullptr, KS, k8, ksc);
    k_pf_attn8<<<dim3((m + 31) / 32, 8), 256>>>(q8, qsc, qcs, dqh, m, P, k8, ksc, KS, dk, dv, near, near > 0, dout);
    k_pf_attn<<<dim3((m + 31) / 32, 8), 256>>>(dqh, m, P, dk, dv, dout16);
    CK(cudaDeviceSynchronize());
    std::vector<float> o = d2h(dout, (size_t) m * 2048), o16 = d2h(dout16, (size_t) m * 2048);
    double worst = 0.0, worst16 = 0.0;
    std::vector<double> sc(S);
    for (int t = 0; t < m; ++t)
        for (int h = 0; h < 32; ++h) {
            const int kvh = h / 4, qp = P + t;
            double mx = -1e300;
            for (int j = 0; j <= qp; ++j) {
                double s = 0.0;
                for (int d = 0; d < 64; ++d)
                    s += (double) __half2float(Q[(size_t) t * 2048 + h * 64 + d]) *
                         (double) __half2float(K[((size_t) j * 8 + kvh) * 64 + d]);
                sc[j] = s; mx = std::max(mx, s);
            }
            double l = 0.0;
            for (int j = 0; j <= qp; ++j) { sc[j] = exp(sc[j] - mx); l += sc[j]; }
            for (int d = 0; d < 64; ++d) {
                double r = 0.0;
                for (int j = 0; j <= qp; ++j) r += sc[j] * (double) __half2float(V[((size_t) j * 8 + kvh) * 64 + d]);
                r /= l;
                worst   = std::max(worst,   fabs(o[(size_t) t * 2048 + h * 64 + d] - r));
                worst16 = std::max(worst16, fabs(o16[(size_t) t * 2048 + h * 64 + d] - r));
            }
        }
    CK(cudaFree(dk)); CK(cudaFree(dv)); CK(cudaFree(dqh)); CK(cudaFree(q8)); CK(cudaFree(qsc));
    CK(cudaFree(k8)); CK(cudaFree(ksc)); CK(cudaFree(cm)); CK(cudaFree(dout)); CK(cudaFree(dout16));
    CK(cudaFree(qcs));
    if (err16) *err16 = worst16;
    return worst;
}
// bound < 0: report only.  bound > 0: gate on max |err|.  rel: gate on err <=
// fp16-kernel err * rel (the exact-tile path must reproduce k_pf_attn).
static void test_attn8(int m, int P, float koff, bool smooth, int near, double bound, double rel = 0.0) {
    char name[96]; snprintf(name, sizeof name, "int8 attn m=%d P=%d koff=%.0f%s near=%d", m, P, koff, smooth ? " sm" : "", near);
    double e16 = 0.0;
    const double e = attn8_err(m, P, koff, smooth, near, &e16);
    const bool ok = rel > 0.0 ? (e <= e16 * rel + 1e-6) : (bound < 0.0 || e < bound);
    printf("  %-44s %s  max |err| = %.3e  (fp16 kernel %.3e)\n", name, bound < 0.0 && rel == 0.0 ? "INFO" : ok ? "PASS" : "FAIL", e, e16);
    if (!ok) ++g_fail;
}

// ---------------------------------------------------------------------------
// Bit-identity of the per-token kernels against the decode kernels.
__global__ void k_test_inv(const float * ssq, float * out) {
    const float v = lk_norm_inv(ssq, threadIdx.x, LK_EPS);
    if (threadIdx.x == 0) *out = v;
}

static bool same_q8(const std::vector<block_q8_1> & ref, const int8_t * qs, const float2 * sc, int nb) {
    for (int b = 0; b < nb; ++b) {
        if (memcmp(ref[b].qs, qs + b * 32, 32)) return false;
        const float d = __low2float(ref[b].ds), s = 8.0f * __high2float(ref[b].ds);
        if (memcmp(&d, &sc[b].x, 4) || memcmp(&s, &sc[b].y, 4)) return false;
    }
    return true;
}

static void test_norms() {
    const int M = 5;
    std::vector<float> x = act_floats((size_t) M * 2048), w(2048);
    for (auto & v : w) v = urand(0.2f, 1.8f);
    float * dx = dalloc<float>(x.size()); h2d(dx, x);
    float * dw = dalloc<float>(2048); h2d(dw, w);
    PfAct a{ dalloc<int8_t>((size_t) M * 2048), dalloc<float2>((size_t) M * 64) };
    float * inv = dalloc<float>(M); float * h32 = dalloc<float>((size_t) M * 2048);
    block_q8_1 * xq = dalloc<block_q8_1>(64);
    float * ssq = dalloc<float>(64); float * dinv = dalloc<float>(1);
    float * dz = dalloc<float>(4 * 2048); CK(cudaMemset(dz, 0, 4 * 2048 * 4));
    float * dw0 = dalloc<float>(4); CK(cudaMemset(dw0, 0, 16));
    float * dres = dalloc<float>(2048);
    bool plain_ok = true, y4_ok = true, inv_ok = true, h_ok = true;

    k_pf_norm<<<M, 256>>>(dx, dw, 1, a, inv, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<int8_t> q = d2h(a.qs, (size_t) M * 2048);
    std::vector<float2> s = d2h(a.sc, (size_t) M * 64);
    for (int t = 0; t < M; ++t) {
        k_rmsnorm_q8_1<<<1, 256, 2048 * 4>>>(dx + (size_t) t * 2048, dw, 2048, LK_EPS, nullptr, nullptr, xq);
        CK(cudaDeviceSynchronize());
        plain_ok &= same_q8(d2h(xq, 64), q.data() + (size_t) t * 2048, s.data() + (size_t) t * 64, 64);
    }
    report("norm plain  == k_rmsnorm_q8_1 (bytes)", plain_ok, "");

    k_pf_norm<<<M, 256>>>(dx, dw, 0, a, inv, h32);
    CK(cudaDeviceSynchronize());
    q = d2h(a.qs, (size_t) M * 2048); s = d2h(a.sc, (size_t) M * 64);
    std::vector<float> iv = d2h(inv, M), hv = d2h(h32, (size_t) M * 2048);
    for (int t = 0; t < M; ++t) {
        // k_moe_accum4 with zero expert weights leaves xv == dst exactly and
        // runs the Phase 4y producer epilogue on it.
        CK(cudaMemcpy(dres, dx + (size_t) t * 2048, 2048 * 4, cudaMemcpyDeviceToDevice));
        LKNormProd np; np.w = dw; np.h32 = dz; np.yq8 = xq; np.ssq = ssq;
        k_moe_accum4<<<8, 256>>>(dres, dz, dw0, 2048, np);
        k_test_inv<<<1, 32>>>(ssq, dinv);
        CK(cudaDeviceSynchronize());
        y4_ok &= same_q8(d2h(xq, 64), q.data() + (size_t) t * 2048, s.data() + (size_t) t * 64, 64);
        const float ri = d2h(dinv, 1)[0];
        inv_ok &= !memcmp(&ri, &iv[t], 4);
        std::vector<float> rh = d2h(dz, 2048);
        h_ok &= !memcmp(rh.data(), hv.data() + (size_t) t * 2048, 2048 * 4);
    }
    report("norm 4y     == Phase 4y producer (bytes)", y4_ok, "");
    report("norm 4y inv == lk_norm_inv (bits)", inv_ok, "");
    report("norm 4y h32 == producer h32 (bits)", h_ok, "");
    CK(cudaFree(dx)); CK(cudaFree(dw)); CK(cudaFree(a.qs)); CK(cudaFree(a.sc)); CK(cudaFree(inv));
    CK(cudaFree(h32)); CK(cudaFree(xq)); CK(cudaFree(ssq)); CK(cudaFree(dinv)); CK(cudaFree(dz));
    CK(cudaFree(dw0)); CK(cudaFree(dres));
}

static void test_conv() {
    const int M = 7;
    std::vector<float> y = act_floats((size_t) M * 6144), cw(3 * 2048), h0(4096);
    for (auto & v : cw) v = urand(-0.5f, 0.5f);
    for (auto & v : h0) v = urand(-1.0f, 1.0f);
    float * dy = dalloc<float>(y.size()); h2d(dy, y);
    float * dcw = dalloc<float>(cw.size()); h2d(dcw, cw);
    float * hb = dalloc<float>(4096); h2d(hb, h0);
    float * hs = dalloc<float>(4096); h2d(hs, h0);
    PfAct a{ dalloc<int8_t>((size_t) M * 2048), dalloc<float2>((size_t) M * 64) };
    block_q8_1 * xq = dalloc<block_q8_1>(64);
    k_pf_conv<<<dim3(M, 8), 256>>>(dy, dcw, hb, a);
    k_pf_conv_hist<<<8, 256>>>(dy, M, hb);
    CK(cudaDeviceSynchronize());
    std::vector<int8_t> q = d2h(a.qs, (size_t) M * 2048);
    std::vector<float2> s = d2h(a.sc, (size_t) M * 64);
    bool ok = true;
    for (int t = 0; t < M; ++t) {
        k_shortconv<<<8, 256>>>(dy + (size_t) t * 6144, 0, dcw, hs, nullptr, xq);
        CK(cudaDeviceSynchronize());
        ok &= same_q8(d2h(xq, 64), q.data() + (size_t) t * 2048, s.data() + (size_t) t * 64, 64);
    }
    std::vector<float> hb_h = d2h(hb, 4096), hs_h = d2h(hs, 4096);
    report("shortconv   == k_shortconv x M (bytes)", ok, "");
    report("conv state  == k_shortconv state (bits)", !memcmp(hb_h.data(), hs_h.data(), 4096 * 4), "");
    CK(cudaFree(dy)); CK(cudaFree(dcw)); CK(cudaFree(hb)); CK(cudaFree(hs)); CK(cudaFree(a.qs));
    CK(cudaFree(a.sc)); CK(cudaFree(xq));
}

static void test_qknr() {
    const int M = 6, P = 1000, S = P + M;
    std::vector<float> y = act_floats((size_t) M * 3072), qn(64), kn(64);
    for (auto & v : qn) v = urand(0.5f, 1.5f);
    for (auto & v : kn) v = urand(0.5f, 1.5f);
    float * dy = dalloc<float>(y.size()); h2d(dy, y);
    float * dqn = dalloc<float>(64); h2d(dqn, qn);
    float * dkn = dalloc<float>(64); h2d(dkn, kn);
    __half * kc0 = dalloc<__half>((size_t) S * 512); __half * vc0 = dalloc<__half>((size_t) S * 512);
    __half * kc1 = dalloc<__half>((size_t) S * 512); __half * vc1 = dalloc<__half>((size_t) S * 512);
    CK(cudaMemset(kc0, 0, (size_t) S * 1024)); CK(cudaMemset(vc0, 0, (size_t) S * 1024));
    CK(cudaMemset(kc1, 0, (size_t) S * 1024)); CK(cudaMemset(vc1, 0, (size_t) S * 1024));
    __half * qh = dalloc<__half>((size_t) M * 2048);
    float * qf = dalloc<float>(2048); float * kf = dalloc<float>(512); float * vf = dalloc<float>(512);
    int * dpos = dalloc<int>(1);
    k_pf_qknr<<<dim3(M, 40), 64>>>(dy, dqn, dkn, P, qh, kc0, vc0);
    bool q_ok = true;
    std::vector<__half> qhv;
    CK(cudaDeviceSynchronize());
    qhv = d2h(qh, (size_t) M * 2048);
    for (int t = 0; t < M; ++t) {
        CK(cudaMemcpy(qf, dy + (size_t) t * 3072, 2048 * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(kf, dy + (size_t) t * 3072 + 2048, 512 * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(vf, dy + (size_t) t * 3072 + 2560, 512 * 4, cudaMemcpyDeviceToDevice));
        const int pos = P + t; h2d(dpos, std::vector<int>{ pos });
        k_qk_norm_rope<<<40, 64>>>(qf, kf, vf, dqn, dkn, dpos, kc1, vc1);
        CK(cudaDeviceSynchronize());
        std::vector<float> qr = d2h(qf, 2048);
        for (int i = 0; i < 2048; ++i) {
            const __half r = __float2half(qr[i] * LK_ATTN_SCALE);
            q_ok &= !memcmp(&r, &qhv[(size_t) t * 2048 + i], 2);
        }
    }
    std::vector<__half> a0 = d2h(kc0, (size_t) S * 512), a1 = d2h(kc1, (size_t) S * 512);
    std::vector<__half> b0 = d2h(vc0, (size_t) S * 512), b1 = d2h(vc1, (size_t) S * 512);
    report("qk-norm+rope K/V cache == decode (bytes)",
           !memcmp(a0.data(), a1.data(), a0.size() * 2) && !memcmp(b0.data(), b1.data(), b0.size() * 2), "");
    report("qk-norm+rope Q == fp16(decode q / 8) (bytes)", q_ok, "");
    CK(cudaFree(dy)); CK(cudaFree(dqn)); CK(cudaFree(dkn)); CK(cudaFree(kc0)); CK(cudaFree(vc0));
    CK(cudaFree(kc1)); CK(cudaFree(vc1)); CK(cudaFree(qh)); CK(cudaFree(qf)); CK(cudaFree(kf));
    CK(cudaFree(vf)); CK(cudaFree(dpos));
}

static void test_router() {
    const int M = 70;
    std::vector<float> h = act_floats((size_t) M * 2048), wr((size_t) 64 * 2048), bias(64), iv(M);
    for (auto & v : wr) v = urand(-0.05f, 0.05f);
    for (auto & v : bias) v = urand(-0.2f, 0.2f);
    for (auto & v : iv) v = urand(0.5f, 1.5f);
    float * dh = dalloc<float>(h.size()); h2d(dh, h);
    float * dwr = dalloc<float>(wr.size()); h2d(dwr, wr);
    float * db = dalloc<float>(64); h2d(db, bias);
    float * dinv = dalloc<float>(M); h2d(dinv, iv);
    float * sel = dalloc<float>((size_t) M * 64); float * sig = dalloc<float>((size_t) M * 64);
    int * idx = dalloc<int>(M * 4); float * w = dalloc<float>(M * 4);
    k_pf_router<<<(M + 63) / 64, 256>>>(dh, M, dwr, db, dinv, sel, sig);
    k_pf_select<<<(M + 7) / 8, 256>>>(sel, sig, M, idx, w);
    CK(cudaDeviceSynchronize());
    std::vector<int> iv1 = d2h(idx, M * 4); std::vector<float> wv1 = d2h(w, M * 4);
    std::vector<float> sg = d2h(sig, (size_t) M * 64);
    float * lg = dalloc<float>(64); int * i2 = dalloc<int>(4); float * w2 = dalloc<float>(4);
    float * ssq = dalloc<float>(64);
    int agree = 0; double wdiff = 0.0, ldiff = 0.0;
    for (int t = 0; t < M; ++t) {
        // decode: logits * ninv, where ninv comes from 64 ssq partials.  Feed a
        // partial vector whose lk_norm_inv is NOT iv[t] in general, so instead
        // compare against logits computed with ssq = null and scale by hand.
        k_router_dot<<<64, 32>>>(dh + (size_t) t * 2048, dwr, lg, db, nullptr, nullptr, nullptr, LK_EPS);
        CK(cudaDeviceSynchronize());
        std::vector<float> l = d2h(lg, 64);
        for (auto & v : l) v *= iv[t];
        h2d(lg, l);
        k_router_top4<<<1, 64>>>(lg, db, i2, w2);
        CK(cudaDeviceSynchronize());
        std::vector<int> ri = d2h(i2, 4); std::vector<float> rw = d2h(w2, 4);
        bool same = true; for (int k = 0; k < 4; ++k) same &= (ri[k] == iv1[t * 4 + k]);
        agree += same;
        if (same) for (int k = 0; k < 4; ++k) wdiff = std::max(wdiff, (double) fabsf(rw[k] - wv1[t * 4 + k]));
        for (int e2 = 0; e2 < 64; ++e2) ldiff = std::max(ldiff, (double) fabsf(1.0f / (1.0f + expf(-l[e2])) - sg[(size_t) t * 64 + e2]));
    }
    report("router top-4 == decode selection", agree >= M - 1, "%.0f/70 tokens identical, max |dw| = %.2e", agree, wdiff);
    report("router sigmoid vs decode", ldiff < 1e-5, "max |dsig| = %.2e", ldiff);
    // The short-prompt router must reproduce the decode router's sigmoids bit
    // for bit.  Both run on the device (host expf differs in the last ulp), with
    // inv = 1, which is exactly k_router_dot's ninv when it is given no ssq.
    {
        std::vector<float> ones(M, 1.0f);
        float * d1 = dalloc<float>(M); h2d(d1, ones);
        k_pf_router_small<<<dim3(64, M), 32>>>(dh, dwr, db, d1, sel, sig);
        CK(cudaDeviceSynchronize());
        std::vector<float> sg2 = d2h(sig, (size_t) M * 64), sl2 = d2h(sel, (size_t) M * 64);
        float * rsel = dalloc<float>(64); float * rsig = dalloc<float>(64);
        bool exact = true;
        for (int t2 = 0; t2 < M; ++t2) {
            k_router_dot<<<64, 32>>>(dh + (size_t) t2 * 2048, dwr, lg, db, rsel, rsig, nullptr, LK_EPS);
            CK(cudaDeviceSynchronize());
            exact &= !memcmp(d2h(rsig, 64).data(), &sg2[(size_t) t2 * 64], 256);
            exact &= !memcmp(d2h(rsel, 64).data(), &sl2[(size_t) t2 * 64], 256);
        }
        report("router (short prompts) == decode (bits)", exact, "");
        CK(cudaFree(d1)); CK(cudaFree(rsel)); CK(cudaFree(rsig));
    }
    CK(cudaFree(dh)); CK(cudaFree(dwr)); CK(cudaFree(db)); CK(cudaFree(dinv)); CK(cudaFree(sel));
    CK(cudaFree(sig)); CK(cudaFree(idx)); CK(cudaFree(w)); CK(cudaFree(lg)); CK(cudaFree(i2));
    CK(cudaFree(w2)); CK(cudaFree(ssq));
}


// ---------------------------------------------------------------------------
// `test_prefill bench`: time the GEMM at the model's shapes and the attention
// at a few context lengths.  Configs are chosen by env (LK_PF_CFG_DENSE /
// LK_PF_CFG_MOE), read once per process, so run the binary once per config.
template <class F> static double time_ms(F f, int reps = 7) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    f(); CK(cudaDeviceSynchronize());
    std::vector<float> t;
    for (int r = 0; r < reps; ++r) {
        CK(cudaEventRecord(a)); f(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static void bench_dense(int N, int bpr, int M) {
    Q4 w = rand_q4(N, bpr);
    uint4 * dq = dalloc<uint4>(w.qs.size()); h2d(dq, w.qs);
    __half * dd = dalloc<__half>(w.d.size()); h2d(dd, w.d);
    std::vector<float> xf = act_floats((size_t) M * bpr * 32);
    float * dx = dalloc<float>(xf.size()); h2d(dx, xf);
    PfAct a{ dalloc<int8_t>((size_t) M * bpr * 32), dalloc<float2>((size_t) M * bpr) };
    k_pf_quant_rows<<<dim3(M, (bpr + 7) / 8), 256>>>(dx, bpr * 32, bpr, a);
    float * dout = dalloc<float>((size_t) M * N);
    PfGemm g{};
    for (int i = 0; i < 3; ++i) { g.qs[i] = dq; g.d[i] = dd; g.seg_end[i] = N; }
    g.N = N; g.bpr = bpr; g.aqs = a.qs; g.asc = a.sc; g.ntok = M; g.out = dout; g.out_ld = N;
    const double ms = time_ms([&] { pf_gemm_dense(g, 0); });
    const double mac = (double) N * bpr * 32 * M;
    printf("  dense N=%5d K=%5d M=%5d  %8.3f ms  %6.1f TMAC/s\n", N, bpr * 32, M, ms, mac / ms / 1e9);
    CK(cudaFree(dq)); CK(cudaFree(dd)); CK(cudaFree(dx)); CK(cudaFree(a.qs)); CK(cudaFree(a.sc)); CK(cudaFree(dout));
}

static void bench_moe(int rows, int bpr, int M, int a_shift) {
    const int E = 64, NS = M * 4, arows = a_shift ? M : NS;
    Q4 w = rand_q4((size_t) E * rows, bpr);
    uint4 * dq = dalloc<uint4>(w.qs.size()); h2d(dq, w.qs);
    __half * dd = dalloc<__half>(w.d.size()); h2d(dd, w.d);
    std::vector<float> xf = act_floats((size_t) arows * bpr * 32);
    float * dx = dalloc<float>(xf.size()); h2d(dx, xf);
    PfAct a{ dalloc<int8_t>((size_t) arows * bpr * 32), dalloc<float2>((size_t) arows * bpr) };
    k_pf_quant_rows<<<dim3(arows, (bpr + 7) / 8), 256>>>(dx, bpr * 32, bpr, a);
    std::vector<int> idx(NS);
    for (int t = 0; t < M; ++t) {
        int got = 0;
        while (got < 4) { int e = rng() % E; bool d = false; for (int k = 0; k < got; ++k) d |= idx[t * 4 + k] == e; if (!d) idx[t * 4 + got++] = e; }
    }
    int * didx = dalloc<int>(NS); h2d(didx, idx);
    int * cnt = dalloc<int>(64); int * off = dalloc<int>(64); int * perm = dalloc<int>(NS);
    const int max_tiles = NS / PF_MOE_BN + 64;
    int4 * tiles = dalloc<int4>(max_tiles); int * ntiles = dalloc<int>(1);
    CK(cudaMemset(cnt, 0, 256));
    k_pf_moe_count<<<(NS + 255) / 256, 256>>>(didx, NS, cnt);
    k_pf_moe_plan<<<1, 32>>>(cnt, off, tiles, ntiles);
    k_pf_moe_scatter<<<64, 1024>>>(didx, NS, off, perm);
    float * dout = dalloc<float>((size_t) NS * rows);
    PfGemm g{};
    for (int i = 0; i < 3; ++i) { g.qs[i] = dq; g.d[i] = dd; g.seg_end[i] = rows; }
    g.N = rows; g.bpr = bpr; g.exp_stride = (long long) rows * bpr;
    g.aqs = a.qs; g.asc = a.sc; g.perm = perm; g.tiles = tiles; g.ntiles = ntiles; g.a_shift = a_shift;
    g.out = dout; g.out_ld = rows;
    const double ms = time_ms([&] { pf_gemm_moe(g, max_tiles, 0); });
    const double mac = (double) NS * rows * bpr * 32;
    printf("  moe   N=%5d K=%5d slots=%5d %8.3f ms  %6.1f TMAC/s\n", rows, bpr * 32, NS, ms, mac / ms / 1e9);
    CK(cudaFree(dq)); CK(cudaFree(dd)); CK(cudaFree(dx)); CK(cudaFree(a.qs)); CK(cudaFree(a.sc));
    CK(cudaFree(didx)); CK(cudaFree(cnt)); CK(cudaFree(off)); CK(cudaFree(perm)); CK(cudaFree(tiles));
    CK(cudaFree(ntiles)); CK(cudaFree(dout));
}

static void bench_attn(int m, int P) {
    const int S = P + m;
    std::vector<__half> K((size_t) S * 512), Q((size_t) m * 2048);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & x : K) x = __float2half(nd(rng));
    for (auto & x : Q) x = __float2half(nd(rng) * 0.125f);
    __half * dk = dalloc<__half>(K.size()); h2d(dk, K);
    __half * dv = dalloc<__half>(K.size()); h2d(dv, K);
    __half * dq = dalloc<__half>(Q.size()); h2d(dq, Q);
    float * o = dalloc<float>((size_t) m * 2048);
    const double ms = time_ms([&] { k_pf_attn<<<dim3((m + 31) / 32, 8), 256>>>(dq, m, P, dk, dv, o); }, 5);
    const double mac = 32.0 * 128.0 * m * (P + (m + 1) / 2.0);
    printf("  attn  m=%5d P=%6d  %8.3f ms  %6.1f TMAC/s (useful)\n", m, P, ms, mac / ms / 1e9);
    CK(cudaFree(dk)); CK(cudaFree(dv)); CK(cudaFree(dq)); CK(cudaFree(o));
}

static void bench_attn8(int m, int P) {
    const int S = P + m;
    std::vector<__half> K((size_t) S * 512), Q((size_t) m * 2048);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & x : K) x = __float2half(nd(rng));
    for (auto & x : Q) x = __float2half(nd(rng) * 0.125f);
    __half * dk = dalloc<__half>(K.size()); h2d(dk, K);
    __half * dv = dalloc<__half>(K.size()); h2d(dv, K);
    __half * dq = dalloc<__half>(Q.size()); h2d(dq, Q);
    int8_t * q8 = dalloc<int8_t>(Q.size()); float * qsc = dalloc<float>((size_t) m * 32);
    float * qcs = dalloc<float>((size_t) m * 32);
    const int KS = (S + 3) & ~3;   // kstride: 16-byte aligned rows
    int8_t * k8 = dalloc<int8_t>(K.size()); float * ksc = dalloc<float>((size_t) 8 * KS + 4);
    k_pf_q8<<<(m * 32 + 7) / 8, 256>>>(dq, m, nullptr, q8, qsc, qcs);
    k_pf_k8<<<(S * 8 + 7) / 8, 256>>>(dk, 0, S, nullptr, KS, k8, ksc);
    float * o = dalloc<float>((size_t) m * 2048);
    const double ms = time_ms([&] { k_pf_attn8<<<dim3((m + 31) / 32, 8), 256>>>(q8, qsc, qcs, dq, m, P, k8, ksc, KS, dk, dv, 256, 1, o); }, 5);
    const double mac = 32.0 * 128.0 * m * (P + (m + 1) / 2.0);
    printf("  attn8 m=%5d P=%6d  %8.3f ms  %6.1f TMAC/s (useful)\n", m, P, ms, mac / ms / 1e9);
    CK(cudaFree(dk)); CK(cudaFree(dv)); CK(cudaFree(dq)); CK(cudaFree(o));
    CK(cudaFree(q8)); CK(cudaFree(qsc)); CK(cudaFree(k8)); CK(cudaFree(ksc)); CK(cudaFree(qcs));
}

static void bench_moe_swiglu(int M) {
    const int E = 64, NI = 1536, bpr = 64, NS = M * 4;
    Q4 wg = rand_q4((size_t) E * NI, bpr), wu = rand_q4((size_t) E * NI, bpr);
    uint4 * gq = dalloc<uint4>(wg.qs.size()); h2d(gq, wg.qs); __half * gd = dalloc<__half>(wg.d.size()); h2d(gd, wg.d);
    uint4 * uq = dalloc<uint4>(wu.qs.size()); h2d(uq, wu.qs); __half * ud = dalloc<__half>(wu.d.size()); h2d(ud, wu.d);
    std::vector<float> xf = act_floats((size_t) M * bpr * 32);
    float * dx = dalloc<float>(xf.size()); h2d(dx, xf);
    PfAct a{ dalloc<int8_t>((size_t) M * bpr * 32), dalloc<float2>((size_t) M * bpr) };
    k_pf_quant_rows<<<dim3(M, 8), 256>>>(dx, bpr * 32, bpr, a);
    std::vector<int> idx(NS);
    for (int t = 0; t < M; ++t) { int got = 0; while (got < 4) { int e = rng() % E; bool d = false; for (int k = 0; k < got; ++k) d |= idx[t*4+k] == e; if (!d) idx[t*4+got++] = e; } }
    int * didx = dalloc<int>(NS); h2d(didx, idx);
    int * cnt = dalloc<int>(64); int * off = dalloc<int>(64); int * perm = dalloc<int>(NS);
    const int max_tiles = NS / PF_MOE_BN + 64;
    int4 * tiles = dalloc<int4>(max_tiles); int * ntiles = dalloc<int>(1);
    CK(cudaMemset(cnt, 0, 256));
    k_pf_moe_count<<<(NS + 255) / 256, 256>>>(didx, NS, cnt);
    k_pf_moe_plan<<<1, 32>>>(cnt, off, tiles, ntiles);
    k_pf_moe_scatter<<<64, 1024>>>(didx, NS, off, perm);
    float * y = dalloc<float>((size_t) NS * 2 * NI);
    PfAct r{ dalloc<int8_t>((size_t) NS * NI), dalloc<float2>((size_t) NS * NI / 32) };
    PfGemm g{};
    g.qs[0] = gq; g.d[0] = gd; g.qs[1] = g.qs[2] = uq; g.d[1] = g.d[2] = ud;
    g.seg_end[0] = NI; g.seg_end[1] = g.seg_end[2] = 2 * NI; g.N = 2 * NI; g.bpr = bpr;
    g.exp_stride = (long long) NI * bpr; g.aqs = a.qs; g.asc = a.sc;
    g.perm = perm; g.tiles = tiles; g.ntiles = ntiles; g.a_shift = 2; g.out = y; g.out_ld = 2 * NI;
    const double t1 = time_ms([&] { pf_gemm_moe(g, max_tiles, 0);
        k_pf_swiglu<<<dim3(NS, 6), 256>>>(y, 2 * NI, NI, NI / 32, r); });
    PfGemm f = g; f.N = NI; f.oqs = r.qs; f.osc = r.sc; f.okb = NI / 32;
    const double t2 = time_ms([&] { pf_gemm_swiglu(f, max_tiles, 0); });
    const double mac = (double) NS * 2 * NI * bpr * 32;
    printf("  moe gate|up+swiglu slots=%d: unfused %.3f ms (%.1f TMAC/s)  fused %.3f ms (%.1f TMAC/s)\n",
           NS, t1, mac / t1 / 1e9, t2, mac / t2 / 1e9);
}

static int bench_main(const char * what) {
    printf("=== prefill bench (%s) dense cfg=%s moe cfg=%s ===\n", what,
           getenv("LK_PF_CFG_DENSE") ? getenv("LK_PF_CFG_DENSE") : "default",
           getenv("LK_PF_CFG_MOE") ? getenv("LK_PF_CFG_MOE") : "default");
    if (!strcmp(what, "gemm") || !strcmp(what, "all")) {
        bench_dense(6144, 64, 4096);
        bench_dense(2048, 64, 4096);
        bench_dense(3072, 64, 4096);
        bench_dense(2048, 368, 4096);
        bench_dense(6144, 64, 512);
        bench_moe(3072, 64, 4096, 2);
        bench_moe(2048, 48, 4096, 0);
        bench_moe_swiglu(4096);
    }
    if (!strcmp(what, "attn") || !strcmp(what, "all")) {
        bench_attn(4096, 0);
        bench_attn(4096, 28672);
        bench_attn(4096, 126976);
        bench_attn8(4096, 0);
        bench_attn8(4096, 28672);
        bench_attn8(4096, 126976);
    }
    return 0;
}


// The SwiGLU epilogue against GEMM-to-f32 + k_pf_swiglu.  Both compute every
// gate/up value with the same block-order accumulation, so silu(g)*u and the
// quants must be IDENTICAL; only the 32-value sum inside the Q8_1 scale is
// reduced in a different order, so sc.y may differ in its last fp16 bit.
static void test_swiglu_fused(bool moe) {
    const int NI = 512, bpr = 64, M = moe ? 150 : 333, E = moe ? 4 : 1;
    char name[96]; snprintf(name, sizeof name, "gemm+swiglu fused (%s) == unfused", moe ? "moe" : "dense");
    const int NS = moe ? M * 4 : M;
    Q4 wg = rand_q4((size_t) E * NI, bpr), wu = rand_q4((size_t) E * NI, bpr);
    uint4 * gq = dalloc<uint4>(wg.qs.size()); h2d(gq, wg.qs); __half * gd = dalloc<__half>(wg.d.size()); h2d(gd, wg.d);
    uint4 * uq = dalloc<uint4>(wu.qs.size()); h2d(uq, wu.qs); __half * ud = dalloc<__half>(wu.d.size()); h2d(ud, wu.d);
    std::vector<float> xf = act_floats((size_t) M * bpr * 32);
    float * dx = dalloc<float>(xf.size()); h2d(dx, xf);
    PfAct a{ dalloc<int8_t>((size_t) M * bpr * 32), dalloc<float2>((size_t) M * bpr) };
    k_pf_quant_rows<<<dim3(M, (bpr + 7) / 8), 256>>>(dx, bpr * 32, bpr, a);
    std::vector<float> sc(M); for (auto & v : sc) v = urand(0.5f, 2.0f);
    float * dsc = dalloc<float>(M); h2d(dsc, sc);
    int * perm = nullptr; int4 * tiles = nullptr; int * ntiles = nullptr; int max_tiles = 0;
    if (moe) {
        std::vector<int> idx(NS);
        for (int t = 0; t < M; ++t) for (int k = 0; k < 4; ++k) idx[t * 4 + k] = (t + k) % E;
        int * didx = dalloc<int>(NS); h2d(didx, idx);
        int * cnt = dalloc<int>(64); int * off = dalloc<int>(64); perm = dalloc<int>(NS);
        max_tiles = NS / PF_MOE_BN + 64; tiles = dalloc<int4>(max_tiles); ntiles = dalloc<int>(1);
        CK(cudaMemset(cnt, 0, 256));
        k_pf_moe_count<<<(NS + 255) / 256, 256>>>(didx, NS, cnt);
        k_pf_moe_plan<<<1, 32>>>(cnt, off, tiles, ntiles);
        k_pf_moe_scatter<<<64, 1024>>>(didx, NS, off, perm);
    }
    // unfused
    float * y = dalloc<float>((size_t) NS * 2 * NI);
    PfAct r1{ dalloc<int8_t>((size_t) NS * NI), dalloc<float2>((size_t) NS * NI / 32) };
    PfAct r2{ dalloc<int8_t>((size_t) NS * NI), dalloc<float2>((size_t) NS * NI / 32) };
    PfGemm g{};
    g.qs[0] = gq; g.d[0] = gd; g.qs[1] = g.qs[2] = uq; g.d[1] = g.d[2] = ud;
    g.seg_end[0] = NI; g.seg_end[1] = g.seg_end[2] = 2 * NI; g.N = 2 * NI; g.bpr = bpr;
    g.exp_stride = moe ? (long long) NI * bpr : 0;
    g.aqs = a.qs; g.asc = a.sc; g.ntok = M; g.scale = dsc;
    g.perm = perm; g.tiles = tiles; g.ntiles = ntiles; g.a_shift = moe ? 2 : 0;
    g.out = y; g.out_ld = 2 * NI;
    if (moe) pf_gemm_moe(g, max_tiles, 0); else pf_gemm_dense(g, 0);
    k_pf_swiglu<<<dim3(NS, (NI / 32 + 7) / 8), 256>>>(y, 2 * NI, NI, NI / 32, r1);
    PfGemm f = g; f.N = NI; f.oqs = r2.qs; f.osc = r2.sc; f.okb = NI / 32;
    pf_gemm_swiglu(f, max_tiles, 0);
    CK(cudaDeviceSynchronize());
    std::vector<int8_t> q1 = d2h(r1.qs, (size_t) NS * NI), q2 = d2h(r2.qs, (size_t) NS * NI);
    std::vector<float2> s1 = d2h(r1.sc, (size_t) NS * NI / 32), s2 = d2h(r2.sc, (size_t) NS * NI / 32);
    // The stored field is 8*fp16(sum); a reassociated fp32 sum can move it by
    // a few fp16 ulps, and near zero an ulp is the subnormal step 2^-24.
    size_t qdiff = 0, ddiff = 0, sdiff = 0; double smax = 0.0;
    for (size_t i = 0; i < q1.size(); ++i) qdiff += q1[i] != q2[i];
    for (size_t i = 0; i < s1.size(); ++i) {
        ddiff += s1[i].x != s2[i].x;
        if (s1[i].y != s2[i].y) {
            ++sdiff;
            const double ulp = 8.0 * std::max(std::ldexp(fabs(s1[i].y) / 8.0, -10), std::ldexp(1.0, -24));
            smax = std::max(smax, fabs((double) s1[i].y - s2[i].y) / ulp);
        }
    }
    const bool ok = qdiff == 0 && ddiff == 0 && smax <= 4.0;
    printf("  %-44s %s  quants differ %zu/%zu, d differ %zu, sum differ %zu (max %.1f fp16 ulp)\n", name,
           ok ? "PASS" : "FAIL", qdiff, q1.size(), ddiff, sdiff, smax);
    if (!ok) ++g_fail;
}


// combine+norm must be byte-identical to combine followed by the 4y norm.
static void test_combine_norm() {
    const int M = 9;
    std::vector<float> x = act_floats((size_t) M * 2048), d = act_floats((size_t) M * 4 * 2048), w(M * 4), nw(2048);
    for (auto & v : w) v = urand(0.0f, 0.5f);
    for (auto & v : nw) v = urand(0.2f, 1.8f);
    float * x1 = dalloc<float>(x.size()); h2d(x1, x);
    float * x2 = dalloc<float>(x.size()); h2d(x2, x);
    float * dd = dalloc<float>(d.size()); h2d(dd, d);
    float * dw = dalloc<float>(w.size()); h2d(dw, w);
    float * dn = dalloc<float>(2048); h2d(dn, nw);
    PfAct a1{ dalloc<int8_t>((size_t) M * 2048), dalloc<float2>((size_t) M * 64) };
    PfAct a2{ dalloc<int8_t>((size_t) M * 2048), dalloc<float2>((size_t) M * 64) };
    float * i1 = dalloc<float>(M); float * i2 = dalloc<float>(M);
    k_pf_moe_combine<<<dim3(M, 8), 256>>>(x1, dd, dw);
    k_pf_norm<<<M, 256>>>(x1, dn, 0, a1, i1, nullptr);
    k_pf_moe_combine_norm<<<M, 256>>>(x2, dd, dw, dn, a2, i2);
    CK(cudaDeviceSynchronize());
    const bool ok = d2h(x1, x.size()) == d2h(x2, x.size())
                 && d2h(a1.qs, (size_t) M * 2048) == d2h(a2.qs, (size_t) M * 2048)
                 && !memcmp(d2h(a1.sc, (size_t) M * 64).data(), d2h(a2.sc, (size_t) M * 64).data(), (size_t) M * 64 * 8)
                 && d2h(i1, M) == d2h(i2, M);
    report("combine+norm == combine, norm (bytes)", ok, "");
}

int main(int argc, char ** argv) {
    if (argc > 1) return bench_main(argv[1]);
    printf("=== batched prefill kernels ===\n");
    test_norms();
    test_conv();
    test_qknr();
    test_router();
    test_combine_norm();
    test_gemm_dense(256, 64, 200, false, false, false);    // BN=64 path, ragged tokens
    test_gemm_dense(512, 64, 700, false, true, false);     // BN=128 path, scaled
    test_gemm_dense(384, 64, 333, true, true, true);       // residual + 3 segments
    test_gemm_dense(256, 368, 130, true, false, false);    // K=11776 (dense down)
    test_gemm_dense(256, 48, 600, false, false, false);    // K=1536
    test_gemm_moe(8, 256, 64, 300, 2);                     // gate|up shape
    test_gemm_moe(8, 256, 48, 300, 0);                     // down shape
    test_swiglu_fused(false);
    test_swiglu_fused(true);
    test_attn(64, 0);
    test_attn(100, 150);
    test_attn(37, 1000);
    // Lossless int8 grid: the int8 path must equal the fp16 kernel.
    test_attn8(64, 0, -1.0f, false, 0, 0.0, 1.01);
    test_attn8(100, 150, -1.0f, false, 0, 0.0, 1.01);
    // Every tile exact (near covers the context), smoothing on: the q.c shift
    // must cancel, reproducing the fp16 kernel.
    test_attn8(100, 150, 3.0f, true, 1 << 20, 0.0, 1.01);
    test_attn8(37, 1000, 3.0f, true, 1 << 20, 0.0, 1.01);
    // Mixed int8 / exact tiles with smoothing: the shift must be consistent
    // across the two kinds of tile (an inconsistency is an O(1) error).
    test_attn8(37, 1000, 3.0f, true, 256, 2e-1);
    // Quantisation error on random data (informational; the gate for this is
    // whole-corpus NLL on the real model).
    test_attn8(100, 150, 0.0f, false, 0, -1.0);
    test_attn8(37, 1000, 3.0f, false, 0, -1.0);
    test_attn8(37, 1000, 3.0f, true, 0, -1.0);
    printf("%s: %d failure(s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail);
    return g_fail ? 1 : 0;
}
