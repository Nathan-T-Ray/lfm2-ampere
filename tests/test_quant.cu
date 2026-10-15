// test_quant.cu - Gate G2 (core): verify the dot products against reference
// values computed from the REAL planar pack data.
//
//   |got - expected_exact|  isolates formula error      (tight bound)
//   |got - expected_fp32 |  includes Q8_1 rounding      (sanity bound)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "../src/quant.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

enum { KIND_Q4_0 = 0, KIND_Q6_K = 1 };

struct Case {
    uint32_t kind, nblocks, nplanes;
    std::vector<std::vector<uint8_t>> planes;
    std::vector<float> x;
    float expected_exact, expected_fp32;
    // device
    void *dp[4] = {nullptr, nullptr, nullptr, nullptr};
    float *dx = nullptr;
    block_q8_1 *dxq = nullptr;
    float *dout = nullptr;
    float got = 0.0f;
};

// ---- one warp per 32-element activation block -----------------------------
__global__ void quantize_blocks(const float * __restrict__ x, block_q8_1 * __restrict__ y, int nblocks) {
    const int b = blockIdx.x;
    const int lane = threadIdx.x;
    if (b >= nblocks) return;
    quantize_q8_1_warp(x[b * 32 + lane], y + b, lane);
}

// ---- one thread per case --------------------------------------------------
__global__ void dot_cases(const Case * __restrict__ cases, int ncases) {
    const int c = blockIdx.x;
    if (c >= ncases) return;
    const Case &k = cases[c];
    float acc = 0.0f;
    float accf = 0.0f;   // raw-fp32 reference: dequantise weights, use ORIGINAL x
    if (k.kind == KIND_Q4_0) {
        const uint4 * qs = (const uint4 *)k.dp[0];
        const __half * d  = (const __half *)k.dp[1];
        for (uint32_t b = 0; b < k.nblocks; ++b) {
            acc += dot_q4_0_q8_1(qs, d, (int)b, k.dxq + b);
            const uint4 q = qs[b];
            const uint8_t * qb = (const uint8_t *)&q;
            const float d4 = __half2float(d[b]);
            const float * xr = k.dx + b * 32;
            for (int j = 0; j < 16; ++j) {
                accf += d4 * (float)((qb[j] & 0x0F) - 8) * xr[j];
                accf += d4 * (float)((qb[j] >> 4) - 8) * xr[j + 16];
            }
        }
    } else {
        const uint8_t * ql = (const uint8_t *)k.dp[0];
        const uint8_t * qh = (const uint8_t *)k.dp[1];
        const int8_t  * sc = (const int8_t  *)k.dp[2];
        const __half  * d  = (const __half  *)k.dp[3];
        for (uint32_t b = 0; b < k.nblocks; ++b)
            acc += dot_q6_K_q8_1(ql, qh, sc, d, (int)b, k.dxq + b * 8);
    }
    k.dout[0] = acc;
    k.dout[1] = accf;
}

static void rd(FILE *f, void *buf, size_t n) {
    if (fread(buf, 1, n, f) != n) { printf("short fixture read\n"); exit(1); }
}
template <class T> static T rdv(FILE *f) { T v; rd(f, &v, sizeof(T)); return v; }

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "ref/quant_fixture.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    char magic[4]; rd(f, magic, 4);
    if (memcmp(magic, "LKQF", 4)) { printf("bad fixture magic\n"); return 1; }
    uint32_t ncases = rdv<uint32_t>(f);

    std::vector<Case> host(ncases);
    int total_q8_blocks = 0;
    for (uint32_t i = 0; i < ncases; ++i) {
        Case &c = host[i];
        c.kind = rdv<uint32_t>(f);
        c.nblocks = rdv<uint32_t>(f);
        c.nplanes = rdv<uint32_t>(f);
        (void)rdv<float>(f);                              // reserved
        for (uint32_t p = 0; p < c.nplanes; ++p) {
            uint32_t nb = rdv<uint32_t>(f);
            c.planes.emplace_back(nb);
            rd(f, c.planes.back().data(), nb);
        }
        uint32_t nx = rdv<uint32_t>(f);
        c.x.resize(nx);
        rd(f, c.x.data(), nx * sizeof(float));
        c.expected_exact = rdv<float>(f);
        c.expected_fp32  = rdv<float>(f);
        const int per_sb = (c.kind == KIND_Q6_K) ? 8 : 1;
        total_q8_blocks += (int)c.nblocks * per_sb;
    }
    fclose(f);

    // device buffers
    block_q8_1 *dxq_all = nullptr;
    float *dx_all = nullptr;
    CK(cudaMalloc(&dxq_all, sizeof(block_q8_1) * total_q8_blocks));
    int total_x = 0;
    for (auto &c : host) total_x += (int)c.x.size();
    CK(cudaMalloc(&dx_all, sizeof(float) * total_x));

    int xoff = 0, qoff = 0;
    std::vector<Case> dev(ncases);
    for (uint32_t i = 0; i < ncases; ++i) {
        Case &c = host[i];
        for (uint32_t p = 0; p < c.nplanes; ++p) {
            CK(cudaMalloc(&c.dp[p], c.planes[p].size()));
            CK(cudaMemcpy(c.dp[p], c.planes[p].data(), c.planes[p].size(), cudaMemcpyHostToDevice));
        }
        CK(cudaMemcpy(dx_all + xoff, c.x.data(), c.x.size() * sizeof(float), cudaMemcpyHostToDevice));
        c.dx = dx_all + xoff;
        c.dxq = dxq_all + qoff;
        CK(cudaMalloc(&c.dout, 2*sizeof(float)));

        const int per_sb = (c.kind == KIND_Q6_K) ? 8 : 1;
        const int nq = (int)c.nblocks * per_sb;
        quantize_blocks<<<nq, 32>>>(c.dx, c.dxq, nq);
        CK(cudaGetLastError());

        dev[i] = c;
        xoff += (int)c.x.size();
        qoff += nq;
    }
    // upload the (pointer-carrying) case table
    Case *dcases = nullptr;
    CK(cudaMalloc(&dcases, sizeof(Case) * ncases));
    CK(cudaMemcpy(dcases, dev.data(), sizeof(Case) * ncases, cudaMemcpyHostToDevice));
    dot_cases<<<ncases, 1>>>(dcases, ncases);
    CK(cudaDeviceSynchronize());

    int fails = 0;
    printf("%-6s %8s %14s %14s %12s %12s\n",
           "kind", "nblocks", "got", "expected_exact", "rel_exact", "rel_fp32");
    for (uint32_t i = 0; i < ncases; ++i) {
        Case &c = host[i];
        CK(cudaMemcpy(&c.got, c.dout, sizeof(float), cudaMemcpyDeviceToHost));
        float gotf = 0.0f;
        CK(cudaMemcpy(&gotf, c.dout + 1, sizeof(float), cudaMemcpyDeviceToHost));
        const float re = fabsf(c.got - c.expected_exact) / (fabsf(c.expected_exact) + 1e-9f);
        const float rf = fabsf(c.got - c.expected_fp32)  / (fabsf(c.expected_fp32)  + 1e-9f);
        // rel_exact: agreement with ggml's exact integer convention.  The
        //   residual (~1e-3) is float32 sequential accumulation in the kernel
        //   vs float64 accumulation in the numpy reference -- NOT formula error.
        // rel_fp32: agreement with the true fp32 dot, whose gap is dominated by
        //   Q8_1 activation rounding (~1-2%).
        const bool ok = (re < 5e-3f) && (rf < 5e-2f);
        if (!ok) ++fails;
        if (c.kind == KIND_Q4_0)
            printf("%-6s %8u %14.6f %14.6f %11.2e %11.2e  %s  [indep fp32 dot=%+.6f, exp=%+.6f]\n",
                   "Q4_0", c.nblocks, c.got, c.expected_exact, re, rf, ok ? "OK" : "FAIL",
                   gotf, c.expected_fp32);
        else
            printf("%-6s %8u %14.6f %14.6f %11.2e %11.2e  %s\n",
                   "Q6_K", c.nblocks, c.got, c.expected_exact, re, rf, ok ? "OK" : "FAIL");
    }
    printf("\nGATE G2 (dots): %s (%d/%u cases)\n",
           fails ? "FAIL" : "PASS - formulas match the exact reference",
           (int)ncases - fails, ncases);
    return fails ? 1 : 0;
}
