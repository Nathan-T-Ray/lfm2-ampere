// bench_attn.cu - isolate k_attn_split and measure its DRAM bandwidth.
//
// WHY THIS EXISTS
//
// The long-context fall-off was diagnosed from a curve fit, not from a
// counter: decode throughput at 131040 implies that the ten attention layers
// move their KV cache at roughly 570 GB/s while every STREAMING kernel in the
// same engine moves weights at 942 GB/s.  That gap was inferred from
// end-to-end timing, and end-to-end timing cannot separate "attention is slow"
// from "attention is slow because of the K load pattern" from "something else
// near attention is slow".
//
// An end-to-end A/B at 131040 costs 627 s of prefill PER SIDE and confounds
// attention with 40 layers of GEMV, MoE routing and ShortConv.  This harness
// times the one kernel under test and nothing else, at a size where the KV
// cache (seq * 2048 B) is far larger than the 6 MB L2, so the number it
// reports is real DRAM traffic and not cache reuse.
//
// WHAT IS MEASURED
//
//   k_attn_split<<<dim3(8, SPLIT), 256>>>  +  k_attn_combine<<<32, 64>>>
//
// which is exactly the pair the engine launches per attention layer.  Bytes
// counted are the ones the kernel is obligated to read:
//
//   K: seq tokens * 8 kv_heads * 64 dims * 2 B  = seq * 1024 B
//   V: seq tokens * 8 kv_heads * 64 dims * 2 B  = seq * 1024 B
//   ------------------------------------------------------------
//                                                  seq * 2048 B
//
// The partial buffer that k_attn_combine reads is 32*SPLIT*66*4 B = 1.1 MB at
// SPLIT=128, under half a percent, and is deliberately NOT counted so the
// figure is a lower bound on true traffic rather than an estimate.
//
// K is read once per (kv_head, slice) and shared by the four query heads on
// that kv_head.  That is phase 4v's whole purpose, and it is why the byte
// count above has 8 kv_heads in it rather than 32 query heads.
//
// USAGE
//
//   bench_attn <seq> [iters] [reps]
//
// It prints a median over reps, because a single timing on this machine has
// been observed to drift by more than the effect being measured.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/kernels.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static uint64_t g_seed = 0x243f6a8885a308d3ull;
static float frand() {
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17;
    return (float)((double)((g_seed >> 11) & ((1ull << 53) - 1)) / (double)(1ull << 53));
}

int main(int argc, char ** argv) {
    const int seq   = argc > 1 ? atoi(argv[1]) : 131040;
    const int iters = argc > 2 ? atoi(argv[2]) : 50;
    const int reps  = argc > 3 ? atoi(argv[3]) : 3;

    if (seq < 1 || seq > LK_ATTN_MAX_SEQ) {
        printf("seq %d out of range 1..%d\n", seq, LK_ATTN_MAX_SEQ);
        return 1;
    }

    // K and V are the entire memory cost of this kernel, so they are the only
    // large allocations.  Random data rather than zeros: an all-zero K makes
    // every score identical and every softmax weight uniform, which is a
    // degenerate input the engine never actually sees.
    const size_t kvn = (size_t) seq * LK_N_KV_HEADS * LK_HEAD_DIM;
    std::vector<__half> kh(kvn), vh(kvn);
    for (size_t i = 0; i < kvn; ++i) {
        kh[i] = __float2half(frand() * 2.0f - 1.0f);
        vh[i] = __float2half(frand() * 2.0f - 1.0f);
    }
    std::vector<float> q(LK_N_HEADS * LK_HEAD_DIM);
    for (size_t i = 0; i < q.size(); ++i) q[i] = frand() * 2.0f - 1.0f;

    float *d_q, *d_part, *d_out;
    __half *d_k, *d_v;
    int *d_pos;
    CK(cudaMalloc(&d_q,    q.size() * 4));
    CK(cudaMalloc(&d_k,    kvn * 2));
    CK(cudaMalloc(&d_v,    kvn * 2));
    CK(cudaMalloc(&d_part, (size_t) LK_N_HEADS * LK_ATTN_SPLIT * LK_ATTN_PARTW * 4));
    CK(cudaMalloc(&d_out,  (size_t) LK_N_HEADS * LK_HEAD_DIM * 4));
    CK(cudaMalloc(&d_pos,  4));
    CK(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_k, kh.data(), kvn * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_v, vh.data(), kvn * 2, cudaMemcpyHostToDevice));
    // d_pos is the index of the token JUST APPENDED, so the kernel computes
    // seq_len = *d_pos + 1.  Writing `seq` here makes it read seq+1 tokens, one
    // past the end of the buffer: at 131040 that landed in cudaMalloc padding
    // and went unnoticed, at 32768 it crossed a page boundary and took an
    // illegal memory access, and at 8192 it read garbage into 4 of 2048 output
    // dims.  The kernel itself clamps t1 to seq_len and is not at fault; this
    // harness was, and it reported the three sizes in exactly that confusing
    // order until the off-by-one was found.
    const int last = seq - 1;
    CK(cudaMemcpy(d_pos, &last, 4, cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    // One untimed call: first touch of a 268 MB allocation is page mapping,
    // and it is not part of the steady state being reported.
    k_attn_split<<<dim3(LK_N_KV_HEADS, LK_ATTN_SPLIT), 256>>>(d_q, d_k, d_v, d_pos, d_part);
    k_attn_combine<<<LK_N_HEADS, LK_HEAD_DIM>>>(d_part, d_pos, d_out, nullptr);
    CK(cudaDeviceSynchronize());

    // Sanity: a wrong kernel that produces NaN would otherwise be timed and
    // reported as the fastest result in the table.
    std::vector<float> out(LK_N_HEADS * LK_HEAD_DIM);
    CK(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost));
    double mx = 0.0; int bad = 0;
    for (float f : out) { if (!std::isfinite(f)) ++bad; mx = std::max(mx, (double) fabsf(f)); }
    if (bad) { printf("FAIL: %d/%zu outputs are not finite\n", bad, out.size()); return 1; }
    if (mx == 0.0) { printf("FAIL: all outputs are exactly zero\n"); return 1; }

    std::vector<double> ms(reps);
    for (int r = 0; r < reps; ++r) {
        CK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) {
            k_attn_split<<<dim3(LK_N_KV_HEADS, LK_ATTN_SPLIT), 256>>>(d_q, d_k, d_v, d_pos, d_part);
            k_attn_combine<<<LK_N_HEADS, LK_HEAD_DIM>>>(d_part, d_pos, d_out, nullptr);
        }
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float t = 0.0f;
        CK(cudaEventElapsedTime(&t, e0, e1));
        ms[r] = (double) t / iters;
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[reps / 2];

    // seq * 2048 bytes is what the kernel must read; L2 is 6 MB on GA102.
    const double bytes   = (double) seq * 2048.0;
    const double gbs     = bytes / (med * 1e-3) / 1e9;
    const double mb      = bytes / 1e6;
    // lk_attn_slices is __device__ only, so the same formula is repeated here
    // for reporting.  It is not used to drive the launch -- the kernel derives
    // its own slice count from d_pos, and the grid is always LK_ATTN_SPLIT.
    int slices = (seq + LK_ATTN_TARGET_N - 1) / LK_ATTN_TARGET_N;
    if (slices < LK_ATTN_SPLIT_MIN) slices = LK_ATTN_SPLIT_MIN;
    if (slices > LK_ATTN_SPLIT)     slices = LK_ATTN_SPLIT;

    printf("seq=%-7d slices=%-4d iters=%-4d mu_s=%-9.2f  MB=%-8.1f  GB_s=%7.1f",
           seq, slices, iters, med * 1000.0, mb, gbs);
    printf("   | x10 layers: %6.3f ms/token -> %6.1f tok/s ceiling\n",
           med * 10.0, 1000.0 / (med * 10.0));
    if (reps > 1) printf("      reps ms: %.4f %.4f %.4f\n", ms[0], ms[reps / 2], ms[reps - 1]);

    CK(cudaFree(d_q)); CK(cudaFree(d_k)); CK(cudaFree(d_v));
    CK(cudaFree(d_part)); CK(cudaFree(d_out)); CK(cudaFree(d_pos));
    return 0;
}
