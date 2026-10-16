// bench_shapes.cu - the tollbooth floor.
//
// The engine's step at ctx 272 is 2.269 ms for 1466.75 MB of weights.  Reading
// that much at the measured asymptote (bandwidth_probe: 956 GB/s) is 1.53 ms,
// so ~0.74 ms is "something else".  Two models are on the table for it:
//
//   MODEL A (per-launch overhead): the 343-node graph pays a fixed toll at
//     every node -- dispatch plus pipeline drain -- and batching M tokens into
//     one pass would pay that toll once instead of M times.  If true, batch-4
//     is worth ~1.84x rather than the byte-floor 1.51x, and speculative
//     verification's cost ratio drops from 2.42x to ~2.01x.
//
//   MODEL B (latency-bound execution): most of the 0.74 ms is real execution in
//     the small kernels, which does not vanish when you batch -- it multiplies.
//     Then batching is worth only the byte reduction, and speculative stays
//     rejected.
//
// These predict very different numbers, and the experiment that separates them
// is cheap: run a graph containing one EMPTY kernel per node, with the engine's
// EXACT (grid, block) for all 343 nodes, and time the replay.  Empty bodies
// preserve the grids, the block sizes and the wave structure, so what remains
// is launch plus drain.
//
// This project has already measured what deleting a node is worth: T1 deleted
// 38 of them for +0.67% (0.40 us/node), T4r deleted 38 for -0.17%.  If the toll
// were 2.07 us/node (0.712 ms / 343), T1 would have gained 3.5%.  It did not.
// So MODEL B is already favoured by 5x -- but "favoured" is not "measured", and
// the whole Phase 5 plan rests on the answer.
//
// Mode 1 additionally gives every node its real byte count, so the floor is
// computed for (shapes + bytes) with no dependencies and no computation at all.
// That is the best a perfect engine with these shapes could possibly do.
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

__global__ void k_empty() {}

__global__ void k_read_n(const uint4 * __restrict__ p, long n4, float * __restrict__ out) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = (long)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (; i < n4; i += stride) {
        const uint4 v = __ldcs(p + i);
        acc += (float)(v.x ^ v.y ^ v.z ^ v.w);
    }
    if ((threadIdx.x & 31) == 0) out[blockIdx.x] = acc;
}

struct Node { int gx, gy, b; double bytes; };

// The engine's 343 nodes at ctx 272, in capture order.  Counts must come out as
//   1 embed + 1 norm_attn + 30*3 ShortConv + 10*5 attention + 2*4 dense
//   + 38*5 MoE + 3 tail = 343
static std::vector<Node> engine_shapes() {
    std::vector<Node> v;
    auto add = [&](int gx, int gy, int b, double bytes) { v.push_back({gx, gy, b, bytes}); };

    add(8, 1, 256, 1680);                       // embed_q6_K
    add(1, 1, 256, 16384);                      // norm_attn, layer 0 only

    for (int il = 0; il < 40; ++il) {
        const bool attn = (il % 4 == 2);
        if (attn) {
            add(384,  1,  256,  3538944);       // attn_qkv
            add(40,   1,   64,        0);       // qk_norm_rope
            add(8,    8,  256,     4096);       // attn_split   (ctx 272 -> 8 slices)
            add(32,   1,   64,        0);       // attn_combine
            add(64,   1, 1024,  2359296);       // attn_out (4y producer)
        } else {
            add(768,  1,  256,  7077888);       // shortconv_in
            add(8,    1,  256,    24576);       // shortconv_conv
            add(64,   1, 1024,  2359296);       // shortconv_out (4y producer)
        }
        if (il < 2) {
            add(1472, 1,  256, 13565952);       // dense_gate
            add(1472, 1,  256, 13565952);       // dense_up
            add(64,   1,  256,        0);       // dense_swiglu
            add(64,   1, 1024, 13565952);       // dense_down (4y producer)
        } else {
            add(64,   1,   32,   524288);       // router_dot
            add(1,    1,   64,        0);       // router_top4
            add(192,  8,  256, 14155776);       // moe_gate_up
            add(48,   1,  256,        0);       // moe_swiglu
            add(64,   1, 1024,  7077888);       // moe_down (T1, includes accum)
        }
    }
    add(16000, 1, 256, 215040000);              // lm_head
    add(125,   1, 256,    512000);              // argmax_part
    add(1,     1, 256,         0);              // argmax_final
    return v;
}

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s  SMs=%d  maxThreadsPerSM=%d\n\n",
           prop.name, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);

    std::vector<Node> nodes = engine_shapes();
    double total = 0; for (const Node & n : nodes) total += n.bytes;
    printf("nodes reconstructed: %zu   total bytes: %.2f MB\n", nodes.size(), total / 1048576.0);
    if (nodes.size() != 343) printf("*** SHAPE LIST IS WRONG, expected 343 ***\n");

    const size_t BUFSZ = 768ull << 20;
    uint4 * buf = nullptr; float * out = nullptr;
    CK(cudaMalloc(&buf, BUFSZ));
    CK(cudaMemset(buf, 0x5a, BUFSZ));
    CK(cudaMalloc(&out, 1 << 22));
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

    // =================================================================
    printf("\n== MODE 0: all nodes EMPTY, exact engine shapes ==\n");
    printf("   (this is the tollbooth floor: launch + dispatch + drain only)\n\n");
    {
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        for (const Node & n : nodes) k_empty<<<dim3(n.gx, n.gy), n.b, 0, st>>>();
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        size_t nn = 0; cudaGraphGetNodes(g, nullptr, &nn);
        for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(ge, st));
        CK(cudaStreamSynchronize(st));
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a, st));
        for (int i = 0; i < 200; ++i) CK(cudaGraphLaunch(ge, st));
        CK(cudaEventRecord(b, st));
        CK(cudaStreamSynchronize(st));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        const double per = ms * 1000.0 / 200.0;
        printf("   %zu nodes, graph replay : %8.3f ms/token   %.3f us/node\n",
               nn, per / 1000.0, per / (double)nn);
        printf("   as a fraction of the real 2.269 ms step: %.1f%%\n",
               100.0 * (per / 1000.0) / 2.269);
        printf("   per-node cost the \"2.07 us toll\" model needs: 2.07 us\n");
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    }

    // =================================================================
    printf("\n== MODE 1: every node reads its REAL byte count, no dependencies ==\n");
    printf("   (shapes + bytes, zero computation -- the best these shapes allow)\n\n");
    {
        cudaGraph_t g; cudaGraphExec_t ge;
        double off = 0;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        for (const Node & n : nodes) {
            if (n.bytes < 16.0) { k_empty<<<dim3(n.gx, n.gy), n.b, 0, st>>>(); continue; }
            const long n4 = (long)(n.bytes / 16.0);
            const size_t start = (size_t)((long long)off % (long long)(BUFSZ - n.bytes));
            off += n.bytes;
            k_read_n<<<dim3(n.gx, n.gy), n.b, 0, st>>>(buf + start / 16, n4, out);
        }
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, st));
        CK(cudaStreamSynchronize(st));
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a, st));
        for (int i = 0; i < 50; ++i) CK(cudaGraphLaunch(ge, st));
        CK(cudaEventRecord(b, st));
        CK(cudaStreamSynchronize(st));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        const double per = ms * 1000.0 / 50.0;
        const double gbs = total / (per * 1e3);
        printf("   %.2f MB in %8.3f ms  = %.1f GB/s  (%.1f%% of the 956 GB/s probe)\n",
               total / 1048576.0, per / 1000.0, gbs, 100.0 * gbs / 956.0);
        printf("   real engine at the same traffic: 2.269 ms, 646 GB/s\n");
        printf("   --> this is the floor these shapes impose on a PERFECT engine\n");
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    }
    printf("\n");
    return 0;
}
