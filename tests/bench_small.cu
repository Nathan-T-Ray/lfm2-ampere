// bench_small.cu - true marginal cost of the engine's small kernels.
//
// The per-kernel profiler brackets each launch in its own event pair, so every
// reading carries ~2.1 us of instrument.  Subtracting a floor measured on an
// EMPTY kernel in a clean loop is an approximation, and the small kernels are
// exactly where that approximation decides the answer.  Here they are launched
// back to back with one event pair around the whole batch, which gives the
// marginal cost -- launch plus execution, with no per-launch instrument.
//
// Both launch paths are measured because they differ by 2x: a direct launch
// costs ~1.2 us and a CUDA-graph node ~0.6 us (tests/bench_launch.cu).  The
// engine replays a graph, so the graph number is the one that matters.
#include "kernels.cuh"
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

static const int N = 2000;

int main() {
    float *x, *w, *y32, *gate, *bias, *logits, *wout, *dg;
    __half *y16;
    block_q8_1 *yq8;
    int *idx;
    CK(cudaMalloc(&x, LK_HIDDEN * 4));
    CK(cudaMalloc(&w, LK_HIDDEN * 4));
    CK(cudaMalloc(&y32, LK_HIDDEN * 4));
    CK(cudaMalloc(&y16, LK_HIDDEN * 2));
    CK(cudaMalloc(&yq8, 64 * sizeof(block_q8_1)));
    CK(cudaMalloc(&gate, 4 * LK_DENSE_INTER));
    CK(cudaMalloc(&bias, LK_N_EXPERTS * 4));
    CK(cudaMalloc(&logits, LK_N_EXPERTS * 4));
    CK(cudaMalloc(&wout, LK_TOP_K * 4));
    CK(cudaMalloc(&dg, LK_HIDDEN * 4));
    CK(cudaMalloc(&idx, LK_TOP_K * 4));
    CK(cudaMemset(x, 0x3e, LK_HIDDEN * 4));
    CK(cudaMemset(w, 0x3e, LK_HIDDEN * 4));

    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

    struct Case { const char * name; void (*run)(cudaStream_t, float*, float*, float*, __half*,
                                                  block_q8_1*, float*, float*, float*, int*, float*); };
    // lambdas cannot be passed as function pointers portably here; use a switch instead
    const char * names[3] = {"rmsnorm_q8_1", "router_top4", "swiglu(dense)"};

    for (int c = 0; c < 3; ++c) {
        // ---- direct, batched
        for (int i = 0; i < 200; ++i) {
            if (c == 0) k_rmsnorm_q8_1<<<1, 256, LK_HIDDEN * 4, st>>>(x, w, LK_HIDDEN, LK_EPS, y32, y16, yq8);
            if (c == 1) k_router_top4<<<1, LK_N_EXPERTS, 0, st>>>(logits, bias, idx, wout);
            if (c == 2) k_swiglu<<<64, 256, 0, st>>>(gate, gate + LK_DENSE_INTER, LK_DENSE_INTER, nullptr, yq8);
        }
        CK(cudaStreamSynchronize(st));
        CK(cudaEventRecord(a, st));
        for (int i = 0; i < N; ++i) {
            if (c == 0) k_rmsnorm_q8_1<<<1, 256, LK_HIDDEN * 4, st>>>(x, w, LK_HIDDEN, LK_EPS, y32, y16, yq8);
            if (c == 1) k_router_top4<<<1, LK_N_EXPERTS, 0, st>>>(logits, bias, idx, wout);
            if (c == 2) k_swiglu<<<64, 256, 0, st>>>(gate, gate + LK_DENSE_INTER, LK_DENSE_INTER, nullptr, yq8);
        }
        CK(cudaEventRecord(b, st));
        CK(cudaStreamSynchronize(st));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        const double direct = ms * 1000.0 / N;

        // ---- graph replay, which is what the engine actually pays
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < N; ++i) {
            if (c == 0) k_rmsnorm_q8_1<<<1, 256, LK_HIDDEN * 4, st>>>(x, w, LK_HIDDEN, LK_EPS, y32, y16, yq8);
            if (c == 1) k_router_top4<<<1, LK_N_EXPERTS, 0, st>>>(logits, bias, idx, wout);
            if (c == 2) k_swiglu<<<64, 256, 0, st>>>(gate, gate + LK_DENSE_INTER, LK_DENSE_INTER, nullptr, yq8);
        }
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, st));
        CK(cudaStreamSynchronize(st));
        CK(cudaEventRecord(a, st));
        for (int i = 0; i < 20; ++i) CK(cudaGraphLaunch(ge, st));
        CK(cudaEventRecord(b, st));
        CK(cudaStreamSynchronize(st));
        CK(cudaEventElapsedTime(&ms, a, b));
        const double graph = ms * 1000.0 / (20.0 * N);

        printf("%-16s direct %6.3f us   graph %6.3f us   (x81 per token: %.1f / %.1f us)\n",
               names[c], direct, graph, direct * 81, graph * 81);
    }
    printf("\ngraph replay is what the engine pays (471 nodes x 0.606 us of dispatch).\n");
    return 0;
}
