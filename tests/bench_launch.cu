// bench_launch.cu - what does one kernel launch actually cost on this GPU?
//
// The per-kernel profile shows 550 launches per token, ~200 of which move no
// appreciable data and still measure 3.1-3.6 us each.  Before believing that,
// establish the floor: an empty kernel, measured three ways, because the three
// answers mean completely different things for the optimisation plan.
//   (1) launches back to back, timed as a batch      -> marginal cost
//   (2) each launch bracketed by its own events      -> what the profiler sees
//   (3) the same launches replayed from a CUDA graph -> what the engine pays
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

__global__ void k_nop() {}
__global__ void k_tiny(float * o) { if (threadIdx.x == 0) o[blockIdx.x] = 1.0f; }

static const int N = 550;

int main() {
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    float * d_o; CK(cudaMalloc(&d_o, N * 4));

    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

    // ---- (1) batch: 20 rounds of N launches, one event pair around each round
    for (int i = 0; i < 3; ++i) for (int j = 0; j < N; ++j) k_nop<<<1, 256, 0, st>>>();
    CK(cudaStreamSynchronize(st));
    CK(cudaEventRecord(a, st));
    for (int r = 0; r < 20; ++r) for (int j = 0; j < N; ++j) k_nop<<<1, 256, 0, st>>>();
    CK(cudaEventRecord(b, st));
    CK(cudaStreamSynchronize(st));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    const double batch_us = ms * 1000.0 / (20.0 * N);
    printf("(1) empty kernel, batched launches   : %6.3f us/launch\n", batch_us);

    // ---- (1b) same total work but a real (tiny) kernel writing memory
    CK(cudaEventRecord(a, st));
    for (int r = 0; r < 20; ++r) for (int j = 0; j < N; ++j) k_tiny<<<1, 256, 0, st>>>(d_o);
    CK(cudaEventRecord(b, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaEventElapsedTime(&ms, a, b));
    printf("(1b) tiny store kernel, batched      : %6.3f us/launch\n", ms * 1000.0 / (20.0 * N));

    // ---- (2) bracketed: exactly what lk_prof_report measures
    std::vector<cudaEvent_t> ea(N), eb(N);
    for (int j = 0; j < N; ++j) { CK(cudaEventCreate(&ea[j])); CK(cudaEventCreate(&eb[j])); }
    CK(cudaEventRecord(a, st));
    for (int j = 0; j < N; ++j) {
        CK(cudaEventRecord(ea[j], st));
        k_nop<<<1, 256, 0, st>>>();
        CK(cudaEventRecord(eb[j], st));
    }
    CK(cudaEventRecord(b, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaEventElapsedTime(&ms, a, b));
    double sum = 0;
    for (int j = 0; j < N; ++j) { float d; CK(cudaEventElapsedTime(&d, ea[j], eb[j])); sum += d; }
    printf("(2) empty kernel, event-bracketed    : %6.3f us/launch (sum of brackets)\n", sum * 1000.0 / N);
    printf("    wall clock for the same %d launches: %6.3f us/launch\n", N, ms * 1000.0 / N);

    // ---- (3) graph replay: what the engine actually pays
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    for (int j = 0; j < N; ++j) k_nop<<<1, 256, 0, st>>>();
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaEventRecord(a, st));
    for (int r = 0; r < 20; ++r) CK(cudaGraphLaunch(ge, st));
    CK(cudaEventRecord(b, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaEventElapsedTime(&ms, a, b));
    printf("(3) empty kernel, graph replay       : %6.3f us/node\n", ms * 1000.0 / (20.0 * N));

    // ---- (4) graph of the tiny store kernel, to price a non-empty node
    cudaGraph_t g2; cudaGraphExec_t ge2;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    for (int j = 0; j < N; ++j) k_tiny<<<1, 256, 0, st>>>(d_o);
    CK(cudaStreamEndCapture(st, &g2));
    CK(cudaGraphInstantiate(&ge2, g2, nullptr, nullptr, 0));
    for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge2, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaEventRecord(a, st));
    for (int r = 0; r < 20; ++r) CK(cudaGraphLaunch(ge2, st));
    CK(cudaEventRecord(b, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaEventElapsedTime(&ms, a, b));
    printf("(4) tiny store kernel, graph replay  : %6.3f us/node\n", ms * 1000.0 / (20.0 * N));

    printf("\nInterpretation: if (2) >> (1), the profiler's floor is event overhead,\n"
           "and the small-kernel rows in the report are inflated by that constant.\n"
           "If (3) << (1), graph replay already hides most of the dispatch cost.\n");
    return 0;
}
