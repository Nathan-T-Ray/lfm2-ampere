// Where do the 2.3 us of k_rmsnorm_q8_1 actually go?
// The kernel already null-checks each output, so the phases can be measured by
// passing nullptrs -- no new kernel code, no risk of measuring a different thing.
#include "kernels.cuh"
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

static const int N = 2000;

static float timeit(cudaStream_t st, cudaEvent_t a, cudaEvent_t b, int variant,
                    float *x, float *w, float *y32, __half *y16, block_q8_1 *yq8, int nthreads) {
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < N; ++i) {
        float * p32 = (variant & 1) ? y32 : nullptr;
        __half * p16 = (variant & 1) ? y16 : nullptr;
        block_q8_1 * pq = (variant & 2) ? yq8 : nullptr;
        k_rmsnorm_q8_1<<<1, nthreads, LK_HIDDEN * 4, st>>>(x, w, LK_HIDDEN, LK_EPS, p32, p16, pq);
    }
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaEventRecord(a, st));
    for (int i = 0; i < 20; ++i) CK(cudaGraphLaunch(ge, st));
    CK(cudaEventRecord(b, st));
    CK(cudaStreamSynchronize(st));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    return (float)(ms * 1000.0 / (20.0 * N));
}

int main() {
    float *x, *w, *y32; __half *y16; block_q8_1 *yq8;
    CK(cudaMalloc(&x, LK_HIDDEN * 4)); CK(cudaMalloc(&w, LK_HIDDEN * 4));
    CK(cudaMalloc(&y32, LK_HIDDEN * 4)); CK(cudaMalloc(&y16, LK_HIDDEN * 2));
    CK(cudaMalloc(&yq8, 64 * sizeof(block_q8_1)));
    CK(cudaMemset(x, 0x3e, LK_HIDDEN * 4)); CK(cudaMemset(w, 0x3e, LK_HIDDEN * 4));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

    printf("k_rmsnorm_q8_1 phases, graph replay, 256 threads (us/launch):\n");
    const float red  = timeit(st, a, b, 0, x, w, y32, y16, yq8, 256);
    const float nq   = timeit(st, a, b, 1, x, w, y32, y16, yq8, 256);
    const float full = timeit(st, a, b, 3, x, w, y32, y16, yq8, 256);
    printf("  reduction + inv only          %6.3f\n", red);
    printf("  + normalise + h/h16 stores    %6.3f   (+%.3f)\n", nq, nq - red);
    printf("  + Q8_1 quantise (full)        %6.3f   (+%.3f)\n", full, full - nq);
    printf("  empty-kernel graph floor      %6.3f\n", 0.606);
    printf("  =>  execution                 %6.3f   x81 = %5.1f us/token\n",
           full - 0.606, (full - 0.606) * 81);

    printf("\nthread-count sweep (order-changing, for reference only):\n");
    for (int t = 256; t <= 1024; t *= 2)
        printf("  %4d threads  %6.3f us   x81 = %5.1f us\n", t,
               timeit(st, a, b, 3, x, w, y32, y16, yq8, t),
               timeit(st, a, b, 3, x, w, y32, y16, yq8, t) * 81);
    return 0;
}
