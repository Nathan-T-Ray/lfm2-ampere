#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>
__global__ void stream_read(const uint4 * __restrict__ p, size_t n, float * out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < n; i += stride) {
        uint4 v = __ldcs(p + i);
        acc += (float)(v.x & 0xFF) + (float)(v.w & 0xFF);
    }
    if (acc == 1234567.f) out[0] = acc;
}
__global__ void stream_read_flat(const uint4 * __restrict__ p, size_t n, float * out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.f;
    if (i < n) { uint4 v = __ldcs(p + i); acc = (float)(v.x & 0xFF); }
    if (acc == 1234567.f) out[0] = acc;
}
int main() {
    const size_t BYTES = 1400ull << 20;
    const size_t N = BYTES / 16;
    uint4 * d; cudaMalloc(&d, BYTES); cudaMemset(d, 1, BYTES);
    float * o; cudaMalloc(&o, 4);
    cudaStream_t st; cudaStreamCreate(&st);
    for (int cfg = 0; cfg < 3; ++cfg) {
        const int blocks = (cfg == 0) ? 84 * 8 : (cfg == 1) ? 84 * 32 : 84 * 128;
        for (int w = 0; w < 2; ++w) stream_read<<<blocks, 256, 0, st>>>(d, N, o);
        cudaStreamSynchronize(st);
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 10; ++r) stream_read<<<blocks, 256, 0, st>>>(d, N, o);
        cudaStreamSynchronize(st);
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        printf("grid-stride blocks=%6d : %7.1f GB/s  (%.1f pct of 1008)\\n", blocks,
               10.0 * BYTES / s / 1e9, 100.0 * (10.0 * BYTES / s / 1e9) / 1008.0);
    }
    {
        const int blocks = (int)((N + 255) / 256);
        for (int w = 0; w < 2; ++w) stream_read_flat<<<blocks, 256, 0, st>>>(d, N, o);
        cudaStreamSynchronize(st);
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 10; ++r) stream_read_flat<<<blocks, 256, 0, st>>>(d, N, o);
        cudaStreamSynchronize(st);
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        printf("one-load-per-thread    : %7.1f GB/s  (%.1f pct of 1008)\\n",
               10.0 * BYTES / s / 1e9, 100.0 * (10.0 * BYTES / s / 1e9) / 1008.0);
    }
    return 0;
}
