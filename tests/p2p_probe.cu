// p2p_probe.cu - how fast can the two cards exchange the hidden state?
//
// nvidia-smi topo reports PHB (separate root ports, one host bridge) and
// cudaDeviceCanAccessPeer returns 0 in both directions: GeForce P2P is disabled
// on this platform, so anything crossing cards stages through host memory.
//
// That matters because the single-GPU batch-1 ceiling is measured at 488.9
// tok/s, and the only way left to 500 is a second card.  Tensor parallelism
// halves the bytes each card reads (~0.89 ms saved) and costs one exchange of
// the 2048-float hidden state per sub-block -- 80 per token.  At 8 KB each,
// those are LATENCY, not bandwidth:
//
//     80 x 2 us = 0.16 ms  -> the idea wins comfortably
//     80 x 8 us = 0.64 ms  -> roughly break-even
//     80 x 18 us = 1.44 ms -> it loses to the 0.89 ms it saves
//
// An earlier version measured a device pointer with cudaMemcpyHostToDevice,
// which is not a valid copy, so its 18 us figure came from an error path.  This
// version stages properly: D2H into PINNED host memory, then H2D on the other
// card, with the dependency expressed by a cross-device event wait -- exactly
// the shape a real all-reduce has.
//
// Events are created on the device whose stream records them; the elapsed time
// for the round trip is taken on the HOST, because cudaEventElapsedTime across
// two devices is not meaningful.
#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

static double now_us() {
    using namespace std::chrono;
    return (double)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

int main() {
    int n = 0; CK(cudaGetDeviceCount(&n));
    if (n < 2) { printf("need two GPUs\n"); return 1; }
    int can01 = 0; CK(cudaDeviceCanAccessPeer(&can01, 0, 1));
    printf("cudaDeviceCanAccessPeer(0->1) = %d  -> %s\n\n", can01,
           can01 ? "peer path available" : "HOST-STAGED ONLY");

    // Per-device objects.  An event must be created on the device whose stream
    // records it, or cudaStreamWaitEvent reports an invalid resource handle.
    cudaStream_t s0, s1; cudaEvent_t a0, a1, b0, b1, xdev;
    CK(cudaSetDevice(0));
    CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&b0));
    CK(cudaEventCreateWithFlags(&xdev, cudaEventDisableTiming));
    CK(cudaSetDevice(1));
    CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CK(cudaEventCreate(&a1)); CK(cudaEventCreate(&b1));

    const size_t sizes[] = { 2048, 8192, 32768, 262144 };
    const int NS = 4, ITER = 200;

    printf("   %8s %10s %10s %16s %s\n", "bytes", "D2H us", "H2D us",
           "round trip us", "80 per token");
    for (int s = 0; s < NS; ++s) {
        const size_t B = sizes[s];
        char *a = nullptr, *b = nullptr, *h = nullptr;
        CK(cudaSetDevice(0)); CK(cudaMalloc(&a, B)); CK(cudaMemset(a, 7, B));
        CK(cudaSetDevice(1)); CK(cudaMalloc(&b, B));
        CK(cudaHostAlloc(&h, B, cudaHostAllocDefault));    // pinned

        for (int i = 0; i < 20; ++i) {                     // warm the path
            CK(cudaSetDevice(0)); CK(cudaMemcpyAsync(h, a, B, cudaMemcpyDeviceToHost, s0));
            CK(cudaStreamSynchronize(s0));
            CK(cudaSetDevice(1)); CK(cudaMemcpyAsync(b, h, B, cudaMemcpyHostToDevice, s1));
            CK(cudaStreamSynchronize(s1));
        }

        // --- D2H alone, device-local events ---
        CK(cudaSetDevice(0));
        CK(cudaEventRecord(a0, s0));
        for (int i = 0; i < ITER; ++i) CK(cudaMemcpyAsync(h, a, B, cudaMemcpyDeviceToHost, s0));
        CK(cudaEventRecord(b0, s0)); CK(cudaStreamSynchronize(s0));
        float ms; CK(cudaEventElapsedTime(&ms, a0, b0));
        const double d2h = ms * 1000.0 / ITER;

        // --- H2D alone, device-local events ---
        CK(cudaSetDevice(1));
        CK(cudaEventRecord(a1, s1));
        for (int i = 0; i < ITER; ++i) CK(cudaMemcpyAsync(b, h, B, cudaMemcpyHostToDevice, s1));
        CK(cudaEventRecord(b1, s1)); CK(cudaStreamSynchronize(s1));
        CK(cudaEventElapsedTime(&ms, a1, b1));
        const double h2d = ms * 1000.0 / ITER;

        // --- dependent round trip, timed on the host ---
        CK(cudaStreamSynchronize(s0)); CK(cudaStreamSynchronize(s1));
        const double t0 = now_us();
        for (int i = 0; i < ITER; ++i) {
            CK(cudaSetDevice(0));
            CK(cudaMemcpyAsync(h, a, B, cudaMemcpyDeviceToHost, s0));
            CK(cudaEventRecord(xdev, s0));
            CK(cudaSetDevice(1));
            CK(cudaStreamWaitEvent(s1, xdev, 0));
            CK(cudaMemcpyAsync(b, h, B, cudaMemcpyHostToDevice, s1));
        }
        CK(cudaStreamSynchronize(s1));
        const double rt = (now_us() - t0) / ITER;

        printf("   %8zu %10.2f %10.2f %16.2f   %.2f ms\n", B, d2h, h2d, rt, rt * 80.0 / 1000.0);

        CK(cudaSetDevice(0)); CK(cudaFree(a));
        CK(cudaSetDevice(1)); CK(cudaFree(b));
        CK(cudaFreeHost(h));
    }

    printf("\n   tensor parallelism would save ~0.89 ms of streaming per token.\n");
    printf("   break-even is 0.89 ms / 80 exchanges = 11.1 us each.\n\n");
    return 0;
}
