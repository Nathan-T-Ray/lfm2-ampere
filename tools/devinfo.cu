// devinfo.cu - GPU/L2 capability probe. Gate G0: establishes the real cache
// budgets that Phase 4's persisting-L2 strategy depends on.
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 1; } } while (0)

int main() {
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    printf("visible devices: %d\n", ndev);
    for (int d = 0; d < ndev; ++d) {
        cudaDeviceProp p{};
        CK(cudaGetDeviceProperties(&p, d));
        int persist_max = 0, window_max = 0;
        cudaDeviceGetLimit((size_t*)&persist_max, cudaLimitPersistingL2CacheSize);
        cudaDeviceGetAttribute(&window_max, cudaDevAttrMaxAccessPolicyWindowSize, d);
        printf("\n[%d] %s  cc %d.%d  SMs=%d\n", d, p.name, p.major, p.minor, p.multiProcessorCount);
        printf("    global mem      : %.1f MiB\n", p.totalGlobalMem / 1048576.0);
        printf("    L2 cache        : %d bytes (%.2f MiB)\n", p.l2CacheSize, p.l2CacheSize / 1048576.0);
        printf("    persistingL2Max : %d bytes (%.2f MiB)\n", p.persistingL2CacheMaxSize,
               p.persistingL2CacheMaxSize / 1048576.0);
        printf("    accessPolicyWin : %d bytes\n", window_max);
        printf("    limit(persist)  : %d bytes (%.2f MiB)\n", persist_max, persist_max / 1048576.0);
        printf("    smem/block      : %zu bytes   regs/SM: %d\n", p.sharedMemPerBlock, p.regsPerMultiprocessor);
        printf("    clock/mem       : %d kHz / %d kHz\n", p.clockRate, p.memoryClockRate);
        printf("    bus width       : %d bits  -> peak %.1f GB/s\n",
               p.memoryBusWidth, 2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1.0e6);
        printf("    async engines   : %d   cudaGraph: %s\n", p.asyncEngineCount,
               p.cooperativeLaunch ? "coop-launch ok" : "no coop");
    }
    return 0;
}
