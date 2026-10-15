// Minimal reproducer / regression test for the Q6_K embedding lookup.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include "../src/kernels.cuh"
#include "../src/tensor_table.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

int main(int argc, char ** argv) {
    const char * pack = getenv("LK_PACK");
    const int token = (argc > 1) ? atoi(argv[1]) : 124890;
    int fd = open(pack, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st{};
    if (fstat(fd, &st)) { perror("fstat"); return 1; }
    void * base = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    printf("pack mapped: %zu bytes\n", (size_t)st.st_size);

    const LKTensor * t = lk_find("token_embd.weight");
    if (!t) { printf("tensor not found\n"); return 1; }
    printf("token_embd: type=%u nelem=%llu cols=%u rows=%u bpr=%u planes={%llu,%llu,%llu,%llu}\n",
           t->ggml_type, (unsigned long long)t->nelem, t->cols, t->rows, t->blocks_per_row,
           (unsigned long long)t->plane[0], (unsigned long long)t->plane[1],
           (unsigned long long)t->plane[2], (unsigned long long)t->plane[3]);

    float * dx = nullptr;
    CK(cudaMalloc(&dx, LK_HIDDEN * sizeof(float)));
    CK(cudaMemset(dx, 0, LK_HIDDEN * sizeof(float)));
    printf("launching k_embed_q6_K for token %d ...\n", token);
    fflush(stdout);
    k_embed_q6_K<<<LK_HIDDEN / 256, 256>>>(
        (const uint8_t *)base + t->plane[0], (const uint8_t *)base + t->plane[1],
        (const int8_t  *)base + t->plane[2], (const __half *)base + t->plane[3] / 2,
        token, dx);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    printf("kernel OK\n");

    float h[8];
    CK(cudaMemcpy(h, dx, sizeof h, cudaMemcpyDeviceToHost));
    printf("x[0..7] =");
    for (int i = 0; i < 8; ++i) printf(" %+.6f", h[i]);
    printf("\n");

    // whole-row sanity
    float * full = (float *)malloc(LK_HIDDEN * sizeof(float));
    CK(cudaMemcpy(full, dx, LK_HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));
    double mn = 1e30, mx = -1e30, sumsq = 0; int nbad = 0;
    for (int i = 0; i < LK_HIDDEN; ++i) {
        if (!std::isfinite(full[i])) ++nbad;
        mn = fmin(mn, full[i]); mx = fmax(mx, full[i]); sumsq += (double)full[i] * full[i];
    }
    printf("row stats: min=%+.6f max=%+.6f rms=%.6f nonfinite=%d\n",
           mn, mx, sqrt(sumsq / LK_HIDDEN), nbad);
    return nbad ? 1 : 0;
}
