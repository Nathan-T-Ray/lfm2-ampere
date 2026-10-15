// Regression test: the Q6_K embedding lookup must match the reference
// dequantiser (tools/gguf.py, itself verified bit-exact against gguf-py).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
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
    const char * refp = (argc > 1) ? argv[1] : "ref/embed_ref.bin";
    FILE * rf = fopen(refp, "rb");
    if (!rf) { printf("cannot open %s\n", refp); return 1; }
    uint32_t n = 0;
    if (fread(&n, 4, 1, rf) != 1) return 1;
    std::vector<int32_t> toks(n);
    if (fread(toks.data(), 4, n, rf) != n) return 1;
    std::vector<float> ref((size_t)n * LK_HIDDEN);
    if (fread(ref.data(), 4, ref.size(), rf) != ref.size()) return 1;
    fclose(rf);

    int fd = open(pack, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st{};
    if (fstat(fd, &st)) { perror("fstat"); return 1; }
    void * hbase = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (hbase == MAP_FAILED) { perror("mmap"); return 1; }
    close(fd);

    void * dbase = nullptr;
    CK(cudaMalloc(&dbase, st.st_size));
    CK(cudaMemcpy(dbase, hbase, st.st_size, cudaMemcpyHostToDevice));

    const LKTensor * t = lk_find("token_embd.weight");
    float * dx = nullptr;
    CK(cudaMalloc(&dx, LK_HIDDEN * sizeof(float)));

    int bad = 0;
    printf("tok      max|got-ref|   rel        rms(got)   rms(ref)\n");
    for (uint32_t i = 0; i < n; ++i) {
        CK(cudaMemset(dx, 0, LK_HIDDEN * sizeof(float)));
        k_embed_q6_K<<<LK_HIDDEN / 256, 256>>>(
            (const uint8_t *)dbase + t->plane[0], (const uint8_t *)dbase + t->plane[1],
            (const int8_t  *)dbase + t->plane[2], (const __half *)dbase + t->plane[3] / 2,
            toks[i], dx);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<float> got(LK_HIDDEN);
        CK(cudaMemcpy(got.data(), dx, LK_HIDDEN * 4, cudaMemcpyDeviceToHost));
        const float * r = ref.data() + (size_t)i * LK_HIDDEN;
        double md = 0, sg = 0, sr = 0, mx = 0;
        for (int j = 0; j < LK_HIDDEN; ++j) {
            md = std::max(md, fabs((double)got[j] - r[j]));
            sg += (double)got[j] * got[j];
            sr += (double)r[j] * r[j];
            mx = std::max(mx, fabs((double)r[j]));
        }
        const bool ok = (md <= 1e-6 * (mx + 1e-9)) || md == 0.0;
        if (!ok) ++bad;
        printf("%6d  %.3e   %.3e  %.6f   %.6f  %s\n", toks[i], md, md / (mx + 1e-9),
               sqrt(sg / LK_HIDDEN), sqrt(sr / LK_HIDDEN), ok ? "OK" : "FAIL");
    }
    printf("\nEMBED TEST: %s\n", bad ? "FAIL" : "PASS - lookup is exact");
    return bad ? 1 : 0;
}
