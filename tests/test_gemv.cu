// Standalone isolation of k_gemv_q4_0 on blk.0.shortconv.out_proj.weight.
// No engine, no layer loop: load planes -> quantise a known input -> GEMV -> dump.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include "../src/kernels.cuh"
#include "../src/tensor_table.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

__global__ void quant(const float * x, block_q8_1 * y, int nb) {
    const int b = blockIdx.x, lane = threadIdx.x & 31;
    if (b >= nb) return;
    quantize_q8_1_warp(x[b * 32 + lane], y + b, lane);
}

__global__ void gemv(const uint4 * qs, const __half * d, const block_q8_1 * xq,
                     int out_rows, int bpr, float * y) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= out_rows) return;
    const uint4 * qrow = qs + (size_t)warp * bpr;
    const __half * drow = d + (size_t)warp * bpr;
    float acc = 0.0f;
    for (int k = lane; k < bpr; k += 32) acc += dot_q4_0_q8_1(qrow, drow, k, xq + k);
    acc = warp_sum(acc);
    if (lane == 0) y[warp] = acc;
}

int main(int argc, char ** argv) {
    const char * pack = getenv("LK_PACK");
    const char * inf  = argv[1];         // 2048 fp32
    const char * outf = argv[2];
    std::vector<float> in(2048);
    FILE * f = fopen(inf, "rb");
    if (!f || fread(in.data(), 4, 2048, f) != 2048) { printf("bad input\n"); return 1; }
    fclose(f);

    int fd = open(pack, O_RDONLY);
    struct stat st{};
    if (fd < 0 || fstat(fd, &st)) { perror("pack"); return 1; }
    void * hb = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (hb == MAP_FAILED) { perror("mmap"); return 1; }
    close(fd);
    void * db = nullptr;
    CK(cudaMalloc(&db, st.st_size));
    CK(cudaMemcpy(db, hb, st.st_size, cudaMemcpyHostToDevice));

    const LKTensor * t = lk_find("blk.0.shortconv.out_proj.weight");
    printf("tensor: rows=%u bpr=%u planes={%llu,%llu}\n", t->rows, t->blocks_per_row,
           (unsigned long long)t->plane[0], (unsigned long long)t->plane[1]);

    float * dx; block_q8_1 * dq; float * dy;
    CK(cudaMalloc(&dx, 2048 * 4));
    CK(cudaMalloc(&dq, 64 * sizeof(block_q8_1)));
    CK(cudaMalloc(&dy, 2048 * 4));
    CK(cudaMemcpy(dx, in.data(), 2048 * 4, cudaMemcpyHostToDevice));
    quant<<<64, 32>>>(dx, dq, 64);
    CK(cudaGetLastError());
    gemv<<<(t->rows + 7) / 8, 256>>>((const uint4 *)((const char *)db + t->plane[0]),
                                     (const __half *)((const char *)db + t->plane[1]),
                                     dq, t->rows, t->blocks_per_row, dy);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<float> out(2048);
    CK(cudaMemcpy(out.data(), dy, 2048 * 4, cudaMemcpyDeviceToHost));
    FILE * g = fopen(outf, "wb");
    fwrite(out.data(), 4, 2048, g);
    fclose(g);
    printf("row1572=%.7f row0=%.7f row1000=%.7f  -> %s\n", out[1572], out[0], out[1000], outf);
    return 0;
}
