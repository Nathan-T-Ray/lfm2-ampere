// bench_launch.cu - what does one kernel launch actually cost on this GPU?
//
// The per-kernel profile shows 451 launches per token, ~200 of which move no
// appreciable data and still measure 3.1-4.7 us each.  Before believing that,
// establish the floor: an empty kernel, measured three ways, because the three
// answers mean completely different things for the optimisation plan.
//   (1) launches back to back, timed as a batch      -> marginal cost
//   (2) each launch bracketed by its own events      -> what the profiler sees
//   (3) the same launches replayed from a CUDA graph -> what the engine pays
//
// Phase 4x adds the measurement the fusion plan actually depends on: the
// marginal cost of ADDING a node to a stream that is already busy, as a
// function of that node's grid size.  Test (3) prices an empty <<<1,256>>>
// node in an otherwise empty graph, which is not the situation any real kernel
// is in -- and the plan's whole projection scales with this number, so it gets
// measured rather than assumed.
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

__global__ void k_nop() {}
__global__ void k_tiny(float * o) { if (threadIdx.x == 0) o[blockIdx.x] = 1.0f; }

// A norm-shaped kernel: one block, 256 threads, an 8 KB dynamic shared array and
// a real 2048-element reduction with two syncthreads.  This is what the 80
// rmsnorms in a token actually are, so it is the thing whose removal is being
// priced -- an empty kernel is the wrong model for it.
__global__ void k_normlike(const float * __restrict__ x, float * __restrict__ y) {
    extern __shared__ float s[];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) {
        s[i] = x[i];
        acc += x[i] * x[i];
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) s[threadIdx.x] += s[threadIdx.x + o];
        __syncthreads();
    }
    const float r = rsqrtf(s[0] / 2048.0f + 1e-5f);
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) y[i] = s[i] * r;
}

// A streaming kernel that actually occupies the machine, so that the small
// nodes interleaved with it are measured in a busy stream rather than a vacuum.
__global__ void k_stream(const float * __restrict__ in, float * __restrict__ out, int n) {
    const float k = 1.0000001f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        out[i] = in[i] * k;
}

static const int N = 550;

static double replay_us(cudaGraphExec_t ge, cudaStream_t st, cudaEvent_t a, cudaEvent_t b,
                        int rounds) {
    for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaEventRecord(a, st));
    for (int r = 0; r < rounds; ++r) CK(cudaGraphLaunch(ge, st));
    CK(cudaEventRecord(b, st));
    CK(cudaStreamSynchronize(st));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    return ms * 1000.0 / rounds;
}

int main() {
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    float * d_o; CK(cudaMalloc(&d_o, (size_t) N * 4));

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
    printf("(3) empty kernel, graph replay       : %6.3f us/node\n",
           replay_us(ge, st, a, b, 20) / N);

    // ---- (4) graph of the tiny store kernel, to price a non-empty node
    cudaGraph_t g2; cudaGraphExec_t ge2;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    for (int j = 0; j < N; ++j) k_tiny<<<1, 256, 0, st>>>(d_o);
    CK(cudaStreamEndCapture(st, &g2));
    CK(cudaGraphInstantiate(&ge2, g2, nullptr, nullptr, 0));
    printf("(4) tiny store kernel, graph replay  : %6.3f us/node\n",
           replay_us(ge2, st, a, b, 20) / N);

    // =====================================================================
    // Phase 4x additions
    // =====================================================================

    // ---- (5) marginal node cost as a function of grid size
    //
    // An empty node with a big grid is NOT the same as an empty node with one
    // block: the front-end dispatches every block, and the tail of a 2048-block
    // grid costs real time even when each block exits immediately.  The
    // fusion plan removes nodes of specific shapes, so price them by shape.
    printf("\n(5) marginal cost of an EMPTY node, by grid (256 threads/block):\n");
    const int grids[] = {1, 8, 64, 256, 768, 1024, 1536};
    for (int gi = 0; gi < 7; ++gi) {
        const int G = grids[gi];
        cudaGraph_t gg; cudaGraphExec_t gge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        for (int j = 0; j < N; ++j) k_nop<<<G, 256, 0, st>>>();
        CK(cudaStreamEndCapture(st, &gg));
        CK(cudaGraphInstantiate(&gge, gg, nullptr, nullptr, 0));
        printf("    grid %5d : %6.3f us/node\n", G, replay_us(gge, st, a, b, 20) / N);
        CK(cudaGraphExecDestroy(gge)); CK(cudaGraphDestroy(gg));
    }

    // ---- (6) the shape that is actually being removed
    float * d_x, * d_y; CK(cudaMalloc(&d_x, 2048 * 4)); CK(cudaMalloc(&d_y, 2048 * 4));
    CK(cudaMemset(d_x, 0, 2048 * 4));
    cudaGraph_t g3; cudaGraphExec_t ge3;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    for (int j = 0; j < N; ++j) k_normlike<<<1, 256, 2048 * 4, st>>>(d_x, d_y);
    CK(cudaStreamEndCapture(st, &g3));
    CK(cudaGraphInstantiate(&ge3, g3, nullptr, nullptr, 0));
    printf("\n(6) norm-shaped kernel (1 block, 256 thr, 8 KB smem, 2048-elem reduction):\n");
    printf("    isolated in a graph                : %6.3f us/node\n",
           replay_us(ge3, st, a, b, 20) / N);

    // ---- (7) THE decision number: marginal cost of adding a small node to a
    //          BUSY stream.
    //
    // This is the only measurement that answers "how much do we get back by
    // deleting the 80 norm launches".  In an empty graph a node's dispatch may
    // be pipelined against nothing; behind a 30 us streaming kernel the
    // front-end has had time to prepare it, so the marginal cost can be far
    // below the isolated figure -- or the node may simply serialize, in which
    // case it is not.
    const int BIG = 2 << 20;                 // 2M floats = 8 MB in, 8 MB out
    float * d_in, * d_out;
    CK(cudaMalloc(&d_in, (size_t) BIG * 4)); CK(cudaMalloc(&d_out, (size_t) BIG * 4));
    CK(cudaMemset(d_in, 0, (size_t) BIG * 4));
    const int M = 32;

    double t_big, t_mix;
    {
        cudaGraph_t gg; cudaGraphExec_t gge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        for (int j = 0; j < M; ++j) k_stream<<<1024, 256, 0, st>>>(d_in, d_out, BIG);
        CK(cudaStreamEndCapture(st, &gg));
        CK(cudaGraphInstantiate(&gge, gg, nullptr, nullptr, 0));
        t_big = replay_us(gge, st, a, b, 10);
        CK(cudaGraphExecDestroy(gge)); CK(cudaGraphDestroy(gg));
    }
    {
        cudaGraph_t gg; cudaGraphExec_t gge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        for (int j = 0; j < M; ++j) {
            k_stream<<<1024, 256, 0, st>>>(d_in, d_out, BIG);
            k_normlike<<<1, 256, 2048 * 4, st>>>(d_x, d_y);
        }
        CK(cudaStreamEndCapture(st, &gg));
        CK(cudaGraphInstantiate(&gge, gg, nullptr, nullptr, 0));
        t_mix = replay_us(gge, st, a, b, 10);
        CK(cudaGraphExecDestroy(gge)); CK(cudaGraphDestroy(gg));
    }
    printf("\n(7) marginal cost of a norm node BEHIND a busy stream:\n");
    printf("    %d streaming nodes alone           : %8.3f us  (%.2f us each)\n",
           M, t_big, t_big / M);
    printf("    same %d + %d norm nodes            : %8.3f us  (%.2f us each)\n",
           M, M, t_mix, t_mix / M);
    printf("    --> MARGINAL COST PER NORM NODE    : %6.3f us\n", (t_mix - t_big) / M);

    printf("\nInterpretation: if (2) >> (1), the profiler's floor is event overhead,\n"
           "and the small-kernel rows in the report are inflated by that constant.\n"
           "If (3) << (1), graph replay already hides most of the dispatch cost.\n"
           "The number that governs Phase 4x is (7): deleting an isolated norm node\n"
           "saves its isolated cost, deleting one behind a busy stream saves (7).\n");
    return 0;
}
