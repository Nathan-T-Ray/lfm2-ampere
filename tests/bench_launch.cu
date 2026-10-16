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

// A pure READ-ONLY streaming kernel: no store, so it is the cleanest possible
// model of a weight-streaming node.  The guard is never true, which keeps the
// loop live without generating traffic.
__global__ void k_readonly(const float * __restrict__ in, float * __restrict__ out, int n) {
    float acc = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        acc += in[i];
    if (acc == 1.2345e30f) out[0] = acc;
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

    // =====================================================================
    // (8) THE RECOVERABLE TOLL.
    //
    // The engine's 0.268 ms launch toll was derived BY SUBTRACTION, not
    // measured, and the whole megakernel proposal rests on it being dead air.
    // But the bracket-corrected profiler sum (2.978 - 343*1.024us = 2.627 ms)
    // EXCEEDS the 2.269 ms wall clock, which cannot happen unless kernels
    // overlap -- i.e. node k+1's CTAs are already launching on SMs freed by
    // node k's retiring tail.  If that overlap is real, most of the "toll" is
    // not recoverable and a persistent kernel buys far less than 11.8%.
    //
    // Measure it directly: N nodes, each reading a distinct cold slab, with the
    // TOTAL bytes held constant.  Then
    //
    //     t = N * a + TOT / BW      ->   t/N = a + (TOT/N)/BW
    //
    // so plotting t/N against (TOT/N)/BW must give a straight line of slope 1
    // whose INTERCEPT is the per-node overhead with real work in every node.
    // Compare that intercept against the empty-node cost from (5), measured at
    // the same grid.  If a_busy << a_empty, the toll is already hidden.
    // =====================================================================
    printf("\n(8) per-node overhead when every node does REAL work, total bytes held\n"
           "    constant at 512 MB (grid 768x256 throughout, so only N varies):\n");
    {
        const size_t TOT = (size_t) 512 << 20;          // 512 MB, >> the 6 MB L2
        const int G8 = 768, B8 = 256;
        float * d_rd = nullptr, * d_sink = nullptr;
        CK(cudaMalloc(&d_rd, TOT));
        CK(cudaMemset(d_rd, 0, TOT));
        CK(cudaMalloc(&d_sink, 4));

        const int CFG8[][2] = {{16, 0}, {32, 0}, {64, 0}, {128, 0}, {256, 0}, {512, 0}};
        printf("    %5s %10s %11s %11s %11s %11s %9s\n",
               "N", "MB/node", "wall us", "us/node", "pure us", "overhead", "GB/s");
        double intercept_sum = 0.0; int intercept_n = 0;
        for (int c = 0; c < 6; ++c) {
            const int NN = CFG8[c][0];
            const size_t per = TOT / (size_t) NN;
            const int nf = (int) (per / 4);

            cudaGraph_t gg; cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            for (int j = 0; j < NN; ++j)
                k_readonly<<<G8, B8, 0, st>>>(d_rd + (size_t) j * nf, d_sink, nf);
            CK(cudaStreamEndCapture(st, &gg));
            CK(cudaGraphInstantiate(&ge, gg, nullptr, nullptr, 0));
            const double t = replay_us(ge, st, a, b, 10);
            CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(gg));

            const double per_node = t / NN;
            const double pure = (double) per / 941.8e9 * 1e6;   // us at the marginal rate
            const double over = per_node - pure;
            const double gbs = (double) TOT / (t * 1e-6) / 1e9;
            printf("    %5d %10.2f %11.3f %11.3f %11.3f %11.3f %9.1f\n",
                   NN, per / 1048576.0, t, per_node, pure, over, gbs);
            if (NN >= 64) { intercept_sum += over; ++intercept_n; }
        }
        const double a_busy = intercept_n ? intercept_sum / intercept_n : 0.0;

        // the same N and the same grid, with EMPTY nodes: what the toll is
        // claimed to be
        const int NE = 512;
        cudaGraph_t gg; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        for (int j = 0; j < NE; ++j) k_nop<<<G8, B8, 0, st>>>();
        CK(cudaStreamEndCapture(st, &gg));
        CK(cudaGraphInstantiate(&ge, gg, nullptr, nullptr, 0));
        const double a_empty = replay_us(ge, st, a, b, 10) / NE;
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(gg));

        printf("\n    per-node overhead, node does real work : %8.3f us\n", a_busy);
        printf("    per-node overhead, node is EMPTY       : %8.3f us  (grid %d)\n",
               a_empty, G8);
        printf("    --> share of the empty-node toll that is HIDDEN behind real work: %5.1f%%\n",
               100.0 * (1.0 - a_busy / a_empty));
        printf("    --> of the engine's 0.268 ms toll (343 x 0.781 us), the part that\n"
               "        would survive a persistent kernel is ~%.3f ms = %.1f%% of the step\n",
               343.0 * a_busy / 1000.0, 100.0 * 343.0 * a_busy / 1000.0 / 2.266);

        // ---- (8b) sum(brackets) vs wall, in the profiler's exact regime ----
        //
        // The engine's profiler puts each kernel in its own event pair and syncs
        // once at the end.  If that sum EXCEEDS the graph replay time for the
        // same work, the kernels overlap and the gaps are not dead air.
        {
            const int NB = 343;                  // the engine's node count
            const size_t per2 = TOT / (size_t) NB;
            const int nf2 = (int) (per2 / 4);
            std::vector<cudaEvent_t> ea(NB), eb(NB);
            for (int j = 0; j < NB; ++j) {
                CK(cudaEventCreate(&ea[j])); CK(cudaEventCreate(&eb[j]));
            }
            for (int w = 0; w < 2; ++w)                       // warm
                for (int j = 0; j < NB; ++j)
                    k_readonly<<<G8, B8, 0, st>>>(d_rd + (size_t) j * nf2, d_sink, nf2);
            CK(cudaStreamSynchronize(st));

            CK(cudaEventRecord(a, st));
            for (int j = 0; j < NB; ++j) {
                CK(cudaEventRecord(ea[j], st));
                k_readonly<<<G8, B8, 0, st>>>(d_rd + (size_t) j * nf2, d_sink, nf2);
                CK(cudaEventRecord(eb[j], st));
            }
            CK(cudaEventRecord(b, st));                       // ONE sync, as the profiler does
            CK(cudaStreamSynchronize(st));
            float wall_ms; CK(cudaEventElapsedTime(&wall_ms, a, b));
            double sum = 0.0;
            for (int j = 0; j < NB; ++j) {
                float ms; CK(cudaEventElapsedTime(&ms, ea[j], eb[j]));
                sum += ms;
            }
            // the same work, but bracketed by ONE pair: the un-instrumented cost
            for (int w = 0; w < 2; ++w)
                for (int j = 0; j < NB; ++j)
                    k_readonly<<<G8, B8, 0, st>>>(d_rd + (size_t) j * nf2, d_sink, nf2);
            CK(cudaStreamSynchronize(st));
            CK(cudaEventRecord(a, st));
            for (int j = 0; j < NB; ++j)
                k_readonly<<<G8, B8, 0, st>>>(d_rd + (size_t) j * nf2, d_sink, nf2);
            CK(cudaEventRecord(b, st));
            CK(cudaStreamSynchronize(st));
            float clean_ms; CK(cudaEventElapsedTime(&clean_ms, a, b));

            printf("\n(8b) the profiler's own regime, %d real nodes:\n", NB);
            printf("     one event pair around everything : %9.3f us\n", clean_ms * 1000.0);
            printf("     one event pair PER NODE, one sync: %9.3f us\n", wall_ms * 1000.0);
            printf("     SUM of the %d per-node brackets  : %9.3f us\n", NB, sum * 1000.0);
            printf("     --> per-node bracket cost        : %9.3f us\n",
                   (wall_ms - clean_ms) * 1000.0 / NB);
            printf("     --> sum(brackets) - wall         : %9.3f us  (instrument, not silicon)\n",
                   (sum - wall_ms) * 1000.0);
            for (int j = 0; j < NB; ++j) {
                CK(cudaEventDestroy(ea[j])); CK(cudaEventDestroy(eb[j]));
            }
        }

        CK(cudaFree(d_rd)); CK(cudaFree(d_sink));
    }

    printf("\nInterpretation: if (2) >> (1), the profiler's floor is event overhead,\n"
           "and the small-kernel rows in the report are inflated by that constant.\n"
           "If (3) << (1), graph replay already hides most of the dispatch cost.\n"
           "The number that governs Phase 4x is (7): deleting an isolated norm node\n"
           "saves its isolated cost, deleting one behind a busy stream saves (7).\n"
           "The number that governs the MEGAKERNEL is (8): if a busy node's overhead\n"
           "is far below an empty node's, the toll is already overlapped and a\n"
           "persistent kernel buys little.  (8b) tests the same thing via the\n"
           "profiler's own instrument and says whether that instrument is lying.\n");
    return 0;
}
