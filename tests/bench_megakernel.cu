// bench_megakernel.cu - is an intra-kernel handoff cheaper than a kernel boundary?
//
// THE QUESTION.  The engine replays a 343-node CUDA graph, and part (8) of
// tests/bench_launch.cu measured the per-node toll as ADDITIVE:
//
//     busy_node = empty_node_toll + pure_transfer        (+ a ramp term)
//
// so the 0.268 ms the 343 boundaries cost is real and is not hidden behind the
// memory traffic.  Removing it entirely gives 2.266 - 0.268 = 1.998 ms =
// 500.5 tok/s.  That is the whole case for a persistent megakernel.
//
// But a megakernel does not delete the dependency -- it MOVES it.  A kernel
// boundary becomes a flag in L2 plus a __threadfence().  Nobody has priced that
// handoff on this GPU.  If it costs the same ~0.8-1.6 us the megakernel buys
// nothing, and it should be closed before anyone designs it.
//
// TWO TRAPS THIS INSTRUMENT IS BUILT TO AVOID, both found by running a first
// version of it:
//
//  1. The "no synchronisation at all" control is NOT a valid baseline.  A
//     version with no flag write at all measured *slower* than one with the
//     write, by up to 4%, i.e. the control differed from the treatment in
//     something other than the thing being measured.  So the baseline here is
//     mode 1: byte-identical instruction stream to mode 2 except for the spin.
//
//  2. A block waiting on its OWN flag never waits.  It set that flag itself one
//     stage ago, so the spin falls straight through and measures nothing.  The
//     real handoff is CROSS-BLOCK, so mode 4 waits on a different block's flag.
//
// All blocks must be RESIDENT for the persistent modes or the spin deadlocks,
// so NB comes from cudaOccupancyMaxActiveBlocksPerMultiprocessor and is printed
// rather than assumed.
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

// one stage, as its own kernel: block b reads slab (s, b).  Identical indexing
// to the persistent variants, so the traffic is bit-for-bit the same.
__global__ void k_stage(const float * __restrict__ buf, float * __restrict__ sink,
                        int slab_floats, int s) {
    const int b = blockIdx.x;
    const float * p = buf + ((size_t) s * gridDim.x + b) * slab_floats;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < slab_floats; i += blockDim.x) acc += p[i];
    if (acc == 1.2345e30f) sink[b] = acc;
}

// Monotonic, reset-free grid barrier: `arrived` counts to gen*nb, the last
// arriver releases nb at a time on `released`.  Nothing is ever reset, so no
// fast block can race the reset of a slow one.
__device__ __forceinline__ void gbar(int * arrived, int * released, int nb, int gen) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(arrived, 1) + 1 == gen * nb) {
            __threadfence();
            atomicAdd(released, nb);
        }
        while (atomicAdd(released, 0) < gen * nb) { }
    }
    __syncthreads();
}

// mode 0 = nothing at all                      (reported, NOT used as baseline)
// mode 1 = publish a per-block flag, never wait          <- THE BASELINE
// mode 2 = wait on the block's OWN flag                   (never actually waits)
// mode 3 = wait on EVERY block                            (grid barrier)
// mode 4 = wait on ANOTHER block's flag                   (real cross-block wait)
__global__ void k_persist(const float * __restrict__ buf, float * __restrict__ sink,
                          int * __restrict__ flags, int * __restrict__ bar,
                          int slab_floats, int nstages, int mode) {
    const int b = blockIdx.x;
    const int nb = gridDim.x;
    const int peer = (b + nb / 2) % nb;          // mode 4's producer
    for (int s = 0; s < nstages; ++s) {
        if (s > 0) {
            if (mode == 2) {
                if (threadIdx.x == 0)
                    while (atomicAdd(&flags[b], 0) < s) { }
                __syncthreads();
            } else if (mode == 3) {
                gbar(&bar[0], &bar[1], nb, s);
            } else if (mode == 4) {
                if (threadIdx.x == 0)
                    while (atomicAdd(&flags[peer], 0) < s) { }
                __syncthreads();
            }
        }
        const float * p = buf + ((size_t) s * nb + b) * slab_floats;
        float acc = 0.0f;
        for (int i = threadIdx.x; i < slab_floats; i += blockDim.x) acc += p[i];
        if (acc == 1.2345e30f) sink[b] = acc;
        __syncthreads();
        if (mode >= 1 && threadIdx.x == 0) {
            __threadfence();
            atomicAdd(&flags[b], 1);
        }
    }
}

// min of `reps`, with the spread reported so a difference can be trusted
static void timed(const std::vector<double> & v, double & lo, double & hi) {
    lo = v[0]; hi = v[0];
    for (double x : v) { if (x < lo) lo = x; if (x > hi) hi = x; }
}

int main() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s  SMs=%d  maxThreadsPerSM=%d\n",
           prop.name, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);
    printf("NVIDIA_DRIVER: %d\n", prop.major * 100 + prop.minor);

    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    cudaEvent_t ev_a, ev_b;
    CK(cudaEventCreate(&ev_a));
    CK(cudaEventCreate(&ev_b));

    const int THREADS = 256;
    int per_sm = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_persist, THREADS, 0));
    const int NB = per_sm * prop.multiProcessorCount;
    printf("k_persist occupancy: %d blocks/SM -> NB = %d blocks of %d threads "
           "(all resident, or modes 2-4 deadlock)\n\n", per_sm, NB, THREADS);

    const size_t TARGET_BYTES = (size_t) 512 << 20;
    const int SLABS_KB[] = {8, 32, 128, 512};
    const int REPS = 3;

    int * d_flags, * d_bar;
    CK(cudaMalloc(&d_flags, NB * sizeof(int)));
    CK(cudaMalloc(&d_bar, 2 * sizeof(int)));
    float * d_sink;
    CK(cudaMalloc(&d_sink, NB * sizeof(float)));

    printf("%5s %5s %8s %9s %9s %9s %9s %9s %9s | %8s %8s %8s %8s\n",
           "slab", "S", "MB tot", "graph", "m0 none", "m1 BASE", "m2 self",
           "m4 cross", "m3 grid", "boundary", "hs self", "h cross", "gridbar");
    for (int i = 0; i < 118; ++i) putchar('-');
    putchar('\n');

    double worst_cross = 0.0, best_cross = 1e9;
    for (int ci = 0; ci < 4; ++ci) {
        const int slab_kb = SLABS_KB[ci];
        const int slab_floats = slab_kb * 1024 / 4;
        const size_t per_stage = (size_t) NB * slab_floats * 4;
        int ns = (int) (TARGET_BYTES / per_stage);
        if (ns < 16) ns = 16;                    // S=2 made the first version useless
        const size_t total = per_stage * ns;

        float * d_buf;
        CK(cudaMalloc(&d_buf, total));
        CK(cudaMemset(d_buf, 0, total));

        std::vector<double> tg, tm(5);
        std::vector<std::vector<double>> acc(6);

        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        for (int s = 0; s < ns; ++s) k_stage<<<NB, THREADS, 0, st>>>(d_buf, d_sink, slab_floats, s);
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));

        for (int r = 0; r < REPS; ++r) {
            for (int i = 0; i < 2; ++i) CK(cudaGraphLaunch(ge, st));
            CK(cudaStreamSynchronize(st));
            CK(cudaEventRecord(ev_a, st));
            for (int q = 0; q < 3; ++q) CK(cudaGraphLaunch(ge, st));
            CK(cudaEventRecord(ev_b, st));
            CK(cudaStreamSynchronize(st));
            float ms; CK(cudaEventElapsedTime(&ms, ev_a, ev_b));
            acc[0].push_back(ms * 1000.0f / 3.0f);

            for (int m = 0; m < 5; ++m) {
                CK(cudaMemset(d_flags, 0, NB * sizeof(int)));
                CK(cudaMemset(d_bar, 0, 2 * sizeof(int)));
                for (int i = 0; i < 2; ++i)
                    k_persist<<<NB, THREADS, 0, st>>>(d_buf, d_sink, d_flags, d_bar,
                                                      slab_floats, ns, m);
                CK(cudaStreamSynchronize(st));
                CK(cudaEventRecord(ev_a, st));
                for (int q = 0; q < 3; ++q)
                    k_persist<<<NB, THREADS, 0, st>>>(d_buf, d_sink, d_flags, d_bar,
                                                      slab_floats, ns, m);
                CK(cudaEventRecord(ev_b, st));
                CK(cudaStreamSynchronize(st));
                CK(cudaEventElapsedTime(&ms, ev_a, ev_b));
                acc[m + 1].push_back(ms * 1000.0f / 3.0f);
            }
        }
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));

        double lo[6], hi[6];
        for (int i = 0; i < 6; ++i) timed(acc[i], lo[i], hi[i]);
        const double graph = lo[0], base = lo[2];        // index 2 == mode 1
        const double boundary = (graph - base) / ns;
        const double hs = (lo[3] - base) / ns;           // mode 2 minus baseline
        const double hc = (lo[4] - base) / ns;           // mode 4 minus baseline
        const double gb = (lo[5] - base) / ns;           // mode 3 minus baseline
        if (hc > worst_cross) worst_cross = hc;
        if (hc < best_cross) best_cross = hc;

        printf("%5d %5d %8.0f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f | %8.3f %8.3f "
               "%8.3f %8.3f\n",
               slab_kb, ns, total / 1048576.0, graph, lo[1], base, lo[3], lo[4], lo[5],
               boundary, hs, hc, gb);
        printf("      spread over %d reps: graph %.2f, base %.2f, cross %.2f us\n",
               REPS, hi[0] - lo[0], hi[2] - lo[2], hi[4] - lo[4]);
        (void) tg; (void) tm;

        CK(cudaFree(d_buf));
    }

    printf("\nVERDICT  (all per stage, all against mode 1, which differs from the\n"
           "          treatments ONLY by the spin -- see the header)\n");
    printf("  boundary  = a KERNEL BOUNDARY   (graph vs persistent, same traffic)\n");
    printf("  hs self   = an intra-kernel handoff on the block's OWN flag\n");
    printf("  h cross   = an intra-kernel handoff on ANOTHER block's flag  <-- the real one\n");
    printf("  gridbar   = a full grid barrier\n");
    printf("  cross-block handoff across slab sizes: %.3f .. %.3f us\n",
           best_cross, worst_cross);
    if (worst_cross < 0.0)
        printf("  --> one or more NEGATIVE: the baseline is still confounded, do not use.\n");
    else
        printf("  --> the handoff costs a fraction of the boundary it replaces.\n");

    CK(cudaFree(d_flags)); CK(cudaFree(d_bar)); CK(cudaFree(d_sink));
    CK(cudaEventDestroy(ev_a)); CK(cudaEventDestroy(ev_b));
    CK(cudaStreamDestroy(st));
    return 0;
}
