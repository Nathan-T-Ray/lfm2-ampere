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
// boundary becomes a flag in L2 plus a __threadfence().  Nobody had priced that
// on this GPU.  If it costs the same ~0.8-1.6 us the megakernel buys nothing,
// and it should be closed before anyone designs it.
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
// THE THIRD TRAP, and the reason for the SKEW sweep.  Every number above was
// measured with all 504 blocks doing identical work, so they all arrive at a
// barrier together and wait zero.  The engine is not like that.  A grid barrier
// waits for the SLOWEST block, whereas in a graph the next kernel's blocks can
// be dispatched onto SMs freed by the previous kernel's early finishers -- i.e.
// the graph OVERLAPS the stragglers and a barrier cannot.  That is the one
// mechanism that could erase the 6-12x advantage, and it is measured here by
// giving block b a slab of (1 + b % K) units, K = 1..8.
//
// All blocks must be RESIDENT for the persistent modes or the spin deadlocks,
// so NB comes from cudaOccupancyMaxActiveBlocksPerMultiprocessor and is printed
// rather than assumed.
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

// ---------------------------------------------------------------------------
// Skewed slab geometry, shared by every variant so the traffic is identical.
//
// Block b of stage s owns slab_len(b) = base * (1 + b % K) floats starting at
// ((s * nb * K) + b * K) * base.  K = 1 is the balanced case and reduces to the
// original layout exactly.  The stride is nb*K so the maximum extent is
// nstages * nb * K * base, whatever the weights are.
// ---------------------------------------------------------------------------
__device__ __forceinline__ const float * slab_ptr(const float * __restrict__ buf,
                                                  int s, int b, int nb, int base, int K) {
    return buf + (((size_t) s * nb * K) + (size_t) b * K) * (size_t) base;
}
__device__ __forceinline__ int slab_len(int b, int base, int K) {
    return base * (1 + (b % K));
}

// one stage, as its own kernel: block b reads its own skewed slab.
__global__ void k_stage(const float * __restrict__ buf, float * __restrict__ sink,
                        int base, int s, int K) {
    const int b = blockIdx.x;
    const float * p = slab_ptr(buf, s, b, gridDim.x, base, K);
    const int n = slab_len(b, base, K);
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += p[i];
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
                          int base, int nstages, int mode, int K) {
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
        const float * p = slab_ptr(buf, s, b, nb, base, K);
        const int n = slab_len(b, base, K);
        float acc = 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) acc += p[i];
        if (acc == 1.2345e30f) sink[b] = acc;
        __syncthreads();
        if (mode >= 1 && threadIdx.x == 0) {
            __threadfence();
            atomicAdd(&flags[b], 1);
        }
    }
}

static void timed(const std::vector<double> & v, double & lo, double & hi) {
    lo = v[0]; hi = v[0];
    for (double x : v) { if (x < lo) lo = x; if (x > hi) hi = x; }
}

int main() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s  SMs=%d  maxThreadsPerSM=%d\n",
           prop.name, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);

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

    const int SLAB_KB = 32;
    const int base = SLAB_KB * 1024 / 4;
    const int NSTAGES = 32;
    const int REPS = 3;
    const int KS[] = {1, 2, 4, 8};

    int * d_flags, * d_bar;
    CK(cudaMalloc(&d_flags, NB * sizeof(int)));
    CK(cudaMalloc(&d_bar, 2 * sizeof(int)));
    float * d_sink;
    CK(cudaMalloc(&d_sink, NB * sizeof(float)));

    // THE SKEW SWEEP.  K = 1 is the balanced case the first version measured.
    // As K grows the spread between the fastest and slowest block grows, and
    // the graph's ability to overlap stragglers -- which a grid barrier cannot
    // match -- should start to show up as the boundary cost falling.
    printf("SKEW SWEEP: block b owns (1 + b%%K) units, so K=1 is balanced.\n");
    printf("%3s %7s %8s %9s %9s %9s | %9s %9s %9s %9s\n",
           "K", "spread", "MB tot", "graph", "m1 BASE", "m3 grid", "boundary",
           "gridbar", "graph/base", "bar/base");
    for (int i = 0; i < 95; ++i) putchar('-');
    putchar('\n');

    for (int ki = 0; ki < 4; ++ki) {
        const int K = KS[ki];
        // NORMALISE the slab so total bytes stay constant across K.  The mean
        // weight is 1 + (K-1)/2, so without this the K=8 case would read 4.5x
        // the data of K=1 and the columns would not be comparable -- which is
        // exactly the mistake this line exists to prevent.
        const int basek = (int) (base / (1.0 + (K - 1) / 2.0));
        // buffer must cover nstages * NB * K * basek floats (worst-case stride)
        const size_t total_floats = (size_t) NSTAGES * NB * K * basek;
        const size_t total_bytes = total_floats * 4;
        const double read_mb = (double) NSTAGES * NB * (1.0 + (K - 1) / 2.0)
                               * basek * 4 / 1048576.0;

        float * d_buf;
        CK(cudaMalloc(&d_buf, total_bytes));
        CK(cudaMemset(d_buf, 0, total_bytes));

        std::vector<std::vector<double>> acc(3);   // graph, mode1, mode3

        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        for (int s = 0; s < NSTAGES; ++s)
            k_stage<<<NB, THREADS, 0, st>>>(d_buf, d_sink, basek, s, K);
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

            const int modes[2] = {1, 3};
            for (int mi = 0; mi < 2; ++mi) {
                CK(cudaMemset(d_flags, 0, NB * sizeof(int)));
                CK(cudaMemset(d_bar, 0, 2 * sizeof(int)));
                for (int i = 0; i < 2; ++i)
                    k_persist<<<NB, THREADS, 0, st>>>(d_buf, d_sink, d_flags, d_bar,
                                                      basek, NSTAGES, modes[mi], K);
                CK(cudaStreamSynchronize(st));
                CK(cudaEventRecord(ev_a, st));
                for (int q = 0; q < 3; ++q)
                    k_persist<<<NB, THREADS, 0, st>>>(d_buf, d_sink, d_flags, d_bar,
                                                      basek, NSTAGES, modes[mi], K);
                CK(cudaEventRecord(ev_b, st));
                CK(cudaStreamSynchronize(st));
                CK(cudaEventElapsedTime(&ms, ev_a, ev_b));
                acc[mi + 1].push_back(ms * 1000.0f / 3.0f);
            }
        }
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));

        double lo[3], hi[3];
        for (int q = 0; q < 3; ++q) timed(acc[q], lo[q], hi[q]);
        const double graph = lo[0], basem = lo[1], grid = lo[2];
        printf("%3d %6dx %8.0f %9.1f %9.1f %9.1f | %9.3f %9.3f %10.3f %8.3f\n",
               K, K, read_mb, graph, basem, grid,
               (graph - basem) / NSTAGES, (grid - basem) / NSTAGES,
               graph / basem, grid / basem);
        printf("      spread over %d reps: graph %.2f, base %.2f, grid %.2f us\n",
               REPS, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);

        CK(cudaFree(d_buf));
    }

    printf("\nVERDICT\n");
    printf("  boundary = graph vs persistent, per stage.  If it HOLDS or RISES with K,\n");
    printf("  the graph is NOT overlapping stragglers and the megakernel case stands.\n");
    printf("  If it COLLAPSES toward the grid-barrier cost as skew grows, then the\n");
    printf("  6-12x advantage was an artifact of perfectly balanced blocks and the\n");
    printf("  megakernel must be re-priced -- or closed.\n");

    CK(cudaFree(d_flags)); CK(cudaFree(d_bar)); CK(cudaFree(d_sink));
    CK(cudaEventDestroy(ev_a)); CK(cudaEventDestroy(ev_b));
    CK(cudaStreamDestroy(st));
    return 0;
}
