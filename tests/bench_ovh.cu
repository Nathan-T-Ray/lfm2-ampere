// bench_ovh.cu - the per-launch fixed cost, and whether a small grid can still
// saturate DRAM.
//
// Two questions, both raised by the engine's profile table.
//
//  1. Every streaming kernel in the profile reports a different GB/s, and the
//     figure falls monotonically as the kernel gets smaller:
//
//         lm_head       215.04 MB/call   925.1 GB/s
//         moe_gate_up    14.16 MB/call   727.8 GB/s
//         moe_down        7.08 MB/call   619.9 GB/s
//         attn_qkv        3.54 MB/call   431.3 GB/s
//         router_dot      0.52 MB/call   110.0 GB/s
//
//     A least-squares fit of time = a + bytes/BW over all eleven streaming
//     kernels gives BW = 941.8 GB/s with a = 4.12 us and residuals inside
//     +-1.03 us.  941.8 GB/s is the measured achievable figure (956 GB/s,
//     bandwidth_probe), so the fit says every one of those kernels is already
//     at peak marginal bandwidth and the whole spread is one constant.
//
//     This measures that constant directly instead of inferring it.
//
//  2. Several planned fusions force a 64-block x 1024-thread shape: one thread
//     block per Q8_1 block is exactly what makes the Phase 4y epilogue free,
//     and it is the only shape in which a 32-channel epilogue needs no
//     cross-block cooperation.  64 blocks on 84 SMs leaves 20 SMs idle, which
//     looks fatal.  Does it still saturate DRAM?
//
// The buffer is far larger than L2 (6 MB on sm_86) and every repetition reads a
// different window, so each measurement is cold.  Without that a 7 MB read
// would be partly L2-resident and would overstate bandwidth.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

// ---- the probe: pure cold streaming read, nothing else ---------------------
__global__ void k_read(const uint4 * __restrict__ p, long n4, float * __restrict__ out) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = (long)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (; i < n4; i += stride) {
        const uint4 v = __ldcs(p + i);
        acc += (float)(v.x ^ v.y ^ v.z ^ v.w);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0) out[(blockIdx.x * blockDim.x + threadIdx.x) >> 5] = acc;
}

static uint4 *    g_buf  = nullptr;
static size_t     g_bufsz = 0;      // bytes
static float *    g_out  = nullptr;
static cudaEvent_t g_ev[8192];
static cudaStream_t g_st = nullptr;   // NOT the legacy stream: that cannot be captured

// Read nbytes cold: repetition r reads the window starting at (r%W)*nbytes, so
// consecutive repetitions never touch the same bytes and W-1 windows' worth of
// data has streamed through L2 in between.
struct Res { double us; Res(double v = 0.0) : us(v) {} };   // default ctor so CK's `return 1` compiles

static Res run_windowed(size_t nbytes, int blocks, int threads, int reps, int bracketed) {
    const size_t n4 = nbytes / 16;
    const size_t windows = g_bufsz / nbytes;
    const size_t win_uint4 = n4;
    cudaStream_t st = g_st;

    for (int w = 0; w < 3; ++w)                       // warmup
        k_read<<<blocks, threads, 0, st>>>(g_buf + (w % windows) * win_uint4, (long)n4, g_out);
    CK(cudaStreamSynchronize(st));

    Res r{0.0};
    if (!bracketed) {
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a, st));
        for (int i = 0; i < reps; ++i)
            k_read<<<blocks, threads, 0, st>>>(g_buf + (i % windows) * win_uint4, (long)n4, g_out);
        CK(cudaEventRecord(b, st));
        CK(cudaStreamSynchronize(st));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        r.us = ms * 1000.0 / reps;
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    } else {
        // replicate the engine's LKP_B/LKP_E exactly: one event pair per launch,
        // on the same stream, elapsed per launch and summed.
        for (int i = 0; i < reps; ++i) {
            CK(cudaEventRecord(g_ev[2 * i], st));
            k_read<<<blocks, threads, 0, st>>>(g_buf + (i % windows) * win_uint4, (long)n4, g_out);
            CK(cudaEventRecord(g_ev[2 * i + 1], st));
        }
        CK(cudaStreamSynchronize(st));
        double tot = 0.0;
        for (int i = 0; i < reps; ++i) {
            float ms; CK(cudaEventElapsedTime(&ms, g_ev[2 * i], g_ev[2 * i + 1]));
            tot += ms * 1000.0;
        }
        r.us = tot / reps;
    }
    return r;
}

int main() {
    int dev = 0; cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    printf("device: %s  SMs=%d  L2=%d KB  mem=%.1f GB\n\n",
           prop.name, prop.multiProcessorCount, prop.l2CacheSize / 1024,
           prop.totalGlobalMem / 1073741824.0);
    const int NSM = prop.multiProcessorCount;

    g_bufsz = 768ull << 20;                            // 768 MB, >> L2 and >> any test size
    CK(cudaMalloc(&g_buf, g_bufsz));
    CK(cudaMemset(g_buf, 0x5a, g_bufsz));
    CK(cudaMalloc(&g_out, 1 << 22));
    CK(cudaStreamCreateWithFlags(&g_st, cudaStreamNonBlocking));
    for (int i = 0; i < 8192; ++i) CK(cudaEventCreateWithFlags(&g_ev[i], cudaEventDefault));

    const size_t MB = 1ull << 20;

    // =====================================================================
    printf("== A. does a small grid still saturate DRAM?  fixed 7.078 MB cold ==\n");
    printf("   (7.078 MB is moe_down's and shortconv_in's per-call traffic)\n\n");
    printf("   %-18s %10s %10s %10s %10s\n", "grid x block", "unbr us", "GB/s", "brk us", "brk GB/s");
    struct Shape { int b, t; const char * note; };
    const Shape shapes[] = {
        {  64, 1024, "T3 candidate / BIGBLOCK"}, { 192, 1024, "T2 candidate moe" },
        { 368, 1024, "T2 candidate dense" },      { 768,  256, "shortconv_in today" },
        {1024,  256, "moe_down today" },          {1536,  256, "moe_gate_up today" },
        {2048,  256, "many small blocks" },       {4096,  256, "oversubscribed" },
    };
    for (const Shape & s : shapes) {
        Res u = run_windowed(7077888ull, s.b, s.t, 300, 0);
        Res k = run_windowed(7077888ull, s.b, s.t, 300, 1);
        printf("   %-18s %10.2f %10.1f %10.2f %10.1f   %s\n",
               "", u.us, 7077888.0 / u.us / 1000.0, k.us, 7077888.0 / k.us / 1000.0, s.note);
    }
    printf("\n");

    // =====================================================================
    printf("== B. time vs bytes, cold, 1024 x 256, un-bracketed ==\n\n");
    printf("   %10s %10s %10s %10s\n", "MB", "us", "GB/s", "us/MB");
    const size_t sizes[] = { 512*1024, 1*MB, 2359296, 3538944, 7077888,
                             14155776, 28311552, 56623104, 120*MB, 215*MB };
    double px[16], py[16]; int np = 0;
    for (size_t s : sizes) {
        Res u = run_windowed(s, 1024, 256, 200, 0);
        printf("   %10.3f %10.2f %10.1f %10.4f\n", s / (double)MB, u.us,
               s / u.us / 1000.0, u.us / (s / (double)MB));
        px[np] = s / (double)MB; py[np] = u.us; ++np;
    }
    // least squares over the four largest, which are the streaming regime
    {
        double sx=0, sy=0, sxy=0, sxx=0; int n=0;
        for (int i = np - 4; i < np; ++i) { sx+=px[i]; sy+=py[i]; sxy+=px[i]*py[i]; sxx+=px[i]*px[i]; ++n; }
        const double sl = (n*sxy - sx*sy) / (n*sxx - sx*sx);
        const double a  = (sy - sl*sx) / n;
        printf("\n   fit over the four largest: marginal BW = %.1f GB/s, intercept = %.2f us\n",
               1000.0 / sl, a);
    }

    // =====================================================================
    printf("\n== C. the same sweep WITH the engine's per-launch event bracket ==\n\n");
    printf("   %10s %10s %10s %10s %10s\n", "MB", "unbr us", "brk us", "bracket us", "GB/s loss");
    for (int i = 0; i < np; ++i) {
        if (px[i] < 3.0 || px[i] > 60.0) continue;
        Res k = run_windowed((size_t)(px[i] * MB), 1024, 256, 200, 1);
        printf("   %10.3f %10.2f %10.2f %10.2f %10.1f\n", px[i], py[i], k.us, k.us - py[i],
               (px[i] * MB) / py[i] / 1000.0 - (px[i] * MB) / k.us / 1000.0);
    }

    // =====================================================================
    printf("\n== D. the same kernels as nodes of one CUDA graph ==\n");
    printf("   (the engine replays a graph, so this is the regime that matters)\n\n");
    {
        const int K = 371;                              // the engine's profiled call count
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStream_t st = g_st;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < K; ++i)
            k_read<<<1024, 256, 0, st>>>(g_buf + (size_t)(i % 64) * (7077888 / 16), 7077888 / 16, g_out);
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        size_t nodes = 0; cudaGraphGetNodes(g, nullptr, &nodes);
        for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, st));
        CK(cudaStreamSynchronize(st));
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a, st));
        for (int i = 0; i < 20; ++i) CK(cudaGraphLaunch(ge, st));
        CK(cudaEventRecord(b, st));
        CK(cudaStreamSynchronize(st));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        const double per = ms * 1000.0 / (20.0 * K);
        printf("   %d nodes, un-bracketed graph replay: %.3f us/node\n", (int)nodes, per);
        printf("   direct (non-graph) launch of the same kernel: %.3f us\n",
               run_windowed(7077888ull, 1024, 256, 300, 0).us);
        printf("   graph dispatch overhead per node: %.3f us\n",
               per - 7077888.0 / 941.8 / 1000.0);
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    }
    printf("\n");
    return 0;
}
