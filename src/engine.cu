// engine.cu - liquid-kernel Phase 3: a correct, complete, deliberately
// unoptimised LFM2.5-24B-A2B forward pass.  Speed work happens in Phase 4;
// this exists to pass the C1/C2 correctness gates first.
#include "kernels.cuh"
#include "prefill.cuh"
#include "tensor_table.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
// LK_DEBUG_STEPS=1 in the environment makes every stage synchronise and report,
// which localises a fault to an exact kernel.  Off by default (it serialises).
static int g_debug_steps = 0;

// ---------------------------------------------------------------------------
// Per-kernel GPU timing (LK_PROFILE=1).
//
// Every hypothesis I formed about where the remaining 2.3 ms goes was
// falsified by a variant build, which means the bottleneck is something I have
// not thought to vary.  Instrument instead of guessing: bracket each launch
// with a pair of events in the same stream.  cudaEventElapsedTime between two
// events already recorded on the stream is GPU execution time for whatever ran
// between them, so this measures the kernel and not the launch overhead.
//
// The paired records do perturb the run (each is a small timestamp write on the
// GPU timeline), so absolute totals run a little high; the distribution is the
// point.  Reporting starts empty because the first records happen before any
// kernel is resident.
#define LK_PROF_MAX 2048
static bool         g_prof_on = false;
static int          g_prof_n  = 0;
static cudaStream_t g_prof_st = nullptr;
static cudaEvent_t  g_prof_ev[LK_PROF_MAX * 2];
static const char * g_prof_tag[LK_PROF_MAX];

static void lk_prof_init(cudaStream_t st) {
    g_prof_st = st;
    for (int i = 0; i < LK_PROF_MAX * 2; ++i)
        CK(cudaEventCreateWithFlags(&g_prof_ev[i], cudaEventDefault));
}

static inline void lk_prof_begin(const char * tag) {
    if (g_prof_n >= LK_PROF_MAX) return;
    g_prof_tag[g_prof_n] = tag;
    cudaEventRecord(g_prof_ev[g_prof_n * 2], g_prof_st);
}
static inline void lk_prof_end(void) {
    if (g_prof_n >= LK_PROF_MAX) return;
    cudaEventRecord(g_prof_ev[g_prof_n * 2 + 1], g_prof_st);
    ++g_prof_n;
}
#define LKP_B(t) do { if (g_prof_on) lk_prof_begin(t); } while (0)
#define LKP_E()  do { if (g_prof_on) lk_prof_end(); } while (0)

// Bytes each kernel must pull from HBM per call, so the report can state the
// achieved bandwidth next to the time.  A kernel that is at the roofline is
// done; one that is far below it has a reason that is worth finding.
static double lk_prof_bytes(const char * t) {
    struct Row { const char * n; double b; };
    static const Row T[] = {
        {"embed",           1680.0}, {"norm_attn",     16384.0},
        // shortconv_out was silently dropped when four quantise entries were
        // removed on a previous pass, which made it look like a compute-bound
        // kernel instead of a 70 MB/token stream at 674 GB/s.
        {"shortconv_out", 2359296.0},
        {"shortconv_in", 7077888.0}, {"shortconv_conv", 24576.0},
                {"attn_qkv",     3538944.0}, {"qk_norm_rope",       0.0},
        {"attention",          0.0}, {"attn_split",         0.0},
        {"attn_combine",       0.0},         {"attn_out",     2359296.0}, {"norm_ffn",      16384.0},
        {"dense_gate",  13565952.0}, {"dense_up",    13565952.0},
        {"dense_swiglu",       0.0},         {"dense_down",  13565952.0}, {"router_dot",    524288.0},
        {"router_top4",        0.0}, {"route_log",          0.0},
        {"moe_gate_up", 14155776.0},
        {"moe_swiglu",         0.0},         {"moe_down",     7077888.0}, {"moe_accum",          0.0},
        {"norm_final",     16384.0}, {"lm_head",    215040000.0},
        {"argmax_part",   512000.0}, {"argmax_final",       0.0},
    };
    for (const Row & r : T) if (!strcmp(r.n, t)) return r.b;
    return 0.0;
}

static void lk_prof_report(int ctx) {
    struct Agg { const char * tag; int calls; double ms; };
    std::vector<Agg> ag;
    double total = 0.0;
    for (int i = 0; i < g_prof_n; ++i) {
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, g_prof_ev[i * 2], g_prof_ev[i * 2 + 1]));
        total += ms;
        int found = -1;
        for (size_t j = 0; j < ag.size(); ++j)
            if (!strcmp(ag[j].tag, g_prof_tag[i])) { found = (int) j; break; }
        if (found < 0) { ag.push_back({g_prof_tag[i], 1, ms}); }
        else           { ag[found].calls++; ag[found].ms += ms; }
    }
    std::sort(ag.begin(), ag.end(),
              [](const Agg & a, const Agg & b) { return a.ms > b.ms; });

    printf("\n=== per-kernel GPU time, one decode step at ctx=%d ===\n", ctx);
    printf("%-16s %6s %10s %9s %10s %9s %7s\n",
           "category", "calls", "total_ms", "avg_us", "MB", "GB/s", "%step");
    double bytes = 0.0;
    for (const Agg & a : ag) {
        double b = lk_prof_bytes(a.tag) * a.calls;
        if (!strcmp(a.tag, "attention") || !strcmp(a.tag, "attn_split"))
            b = (double) ctx * 2048.0 * a.calls;
        bytes += b;
        const double gbs = (a.ms > 0.0) ? b / (a.ms * 1e-3) / 1e9 : 0.0;
        printf("%-16s %6d %10.3f %9.2f %10.2f %9.1f %6.1f%%\n",
               a.tag, a.calls, a.ms, a.ms * 1000.0 / a.calls,
               b / 1e6, gbs, 100.0 * a.ms / total);
    }
    printf("%-16s %6d %10.3f %9s %10.2f %9.1f %6s\n", "TOTAL", g_prof_n, total,
           "", bytes / 1e6, bytes / (total * 1e-3) / 1e9, "100.0%");
    printf("(sum of kernel times; excludes launch gaps)\n\n");
    fflush(stdout);
}

#define LK_STEP(tag) do { if (g_debug_steps) { \
    cudaError_t e_ = cudaStreamSynchronize(st); \
    if (e_ != cudaSuccess) { fprintf(stderr, "FAULT at stage [%s] layer %d: %s\n", \
        tag, il, cudaGetErrorString(e_)); exit(1); } } } while (0)

struct Layer {
    const LKTensor *attn_norm = nullptr, *ffn_norm = nullptr;
    bool is_attn = false, is_moe = false;
    const LKTensor *sc_conv = nullptr, *sc_in = nullptr, *sc_out = nullptr;
    const LKTensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr,
                   *qnorm = nullptr, *knorm = nullptr;
    const LKTensor *ffn_gate = nullptr, *ffn_up = nullptr, *ffn_down = nullptr;
    const LKTensor *router = nullptr, *rbias = nullptr,
                   *eg = nullptr, *eu = nullptr, *ed = nullptr;
};

struct Engine {
    void *   pack = nullptr;      // host mmap, used only to stage the copy
    void *   d_pack = nullptr;    // DEVICE copy of the pack -- all kernels read
                                  // weights from here.  A kernel cannot
                                  // dereference pageable host memory (mmap is
                                  // not pinned), and even with cudaHostRegister
                                  // zero-copy over PCIe would be ~40x too slow
                                  // to reach the HBM roofline.
    size_t   pack_size = 0;
    Layer    L[LK_N_LAYERS];
    const LKTensor *embd = nullptr, *out_norm = nullptr;
    int      max_seq = 0;

    // device state
    float *  x = nullptr;        // [2048] residual stream
    float *  h = nullptr;        // [2048] post-norm (fp32, for the router)
    __half * h16 = nullptr;      // [2048] post-norm (fp16, router input)
    block_q8_1 * xq = nullptr;   // [368] quantised hidden-state activations
    // Phase 4x fused norm epilogue.  One scratch array and one counter, shared
    // by every fused call site.  Safe because they are stream-ordered and the
    // last block resets the counter before exiting -- but the reset is
    // load-bearing, so tests/test_normfuse.cu replays twice.
    float *        norm_partial = nullptr;   // [LK_NORM_PARTIALS]
    unsigned int * norm_ctr = nullptr;       // [1], zeroed at init
    // Separate counter for the router fusion: sharing one between two fused
    // epilogues would work only because they are stream-ordered, which is too
    // subtle a reason to rely on.
    unsigned int * router_ctr = nullptr;     // [1], zeroed at init
    // Phase 4y: sum-of-squares partials handed from a producer kernel to the
    // kernel that consumes its activation.  No counter and nothing to reset --
    // the producer overwrites all LK_NORM_BLOCKS entries on every call, so a
    // stale value is impossible by construction (unlike norm_ctr above).
    float *        norm_ssq = nullptr;       // [LK_NORM_BLOCKS]
    float *  attn_part = nullptr; // [heads][SPLIT][2 + head_dim] split-K scratch
    block_q8_1 * xq2 = nullptr;  // [368] second scratch: the MoE expert loop needs
                                 //       the hidden-state activation to survive
                                 //       across all 4 experts while the expert
                                 //       intermediate activation is also live
    float *  y = nullptr;        // [2*11776] dense gate at 0, up at 11776
    float *  inter = nullptr;    // [11776] SwiGLU output (distinct from y, which
                                 //         k_swiglu's __restrict__ inputs alias)
    float *  tmp = nullptr;      // [2048]  conv/attn output scratch
    float *  qv = nullptr;       // [2048] Q
    float *  kv = nullptr;       // [512]  K
    float *  vv = nullptr;       // [512]  V
    float *  logits = nullptr;   // [128000]
    int *    d_token = nullptr;   // current input token (= previous argmax)
    int *    d_pos = nullptr;     // current sequence position
    float *  conv_hist = nullptr;// [30 * 2048 * 2]
    __half * kcache = nullptr;   // [10][max_seq][8][64]
    __half * vcache = nullptr;
    float *  moe_gu = nullptr;    // [2 * TOPK * MOE_INTER] gate4 then up4
    float *  moe_inter4 = nullptr;// [TOPK * MOE_INTER]
    float *  moe_down4 = nullptr; // [TOPK * HIDDEN]
    float *  moe_gate = nullptr, * moe_up = nullptr, * moe_inter = nullptr;
    float *  moe_down = nullptr;
    float *  d_gate = nullptr;   // [64]  router logits, one per expert
    // T4r: the two 64-float arrays the selection actually consumes.  `d_sel` is
    // sigmoid + bias and drives the RANKING; `d_sig` is the unbiased sigmoid and
    // supplies the WEIGHTS.  Both are written by k_router_dot and re-read by
    // every block of moe_gate_up, which is what lets k_router_top4 disappear.
    float *  d_sel = nullptr;    // [64]
    float *  d_sig = nullptr;    // [64]
    // Speculative L2 prefetch: bytes to pull forward per moe_swiglu call.
    // 2 MB measured best in a 3-round interleaved sweep (0 / 512K / 1M / 2M /
    // 4M gave 441.31 / 439.45 / 442.86 / 443.49 / 441.34), and 2 MB then beat 0
    // in 6 of 7 confirmation rounds.  LK_PF_BYTES overrides it for sweeping.
    long pf_bytes = 2 * 1024 * 1024;
    float *  d_apv = nullptr;    // [125] argmax stage-1 block winners
    int *    d_api = nullptr;    // [125] their indices
    int *    d_routelog = nullptr; // [max_seq][LK_N_LAYERS][LK_TOP_K], G4b only
    int *    d_idx = nullptr;
    float *  d_w = nullptr;

    // Batched prefill buffers (prefill.cuh), sized for a chunk of pf_mc tokens
    // and allocated on first use so decode-only runs pay nothing for them.
    struct {
        int      mc = 0;
        int *    tok = nullptr;    // [mc]
        float *  x = nullptr;      // [mc][2048] residual stream
        float *  h32 = nullptr;    // [mc][2048] router input (raw x*w)
        float *  inv = nullptr;    // [mc] Phase 4y norm scale for the consumer
        PfAct    a{};              // [mc][<=11776] activations
        PfAct    a2{};             // SwiGLU output: [mc][11776] dense, [mc*4][1536] MoE
        float *  y = nullptr;      // [mc][23552] GEMM outputs
        float *  o = nullptr;      // [mc][2048] attention output
        __half * qh = nullptr;     // [mc][2048] fp16 Q for the tensor-core attention
        float *  sel = nullptr, * sig = nullptr;   // [mc][64]
        int *    idx = nullptr;    // [mc][4]
        float *  w = nullptr;      // [mc][4]
        int *    cnt = nullptr, * off = nullptr;   // [64]
        int *    perm = nullptr;   // [mc*4]
        int4 *   tiles = nullptr;  // [mc*4/PF_MOE_BN + 64]
        int *    ntiles = nullptr;
        // int8 attention scores (k_pf_attn8; LK_PF_ATTN8=0 restores k_pf_attn).
        // The int8 K is a prefill-only copy of the fp16 cache, valid for
        // positions [0, k8_valid).  Key tiles within a8near of the query block,
        // and tile 0 (a8sink), are scored in fp16 exactly as k_pf_attn does.
        // Measured on real activations (LK_PF_ATTN_CMP, 32k prompt): attention
        // output error 0.26% rel. RMS against k_pf_attn, where the Q8_1 rounding
        // the next GEMM applies to that output is already 0.63%.  Smoothing
        // (LK_PF_KSMOOTH=1) measured slightly WORSE on real keys, so it is off.
        int      attn8 = 1, ksmooth = 0, a8near = 1024, a8sink = 1, k8_valid = 0, kstr = 0;   // kstr: ksc row stride, max_seq rounded up to 4
        int8_t * q8 = nullptr;     // [mc][2048]
        float *  qsc = nullptr;    // [mc][32]
        float *  qcs = nullptr;    // [mc][32] smoothing shift q.c
        int8_t * k8 = nullptr;     // [10][max_seq][8][64]
        float *  ksc = nullptr;    // [10][8][max_seq] (+4 pad)
        float *  kmean = nullptr;  // [10][8][64]
    } pf;
};

static inline const void * P(const Engine & e, const LKTensor * t, int plane) {
    return (const char *)e.d_pack + t->plane[plane];
}
static inline const float *   P32(const Engine & e, const LKTensor * t) { return (const float *)P(e, t, 0); }
static inline const uint4 *   PQS(const Engine & e, const LKTensor * t) { return (const uint4 *)P(e, t, 0); }
static inline const __half *  PD (const Engine & e, const LKTensor * t) { return (const __half *)P(e, t, 1); }

static inline int grid_for(int rows) { return (rows + 7) / 8; }   // 8 warps/block

// Phase 4y: descriptor for the mixer output projection, which is also the
// producer of this layer's ffn_norm.  h32 is the router's input and exists only
// on MoE layers; the router is the one consumer that needs f32 rather than Q8_1.
#if LK_NORM_PROD_GEMV
static inline LKNormProd ffn_prod(Engine & e, const Layer & l) {
    LKNormProd np;
    np.w   = LK_FUSE_FFN_NORM(l) ? P32(e, l.ffn_norm) : nullptr;
    np.h32 = (LK_FUSE_FFN_NORM(l) && l.is_moe) ? e.h : nullptr;
    // xq2, NOT xq.  This kernel reads its activation from xq and writes its
    // quantised output one block per thread block, so targeting xq would let
    // block 0 overwrite a block of activations that every other block is still
    // reading.  That race is silent, localised, and produced a model that
    // repeated one token forever -- caught by tools/ab.py, not by any gate.
    np.yq8 = e.xq2;
    np.ssq = e.norm_ssq;
    return np;
}
#endif

// Phase 4x: the FFN norm that follows the mixer's output projection, described
// as an epilogue for that projection to run for free.  See kernels.cuh.
#if LK_FUSE_NORM
static inline LKNormEpilogue ffn_norm_epi(Engine & e, const Layer & l) {
    LKNormEpilogue ne;
    ne.w       = P32(e, l.ffn_norm);
    ne.eps     = LK_EPS;
    ne.y32     = l.is_moe ? e.h : nullptr;   // the router's input, MoE layers only
    ne.yq8     = e.xq;
    ne.partial = e.norm_partial;
    ne.ctr     = e.norm_ctr;
    return ne;
}
#endif

// ---------------------------------------------------------------------------
static void load_layers(Engine & e) {
    char nm[128];
    for (int i = 0; i < LK_N_LAYERS; ++i) {
        Layer & l = e.L[i];
        snprintf(nm, sizeof nm, "blk.%d.attn_norm.weight", i); l.attn_norm = lk_find(nm);
        snprintf(nm, sizeof nm, "blk.%d.ffn_norm.weight", i);  l.ffn_norm  = lk_find(nm);
        if (!l.attn_norm || !l.ffn_norm) { fprintf(stderr, "missing norms for layer %d\n", i); exit(1); }
        l.is_attn = (i % 4 == 2);
        l.is_moe  = (i >= 2);
        if (l.is_attn) {
            snprintf(nm, sizeof nm, "blk.%d.attn_q.weight", i);       l.wq = lk_find(nm);
            snprintf(nm, sizeof nm, "blk.%d.attn_k.weight", i);       l.wk = lk_find(nm);
            snprintf(nm, sizeof nm, "blk.%d.attn_v.weight", i);       l.wv = lk_find(nm);
            snprintf(nm, sizeof nm, "blk.%d.attn_output.weight", i);  l.wo = lk_find(nm);
            snprintf(nm, sizeof nm, "blk.%d.attn_q_norm.weight", i);  l.qnorm = lk_find(nm);
            snprintf(nm, sizeof nm, "blk.%d.attn_k_norm.weight", i);  l.knorm = lk_find(nm);
            if (!l.wq || !l.wk || !l.wv || !l.wo || !l.qnorm || !l.knorm) {
                fprintf(stderr, "missing attn tensors in layer %d\n", i); exit(1); }
        } else {
            snprintf(nm, sizeof nm, "blk.%d.shortconv.conv.weight", i);     l.sc_conv = lk_find(nm);
            snprintf(nm, sizeof nm, "blk.%d.shortconv.in_proj.weight", i);  l.sc_in   = lk_find(nm);
            snprintf(nm, sizeof nm, "blk.%d.shortconv.out_proj.weight", i); l.sc_out  = lk_find(nm);
            if (!l.sc_conv || !l.sc_in || !l.sc_out) {
                fprintf(stderr, "missing shortconv tensors in layer %d\n", i); exit(1); }
        }
        if (l.is_moe) {
            snprintf(nm, sizeof nm, "blk.%d.ffn_gate_inp.weight", i);   l.router = lk_find(nm);
            snprintf(nm, sizeof nm, "blk.%d.exp_probs_b.bias", i);      l.rbias  = lk_find(nm);
            snprintf(nm, sizeof nm, "blk.%d.ffn_gate_exps.weight", i);  l.eg = lk_find(nm);
            snprintf(nm, sizeof nm, "blk.%d.ffn_up_exps.weight", i);    l.eu = lk_find(nm);
            snprintf(nm, sizeof nm, "blk.%d.ffn_down_exps.weight", i);  l.ed = lk_find(nm);
            if (!l.router || !l.rbias || !l.eg || !l.eu || !l.ed) {
                fprintf(stderr, "missing MoE tensors in layer %d\n", i); exit(1); }
        } else {
            snprintf(nm, sizeof nm, "blk.%d.ffn_gate.weight", i); l.ffn_gate = lk_find(nm);
            snprintf(nm, sizeof nm, "blk.%d.ffn_up.weight", i);   l.ffn_up   = lk_find(nm);
            snprintf(nm, sizeof nm, "blk.%d.ffn_down.weight", i); l.ffn_down = lk_find(nm);
            if (!l.ffn_gate || !l.ffn_up || !l.ffn_down) {
                fprintf(stderr, "missing dense FFN tensors in layer %d\n", i); exit(1); }
        }
    }
    e.embd = lk_find("token_embd.weight");
    e.out_norm = lk_find("token_embd_norm.weight");
    if (!e.embd || !e.out_norm) { fprintf(stderr, "missing embedding/norm\n"); exit(1); }
}

static Engine * engine_create(const char * pack_path, int max_seq) {
    Engine * e = new Engine();
    e->max_seq = max_seq;
    int fd = open(pack_path, O_RDONLY);
    if (fd < 0) { perror("open pack"); exit(1); }
    struct stat st{};
    if (fstat(fd, &st) != 0) { perror("fstat"); exit(1); }
    e->pack_size = st.st_size;
    e->pack = mmap(nullptr, e->pack_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (e->pack == MAP_FAILED) { perror("mmap"); exit(1); }
    close(fd);

    // Stage the whole pack into device memory.  This is a one-time cost
    // (~0.6 s over PCIe gen4 x16) paid once at load, and it is what makes the
    // 1008 GB/s HBM roofline reachable at all.
    fprintf(stderr, "copying %.2f GiB of weights to the device ...\n", e->pack_size / 1073741824.0);
    CK(cudaMalloc(&e->d_pack, e->pack_size));
    CK(cudaMemcpy(e->d_pack, e->pack, e->pack_size, cudaMemcpyHostToDevice));
    fprintf(stderr, "weights resident on device\n");
    load_layers(*e);

    const size_t KV = (size_t)10 * max_seq * LK_N_KV_HEADS * LK_HEAD_DIM;
    CK(cudaMalloc(&e->x,  LK_HIDDEN * 4));
    CK(cudaMalloc(&e->h,  LK_HIDDEN * 4));
    CK(cudaMalloc(&e->h16, LK_HIDDEN * 2));
    CK(cudaMalloc(&e->xq,  368 * sizeof(block_q8_1)));
    CK(cudaMalloc(&e->xq2, 368 * sizeof(block_q8_1)));
    CK(cudaMalloc(&e->norm_partial, LK_NORM_PARTIALS * 4));
    // Phase 4y: fully overwritten by every producer call, so it needs no
    // initialisation and has no reset hazard.
    CK(cudaMalloc(&e->norm_ssq, LK_NORM_BLOCKS * 4));
    CK(cudaMalloc(&e->norm_ctr, 4));
    // Must start at zero: the fused epilogue is skipped, silently and with
    // plausible output, if the counter is stale.
    CK(cudaMemset(e->norm_ctr, 0, 4));
    CK(cudaMalloc(&e->router_ctr, 4));
    CK(cudaMemset(e->router_ctr, 0, 4));
    CK(cudaMalloc(&e->y,  2 * LK_DENSE_INTER * 4));
    CK(cudaMalloc(&e->inter, LK_DENSE_INTER * 4));
    CK(cudaMalloc(&e->tmp, LK_HIDDEN * 4));
    CK(cudaMalloc(&e->qv, LK_HIDDEN * 4));
    CK(cudaMalloc(&e->kv, 512 * 4));
    CK(cudaMalloc(&e->vv, 512 * 4));
    CK(cudaMalloc(&e->logits, (size_t)LK_VOCAB * 4));
    CK(cudaMalloc(&e->d_token, 4));
    CK(cudaMalloc(&e->d_pos, 4));
    CK(cudaMemset(e->d_pos, 0, 4));
    CK(cudaMalloc(&e->conv_hist, (size_t)30 * LK_HIDDEN * 2 * 4));
    CK(cudaMalloc(&e->kcache, KV * 2));
    CK(cudaMalloc(&e->vcache, KV * 2));
    CK(cudaMalloc(&e->moe_gu, 2 * LK_TOP_K * LK_MOE_INTER * 4));
    CK(cudaMalloc(&e->moe_inter4, LK_TOP_K * LK_MOE_INTER * 4));
    CK(cudaMalloc(&e->moe_down4, LK_TOP_K * LK_HIDDEN * 4));
    CK(cudaMalloc(&e->moe_gate, LK_MOE_INTER * 4));
    CK(cudaMalloc(&e->moe_up,   LK_MOE_INTER * 4));
    CK(cudaMalloc(&e->moe_inter, LK_MOE_INTER * 4));
    CK(cudaMalloc(&e->moe_down, LK_HIDDEN * 4));
    if (getenv("LK_ROUTE_LOG"))
        CK(cudaMalloc(&e->d_routelog, (size_t) e->max_seq * LK_N_LAYERS * LK_TOP_K * 4));
    CK(cudaMalloc(&e->attn_part,
                  (size_t) LK_N_HEADS * LK_ATTN_SPLIT * LK_ATTN_PARTW * 4));
    CK(cudaMalloc(&e->d_gate, LK_N_EXPERTS * 4));
    CK(cudaMalloc(&e->d_sel, LK_N_EXPERTS * 4));
    CK(cudaMalloc(&e->d_sig, LK_N_EXPERTS * 4));
    if (const char * pfb = getenv("LK_PF_BYTES")) e->pf_bytes = atol(pfb);
    CK(cudaMalloc(&e->d_apv, LK_ARGMAX_BLOCKS * 4));
    CK(cudaMalloc(&e->d_api, LK_ARGMAX_BLOCKS * 4));
    CK(cudaMalloc(&e->d_idx, LK_TOP_K * 4));
    CK(cudaMalloc(&e->d_w, LK_TOP_K * 4));
    // d_idx feeds an address computation (base = d_idx[k] * rows_per_expert *
    // bpr) in the expert GEMVs.  It is written by the router before it is read
    // on every real path, but leaving it uninitialised means any path that
    // skips the router -- an ablation build, a future early exit -- reads
    // arbitrary memory.  Zeroing it costs one call at startup.
    CK(cudaMemset(e->d_idx, 0, LK_TOP_K * 4));
    CK(cudaMemset(e->d_w, 0, LK_TOP_K * 4));
    CK(cudaMemset(e->conv_hist, 0, (size_t)30 * LK_HIDDEN * 2 * 4));
    CK(cudaMemset(e->kcache, 0, KV * 2));
    CK(cudaMemset(e->vcache, 0, KV * 2));
    return e;
}


// Debug taps (LK_DBG_IL=<layer>, LK_DBG_OUT=<file>): append named f32 arrays
// for ONE layer of the FIRST token through either path, so the decode graph
// and the batched prefill can be diffed stage by stage.  Needs LK_NO_GRAPH on
// the decode side.  Format per record: 16-byte name, int32 count, floats.
static int    g_dbg_il = -2;
static FILE * g_dbg_f  = nullptr;
static int    g_dbg_done = 0;
static void dbg_tap(cudaStream_t st, int il, const char * name, const float * dptr, int n) {
    if (g_dbg_il == -2) {
        g_dbg_il = getenv("LK_DBG_IL") ? atoi(getenv("LK_DBG_IL")) : -1;
        if (g_dbg_il >= 0 && getenv("LK_DBG_OUT")) g_dbg_f = fopen(getenv("LK_DBG_OUT"), "wb");
    }
    if (!g_dbg_f || il != g_dbg_il || g_dbg_done) return;
    cudaStreamCaptureStatus cs;
    CK(cudaStreamIsCapturing(st, &cs));
    if (cs != cudaStreamCaptureStatusNone) return;    // the decode graph is being recorded
    std::vector<float> h(n);
    CK(cudaStreamSynchronize(st));
    CK(cudaMemcpy(h.data(), dptr, (size_t) n * 4, cudaMemcpyDeviceToHost));
    char nm[16] = {0}; strncpy(nm, name, 15);
    fwrite(nm, 1, 16, g_dbg_f); fwrite(&n, 4, 1, g_dbg_f); fwrite(h.data(), 4, n, g_dbg_f); fflush(g_dbg_f);
    if (!strcmp(name, "x_ffn")) g_dbg_done = 1;      // the layer's last tap
}
// ---------------------------------------------------------------------------
// One decode step: token -> next token.
// Position-independent: `token` and `pos` live in device memory, so the
// entire 40-layer step can be captured once into a cudaGraphExec_t and replayed
// for every subsequent position.
static void forward(Engine & e, cudaStream_t st) {
    LKP_B("embed");
    k_embed_q6_K<<<LK_HIDDEN / 256, 256, 0, st>>>(
        (const uint8_t *)P(e, e.embd, 0), (const uint8_t *)P(e, e.embd, 1),
        (const int8_t *)P(e, e.embd, 2), (const __half *)P(e, e.embd, 3),
        e.d_token, e.x); LKP_E();
    CK(cudaGetLastError());

    static FILE * dumpf = nullptr;
    static bool dump_init = false;
    if (!dump_init) {
        dump_init = true;
        const char * dp = getenv("LK_DUMP_LAYERS");
        if (dp) dumpf = fopen(dp, "wb");
    }
    int conv_i = 0, attn_i = 0;
    for (int il = 0; il < LK_N_LAYERS; ++il) {
        const Layer & l = e.L[il];
        // Phase 4y: layer il's attn_norm is folded into layer il-1's moe_accum4,
        // which is the kernel that writes the residual it would reduce.  Layers
        // 0..2 keep their norm node: 0 comes from the embedding and 1..2 from
        // the dense_down GEMV, which needs the LK_GEMV_BIGBLOCK shape to own a
        // whole Q8_1 block and is not converted yet.
        // Phase 4y.  attn_norm_ssq is non-null exactly when layer il's
        // attn_norm was folded into the kernel that produced this residual.
        const float * attn_norm_ssq = LK_FUSE_ATTN_NORM(il) ? e.norm_ssq : nullptr;
        const float * ffn_norm_ssq  = LK_FUSE_FFN_NORM(l)  ? e.norm_ssq : nullptr;
#if LK_NORM_PROD_GEMV
        // The mixer's output projection quantises its own output, and it cannot
        // write into the xq it is still reading, so the POST-mixer activation
        // lives in xq2 and everything downstream of the mixer shares it: the
        // FFN gate/up, the expert/dense intermediate, and the down projection.
        // They are strictly sequential, and the norm's output is dead the
        // moment gate_up has read it.
        block_q8_1 * const ffn_xq = e.xq2;
#else
        // norm_ffn is a node and writes xq itself; nothing aliases.
        block_q8_1 * const ffn_xq = e.xq;
#endif
        if (dumpf) {   // residual stream BEFORE layer il
            float buf[LK_HIDDEN];
            // cudaMemcpy is only ordered against the DEFAULT stream; our kernels
            // run on `st`. Without this sync the copy can race with in-flight
            // work and capture a partially-written e.x.
            CK(cudaStreamSynchronize(st));
            CK(cudaMemcpy(buf, e.x, LK_HIDDEN * 4, cudaMemcpyDeviceToHost));
            fwrite(buf, 4, LK_HIDDEN, dumpf);
        }

        // ---- sequence mixer -------------------------------------------
        if (!attn_norm_ssq) {
        LKP_B("norm_attn");
    // dead outputs: nothing reads the f32 or fp16 form of the PRE-mixer norm
    // (the router consumes the POST-ffn_norm one), so skip 12 KB of stores per
    // call.  The kernel already branches on null.
    k_rmsnorm_q8_1<<<1, 256, LK_HIDDEN * 4, st>>>(e.x, P32(e, l.attn_norm), LK_HIDDEN, LK_EPS,
                                              nullptr, nullptr, e.xq); LKP_E();
        }
        LK_STEP("mixer-norm");

        if (!l.is_attn) {
            float * hist = e.conv_hist + (size_t)conv_i * LK_HIDDEN * 2;
            ++conv_i;
            LKP_B("shortconv_in");
    k_gemv_q4_0<<<LK_GEMV_BLOCKS(6144), 256, 0, st>>>(
                PQS(e, l.sc_in), PD(e, l.sc_in), e.xq, 6144, 64, e.y, 0,
                attn_norm_ssq, LK_EPS); LKP_E();
            if (il == 0 && getenv("LK_DUMP_L0")) {   // bisect layer 0 tensors
                const char * dp = getenv("LK_DUMP_L0");
                FILE * df = fopen(dp, "wb");
                float th[LK_HIDDEN]; int8_t tq[64 * sizeof(block_q8_1)]; float tb[6144], tc[LK_HIDDEN];
                CK(cudaStreamSynchronize(st)); CK(cudaMemcpy(th, e.h, LK_HIDDEN * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(tq, e.xq, 64 * (int)sizeof(block_q8_1), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(tb, e.y, 6144 * 4, cudaMemcpyDeviceToHost));
                fwrite(th, 4, LK_HIDDEN, df);
                fwrite(tq, 1, 64 * sizeof(block_q8_1), df);
                fwrite(tb, 4, 6144, df);
                CK(cudaMemcpy(tc, e.tmp, LK_HIDDEN * 4, cudaMemcpyDeviceToHost));  // pre-conv scratch
                fclose(df);
                fprintf(stderr, "dumped layer-0 h/xq/bcx to %s\n", dp);
            }
            dbg_tap(st, il, "in_proj", e.y, 6144);
            LKP_B("shortconv_conv");
    k_shortconv<<<8, 256, 0, st>>>(e.y, 0, P32(e, l.sc_conv), hist, e.tmp, e.xq); LKP_E();
            if (il == 0 && getenv("LK_DUMP_L0B")) {   // conv output before quantise
                const char * dp = getenv("LK_DUMP_L0B");
                FILE * df = fopen(dp, "wb");
                float ty[LK_HIDDEN];
                CK(cudaStreamSynchronize(st)); CK(cudaMemcpy(ty, e.tmp, LK_HIDDEN * 4, cudaMemcpyDeviceToHost));
                fwrite(ty, 4, LK_HIDDEN, df);
                fclose(df);
            }
            
            if (il == 0 && getenv("LK_DUMP_L0Q")) {   // q8 of the conv output
                const char * dp = getenv("LK_DUMP_L0Q");
                FILE * df = fopen(dp, "wb");
                int8_t tq[64 * (int)sizeof(block_q8_1)];
                CK(cudaStreamSynchronize(st)); CK(cudaMemcpy(tq, e.xq, sizeof tq, cudaMemcpyDeviceToHost));
                fwrite(tq, 1, sizeof tq, df);
                fclose(df);
            }
            // LK_NO_RESIDUAL forces add_residual=0 for layer 0's out_proj so the
            // raw projection can be compared in isolation (debug only).
            dbg_tap(st, il, "conv_out", e.tmp, 2048);
            const int ar = (il == 0 && getenv("LK_NO_RESIDUAL")) ? 0 : 1;
            LKP_B("shortconv_out");
#if LK_NORM_PROD_GEMV
    // Phase 4y: this GEMV writes exactly the 2048 residual values that this
    // layer's ffn_norm reduces, so it quantises them itself.  No barrier, no
    // counter -- see kernels.cuh.
    k_gemv_q4_0_prod<<<LK_GEMV_WBLOCKS(LK_HIDDEN), LK_GEMV_WTHREADS, 0, st>>>(
                PQS(e, l.sc_out), PD(e, l.sc_out), e.xq, LK_HIDDEN, 64, e.x, ar,
                ffn_prod(e, l)); LKP_E();
#elif LK_FUSE_NORM
    // Superseded by LK_NORM_PROD_GEMV, which gets the same result without the
    // last-block barrier.  Kept so the negative result stays reproducible.
    k_gemv_q4_0_norm<<<LK_GEMV_BLOCKS(LK_HIDDEN), 256, 0, st>>>(
                PQS(e, l.sc_out), PD(e, l.sc_out), e.xq, LK_HIDDEN, 64, e.x, ar,
                ffn_norm_epi(e, l)); LKP_E();
#else
    k_gemv_q4_0<<<LK_GEMV_WBLOCKS(LK_HIDDEN), LK_GEMV_WTHREADS, 0, st>>>(
                PQS(e, l.sc_out), PD(e, l.sc_out), e.xq, LK_HIDDEN, 64, e.x, ar,
                nullptr, 0.0f); LKP_E();
#endif
            if (il == 0 && getenv("LK_DUMP_L0X")) {   // immediately after the GEMV
                const char * dp = getenv("LK_DUMP_L0X");
                FILE * df = fopen(dp, "wb");
                float tx[LK_HIDDEN];
                CK(cudaStreamSynchronize(st));
                CK(cudaMemcpy(tx, e.x, LK_HIDDEN * 4, cudaMemcpyDeviceToHost));
                fwrite(tx, 4, LK_HIDDEN, df);
                fclose(df);
            }
            dbg_tap(st, il, "x_mixer", e.x, 2048);
            LK_STEP("shortconv");
        } else {
            const int al = attn_i++;
            LKP_B("attn_qkv");
    k_gemv_q4_0_3<<<LK_GEMV_BLOCKS(LK_GEMV_ROWS * (LK_HIDDEN + 1024)), 256, 0, st>>>(
                PQS(e, l.wq), PD(e, l.wq), LK_HIDDEN, e.qv,
                PQS(e, l.wk), PD(e, l.wk), 512, e.kv,
                PQS(e, l.wv), PD(e, l.wv), 512, e.vv,
                e.xq, 64, attn_norm_ssq, LK_EPS); LKP_E();
            __half * kc = e.kcache + (size_t)al * e.max_seq * LK_N_KV_HEADS * LK_HEAD_DIM;
            __half * vc = e.vcache + (size_t)al * e.max_seq * LK_N_KV_HEADS * LK_HEAD_DIM;
            LKP_B("qk_norm_rope");
    k_qk_norm_rope<<<LK_N_HEADS + LK_N_KV_HEADS, LK_HEAD_DIM, 0, st>>>(
                e.qv, e.kv, e.vv, P32(e, l.qnorm), P32(e, l.knorm),
                e.d_pos, kc, vc); LKP_E();
            LKP_B("attention");
    // LK_ATTN_DUAL (diagnostic, needs LK_NO_GRAPH): run BOTH decode attention
    // kernels on the same q and KV cache and report how far apart they are, per
    // layer, on real activations rather than on test-attn's random ones.
    static const bool attn_dual = getenv("LK_ATTN_DUAL") != nullptr;
    if (attn_dual) {
        static float * dual = nullptr;
        if (!dual) CK(cudaMalloc(&dual, LK_HIDDEN * 4));
        k_attn_split<<<dim3(LK_N_KV_HEADS, LK_ATTN_SPLIT), 256, 0, st>>>(e.qv, kc, vc, e.d_pos, e.attn_part);
        k_attn_combine<<<LK_N_HEADS, LK_HEAD_DIM, 0, st>>>(e.attn_part, e.d_pos, e.tmp, nullptr);
        k_attention_decode<<<LK_N_HEADS, LK_ATTN_BLOCK, 0, st>>>(e.qv, kc, vc, e.d_pos, dual, nullptr);
        CK(cudaStreamSynchronize(st));
        float a[LK_HIDDEN], b[LK_HIDDEN]; int pos = 0;
        CK(cudaMemcpy(a, e.tmp, sizeof a, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b, dual, sizeof b, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&pos, e.d_pos, 4, cudaMemcpyDeviceToHost));
        double md = 0.0, mx = 0.0; int wh = -1;
        for (int i = 0; i < LK_HIDDEN; ++i) {
            const double d = fabs((double) a[i] - b[i]);
            if (d > md) { md = d; wh = i / LK_HEAD_DIM; }
            mx = std::max(mx, (double) fabs(b[i]));
        }
        fprintf(stderr, "ATTN_DUAL pos=%d layer=%d max|split-serial|=%.3e (head %d) max|serial|=%.3e\n",
                pos, il, md, wh, mx);
    }
    if (getenv("LK_ATTN_SERIAL")) {
        k_attention_decode<<<LK_N_HEADS, LK_ATTN_BLOCK, 0, st>>>(
                    e.qv, kc, vc, e.d_pos, e.tmp, e.xq);
    } else {
        k_attn_split<<<dim3(LK_N_KV_HEADS, LK_ATTN_SPLIT), 256, 0, st>>>(
                    e.qv, kc, vc, e.d_pos, e.attn_part);
        k_attn_combine<<<LK_N_HEADS, LK_HEAD_DIM, 0, st>>>(
                    e.attn_part, e.d_pos, e.tmp, e.xq);
    }
    LKP_E();
            
            LKP_B("attn_out");
#if LK_NORM_PROD_GEMV
    k_gemv_q4_0_prod<<<LK_GEMV_WBLOCKS(LK_HIDDEN), LK_GEMV_WTHREADS, 0, st>>>(
                PQS(e, l.wo), PD(e, l.wo), e.xq, LK_HIDDEN, 64, e.x, 1,
                ffn_prod(e, l)); LKP_E();
#elif LK_FUSE_NORM
    k_gemv_q4_0_norm<<<LK_GEMV_BLOCKS(LK_HIDDEN), 256, 0, st>>>(
                PQS(e, l.wo), PD(e, l.wo), e.xq, LK_HIDDEN, 64, e.x, 1,
                ffn_norm_epi(e, l)); LKP_E();
#else
    k_gemv_q4_0<<<LK_GEMV_WBLOCKS(LK_HIDDEN), LK_GEMV_WTHREADS, 0, st>>>(
                PQS(e, l.wo), PD(e, l.wo), e.xq, LK_HIDDEN, 64, e.x, 1,
                nullptr, 0.0f); LKP_E();
#endif
            LK_STEP("attention");
        }

        // ---- channel mixer --------------------------------------------
#if (LK_FUSE_NORM || LK_NORM_PROD_GEMV)
        // The norm already ran inside whichever projection produced e.x
        // (shortconv_out or attn_out, above).  Deliberately NO LKP_B/LKP_E here:
        // an empty bracket pair costs ~1.3 us of measured time and shows up in
        // the profile as a `norm_ffn` row for a kernel that does not exist --
        // the profiler then reports the fused build as ~0.05 ms slower than it
        // is.  The LK_STEP below still covers the phase in the debug trace.
#else
        LKP_B("norm_ffn");
    // the f32 form is the router's input and exists only in MoE layers; the
    // fp16 form is never read anywhere in the engine.
    k_rmsnorm_q8_1<<<1, 256, LK_HIDDEN * 4, st>>>(e.x, P32(e, l.ffn_norm), LK_HIDDEN, LK_EPS,
                                              l.is_moe ? e.h : nullptr, nullptr, e.xq); LKP_E();
#endif
        LK_STEP("ffn-norm");

        if (!l.is_moe) {
            LKP_B("dense_gate");
    k_gemv_q4_0<<<LK_GEMV_BLOCKS(LK_DENSE_INTER), 256, 0, st>>>(
                PQS(e, l.ffn_gate), PD(e, l.ffn_gate), ffn_xq, LK_DENSE_INTER, 64, e.y, 0,
                ffn_norm_ssq, LK_EPS); LKP_E();
            LKP_B("dense_up");
    k_gemv_q4_0<<<LK_GEMV_BLOCKS(LK_DENSE_INTER), 256, 0, st>>>(
                PQS(e, l.ffn_up), PD(e, l.ffn_up), ffn_xq, LK_DENSE_INTER, 64, e.y + LK_DENSE_INTER, 0,
                ffn_norm_ssq, LK_EPS); LKP_E();
            dbg_tap(st, il, "gate_up", e.y, 2 * LK_DENSE_INTER);
            LKP_B("dense_swiglu");
    k_swiglu<<<64, 256, 0, st>>>(e.y, e.y + LK_DENSE_INTER, LK_DENSE_INTER, e.inter, ffn_xq); LKP_E();
            if (il == 0 && getenv("LK_DUMP_FFN")) {   // dense-FFN intermediates
                const char * dp = getenv("LK_DUMP_FFN");
                FILE * df = fopen(dp, "wb");
                float th2[LK_HIDDEN], tg[LK_DENSE_INTER], ti[LK_DENSE_INTER], tx[LK_HIDDEN];
                int8_t tq[64 * (int)sizeof(block_q8_1)];
                CK(cudaStreamSynchronize(st));
                CK(cudaMemcpy(th2, e.h, LK_HIDDEN * 4, cudaMemcpyDeviceToHost));
                fwrite(th2, 4, LK_HIDDEN, df);        // h2 = ffn_norm output
                CK(cudaMemcpy(tq, e.xq, sizeof tq, cudaMemcpyDeviceToHost));
                fwrite(tq, 1, sizeof tq, df);         // its Q8_1 blocks
                CK(cudaMemcpy(tg,  e.y, LK_DENSE_INTER * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(ti,  e.y + LK_DENSE_INTER, LK_DENSE_INTER * 4, cudaMemcpyDeviceToHost));
                fwrite(ti, 4, LK_DENSE_INTER, df);   // up (before swiglu overwrote it)
                CK(cudaMemcpy(ti,  e.inter, LK_DENSE_INTER * 4, cudaMemcpyDeviceToHost));
                fwrite(tg, 4, LK_DENSE_INTER, df);   // gate
                fwrite(ti, 4, LK_DENSE_INTER, df);   // swiglu intermediate
                (void)tx;
                fclose(df);
                fprintf(stderr, "dumped layer-0 dense-FFN gate+up+inter\n");
            }
            
            LKP_B("dense_down");
#if LK_NORM_PROD_GEMV
    // Also the producer of attn_norm(il+1) -- for layers 0 and 1, whose
    // successor's norm would otherwise be a node.  Dense layers have no router,
    // so no f32 h32 output is needed here.
    {
        LKNormProd np;
        np.w   = LK_FUSE_ATTN_NORM_G(il + 1) ? P32(e, e.L[il + 1].attn_norm) : nullptr;
        np.h32 = nullptr;
        np.yq8 = e.xq;
        np.ssq = e.norm_ssq;
        k_gemv_q4_0_prod<<<LK_GEMV_WBLOCKS(LK_HIDDEN), LK_GEMV_WTHREADS, 0, st>>>(
                PQS(e, l.ffn_down), PD(e, l.ffn_down), ffn_xq, LK_HIDDEN, 368, e.x, 1,
                np); LKP_E();
    }
#else
    k_gemv_q4_0<<<LK_GEMV_WBLOCKS(LK_HIDDEN), LK_GEMV_WTHREADS, 0, st>>>(
                PQS(e, l.ffn_down), PD(e, l.ffn_down), ffn_xq, LK_HIDDEN, 368, e.x, 1,
                nullptr, 0.0f); LKP_E();
#endif
            dbg_tap(st, il, "x_ffn", e.x, 2048);
            LK_STEP("dense-ffn");
        } else {
            // The router consumes the POST-ffn_norm hidden state in FP32:
            // llama.cpp's ggml_rms_norm emits f32 and ffn_gate_inp is F32, so
            // build_lora_mm runs f32 x f32.  (The experts, by contrast, take
            // the Q8_1-quantised activation.)
            LKP_B("router_dot");
    // T4r.  k_router_dot now also emits the two 64-float arrays the selection
    // needs -- `sel` = sigmoid + bias (what the RANKING uses) and `sig` = the
    // unbiased sigmoid (what the WEIGHTS use).  The expf happens here, once per
    // expert, because this is the only kernel that has the logit in a register.
    // k_router_top4 is gone: every block of moe_gate_up rebuilds the top-4 from
    // those two arrays in its own prologue, with no barrier.
#if LK_ROUTER_PRO
    k_router_dot<<<LK_N_EXPERTS, 32, 0, st>>>(e.h, P32(e, l.router), e.d_gate,
                                              P32(e, l.rbias), e.d_sel, e.d_sig,
                                              ffn_norm_ssq, LK_EPS); LKP_E();
#else
    // T4r is OFF by default: measured 0.17% SLOWER.  See LK_ROUTER_PRO.
    k_router_dot<<<LK_N_EXPERTS, 32, 0, st>>>(e.h, P32(e, l.router), e.d_gate,
                                              P32(e, l.rbias), nullptr, nullptr,
                                              ffn_norm_ssq, LK_EPS); LKP_E();
            LKP_B("router_top4");
    // one lane per expert: the sigmoids are computed cooperatively, so a
    // 32-thread block would leave experts 32..63 uninitialised.
    k_router_top4<<<1, LK_N_EXPERTS, 0, st>>>(e.d_gate, P32(e, l.rbias),
                                    e.d_idx, e.d_w); LKP_E();
#endif
            // Phase 4: ONE launch per matrix for all 4 experts (grid.y = expert),
            // replacing 16 launches per MoE layer.  Weights stay on device, so the
            // per-layer cudaStreamSynchronize host round-trip is gone too.
            const int GU = LK_TOP_K * LK_MOE_INTER;
            LKP_B("moe_gate_up");
    k_gemv_q4_0_exp2<<<dim3(grid_for(LK_MOE_INTER), 2 * LK_TOP_K), 256, 0, st>>>(
                PQS(e, l.eg), PD(e, l.eg), PQS(e, l.eu), PD(e, l.eu), ffn_xq,
                LK_MOE_INTER, 64, LK_MOE_INTER, 0,
                // Under T4r the d_idx parameter is superseded by the prologue's
                // own selection, and passing it twice would alias two
                // __restrict__ parameters -- so exactly one group is live.
#if LK_ROUTER_PRO
                nullptr, e.d_sel, e.d_sig, e.d_idx, e.d_w,
#else
                e.d_idx, nullptr, nullptr, nullptr, nullptr,
#endif
                e.moe_gu, e.moe_gu + GU, ffn_norm_ssq, LK_EPS); LKP_E();
            if (e.d_routelog) {          // G4b instrumentation (off unless asked)
                // After moe_gate_up, which is what now writes d_idx.
                LKP_B("route_log");
    k_route_log<<<1, 32, 0, st>>>(e.d_idx, e.d_pos, il, e.d_routelog); LKP_E();
            }
            LKP_B("moe_swiglu");
    // The prefetch target is moe_down, whose experts are known by now.
    // PQS(e, l.ed) is the qs plane; geometry matches the k_moe_down_accum call
    // below (rows_per_expert = LK_HIDDEN, bpr = 48).
    k_swiglu<<<48, 256, 0, st>>>(e.moe_gu, e.moe_gu + GU, GU, nullptr, e.xq2,
                                 PQS(e, l.ed), e.d_idx, LK_HIDDEN, 48,
                                 e.pf_bytes); LKP_E();
            
            // down: each expert consumes its OWN 48-block activation slice.
            // T1: this node also now does what k_moe_accum4 used to do -- weight
            // the four expert outputs, add the residual, and produce the NEXT
            // layer's attn_norm (Phase 4y).  The four outputs never leave the
            // kernel, so moe_down4 has left the graph entirely.
            LKP_B("moe_down");
    // Phase 4y: layer il's accumulator is also the producer of layer il+1's
    // attn_norm, and of norm_final at the last layer.  `np.w == nullptr` when
    // this layer's successor still has its own norm node (the LK_NORM_PROD_GEMV
    // half is off, so only MoE -> MoE transitions fuse).
    {
        const LKTensor * nw = (il + 1 < LK_N_LAYERS) ? e.L[il + 1].attn_norm : e.out_norm;
        LKNormProd np;
        np.w   = LK_FUSE_ATTN_NORM_A(il + 1) ? P32(e, nw) : nullptr;
        np.h32 = nullptr;                 // the router reads the POST-ffn norm
        np.yq8 = e.xq;
        np.ssq = e.norm_ssq;
    k_moe_down_accum<<<LK_HIDDEN / 32, 1024, 0, st>>>(
                PQS(e, l.ed), PD(e, l.ed), e.xq2, LK_HIDDEN, 48, LK_HIDDEN, 48,
                e.d_idx, e.d_w, e.x, np); LKP_E();
    }
        }
    }

    if (dumpf) {       // residual stream AFTER the last layer
        float buf[LK_HIDDEN];
        CK(cudaStreamSynchronize(st));
        CK(cudaMemcpy(buf, e.x, LK_HIDDEN * 4, cudaMemcpyDeviceToHost));
        fwrite(buf, 4, LK_HIDDEN, dumpf);
        fflush(dumpf);
    }

    // ---- final norm + tied Q6_K lm_head + argmax ----------------------
    // Phase 4y: norm_final is produced by the last layer's k_moe_accum4, so the
    // node disappears for the same reason attn_norm's does.
    if (!LK_FUSE_ATTN_NORM(LK_N_LAYERS)) {
    LKP_B("norm_final");
    k_rmsnorm_q8_1<<<1, 256, LK_HIDDEN * 4, st>>>(e.x, P32(e, e.out_norm), LK_HIDDEN, LK_EPS,
                                          nullptr, nullptr, e.xq); LKP_E();
    }
    const float * final_ssq = LK_FUSE_ATTN_NORM(LK_N_LAYERS) ? e.norm_ssq : nullptr;
    LKP_B("lm_head");
    // LK_Q6K_SCALAR keeps the original scalar kernel reachable so the two can be
    // compared on real weights without a rebuild.
    if (getenv("LK_Q6K_SCALAR"))
        k_gemv_q6_K<<<grid_for(LK_VOCAB), 256, 0, st>>>(
            (const uint8_t *)P(e, e.embd, 0), (const uint8_t *)P(e, e.embd, 1),
            (const int8_t *)P(e, e.embd, 2), (const __half *)P(e, e.embd, 3),
            e.xq, LK_VOCAB, 8, e.logits, final_ssq, LK_EPS);
    else
        k_gemv_q6_K_dp4a<<<grid_for(LK_VOCAB), 256, 0, st>>>(
            (const uint8_t *)P(e, e.embd, 0), (const uint8_t *)P(e, e.embd, 1),
            (const int8_t *)P(e, e.embd, 2), (const __half *)P(e, e.embd, 3),
            e.xq, LK_VOCAB, 8, e.logits, final_ssq, LK_EPS);
    LKP_E();
    LKP_B("argmax_part");
    k_argmax_part<<<LK_ARGMAX_BLOCKS, 256, 0, st>>>(e.logits, e.d_apv, e.d_api); LKP_E();
    LKP_B("argmax_final");
    k_argmax_final<<<1, 256, 0, st>>>(e.d_apv, e.d_api, LK_ARGMAX_BLOCKS,
                                      e.d_token, e.d_pos); LKP_E();
    CK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// P0: THE BATCHED-PREFILL SEAM.
//
// There is no batched prefill in this engine and never has been.  Prefill is a
// loop of one full decode step per prompt token, which is why ingesting 131072
// tokens costs 131072 sequential steps and why llama.cpp is 15.3x faster at it.
//
// This struct and the function below exist so that the loop in main() no longer
// knows how prefill actually happens.  It is deliberately a NO-OP refactor:
// prefill_forward still executes exactly one decode step per token, in order,
// with the same H2D of the same 4 bytes, so LK_PREFILL_B>1 changes the
// CHUNKING and nothing else.  That is the point -- it makes the bit-identity
// gate (tools/ab.py, 7/7) a proof about the plumbing rather than a hope, and it
// leaves one call site to replace when the real batched kernels land.
//
// It is a seam, not a design.  `nb` is currently ignored beyond being a loop
// bound; when the batched path exists it becomes the batch dimension, and the
// tokens being contiguous in a host vector is what makes a single H2D legal.
struct PrefillCtx {
    Engine *         e;
    cudaStream_t     st;
    cudaGraphExec_t  gexec;
    bool             direct;   // LK_NO_GRAPH or LK_PROFILE: call forward() directly
};

// Ingest `nb` tokens, in order.  Does NOT sync: the caller keeps the single
// sync that used to follow the whole prefill loop, so the number of
// synchronisations is unchanged as well.
static void prefill_forward(PrefillCtx & pc, const int * toks, int nb) {
    for (int i = 0; i < nb; ++i) {
        CK(cudaMemcpy(pc.e->d_token, toks + i, 4, cudaMemcpyHostToDevice));
        if (pc.direct) forward(*pc.e, pc.st);
        else           CK(cudaGraphLaunch(pc.gexec, pc.st));
    }
}

// ---------------------------------------------------------------------------
// THE BATCHED PREFILL (prefill.cuh).
//
// Ingests prompt positions [P, P+n) a chunk at a time: every layer of the chunk
// is a handful of launches over all of its tokens -- tensor-core GEMMs, a
// causal flash attention, and per-token kernels that carry the decode path's
// formulas -- instead of one decode step per token.  It writes exactly the
// state the decode graph reads afterwards: the KV cache at those positions and
// each shortconv layer's two-tap history.  It computes no logits; the caller
// runs the LAST prompt token through the decode graph, which produces d_token,
// the logits and the d_pos increment exactly as it does for every later token.
// LK_PF_PROF=1: per-stage GPU time of the batched prefill, aggregated by tag
// and reported once at the end.  Treat totals as slightly high: every event
// pair costs a timestamp on the stream.
struct PfProf { const char * tag; cudaEvent_t a, b; };
static std::vector<PfProf> g_pfp;
static int g_pfp_on = -1;
static void pfp_b(cudaStream_t st, const char * tag) {
    if (g_pfp_on < 0) g_pfp_on = getenv("LK_PF_PROF") ? 1 : 0;
    if (!g_pfp_on) return;
    PfProf p{ tag, nullptr, nullptr };
    CK(cudaEventCreate(&p.a)); CK(cudaEventCreate(&p.b));
    CK(cudaEventRecord(p.a, st));
    g_pfp.push_back(p);
}
static void pfp_e(cudaStream_t st) { if (g_pfp_on > 0) CK(cudaEventRecord(g_pfp.back().b, st)); }
static void pfp_report(int ntok) {
    if (g_pfp_on <= 0 || g_pfp.empty()) return;
    CK(cudaDeviceSynchronize());
    std::vector<std::pair<std::string, double>> agg;
    double tot = 0.0;
    for (auto & p : g_pfp) {
        float ms = 0.0f; CK(cudaEventElapsedTime(&ms, p.a, p.b));
        tot += ms;
        bool f = false;
        for (auto & a : agg) if (a.first == p.tag) { a.second += ms; f = true; }
        if (!f) agg.push_back({ p.tag, ms });
        cudaEventDestroy(p.a); cudaEventDestroy(p.b);
    }
    g_pfp.clear();
    std::sort(agg.begin(), agg.end(), [](const std::pair<std::string, double> & x,
                                         const std::pair<std::string, double> & y) { return x.second > y.second; });
    fprintf(stderr, "\n=== batched prefill, %d tokens: GPU time by stage ===\n", ntok);
    for (auto & a : agg)
        fprintf(stderr, "  %-14s %9.2f ms %6.1f%%  %7.2f us/token\n", a.first.c_str(), a.second,
                100.0 * a.second / tot, 1000.0 * a.second / ntok);
    fprintf(stderr, "  %-14s %9.2f ms         %7.2f us/token  (%.0f tok/s)\n", "TOTAL", tot,
            1000.0 * tot / ntok, ntok / (tot * 1e-3));
}

static void pf_alloc(Engine & e, int mc) {
    auto & p = e.pf;
    if (p.mc >= mc) return;
    if (p.mc) { fprintf(stderr, "pf_alloc: chunk size cannot grow after first use\n"); exit(1); }
    p.mc = mc;
    const size_t M = (size_t) mc;
    CK(cudaMalloc(&p.tok, M * 4));
    CK(cudaMalloc(&p.x,   M * LK_HIDDEN * 4));
    CK(cudaMalloc(&p.h32, M * LK_HIDDEN * 4));
    CK(cudaMalloc(&p.inv, M * 4));
    CK(cudaMalloc(&p.a.qs, M * LK_DENSE_INTER));
    CK(cudaMalloc(&p.a.sc, M * (LK_DENSE_INTER / 32) * sizeof(float2)));
    // a2 holds the SwiGLU output: [mc][11776] for the dense FFN, or
    // [mc*4][1536] (one row per MoE slot).  The dense case is the larger.
    static_assert(LK_DENSE_INTER >= LK_TOP_K * LK_MOE_INTER, "a2 sizing");
    CK(cudaMalloc(&p.a2.qs, M * LK_DENSE_INTER));
    CK(cudaMalloc(&p.a2.sc, M * (LK_DENSE_INTER / 32) * sizeof(float2)));
    CK(cudaMalloc(&p.y,  M * 2 * LK_DENSE_INTER * 4));
    CK(cudaMalloc(&p.o,  M * LK_HIDDEN * 4));
    CK(cudaMalloc(&p.qh, M * LK_HIDDEN * 2));
    CK(cudaMalloc(&p.sel, M * LK_N_EXPERTS * 4));
    CK(cudaMalloc(&p.sig, M * LK_N_EXPERTS * 4));
    CK(cudaMalloc(&p.idx, M * LK_TOP_K * 4));
    CK(cudaMalloc(&p.w,   M * LK_TOP_K * 4));
    CK(cudaMalloc(&p.cnt, LK_N_EXPERTS * 4));
    CK(cudaMalloc(&p.off, LK_N_EXPERTS * 4));
    CK(cudaMalloc(&p.perm, M * LK_TOP_K * 4));
    CK(cudaMalloc(&p.tiles, (M * LK_TOP_K / PF_MOE_BN + LK_N_EXPERTS) * sizeof(int4)));
    CK(cudaMalloc(&p.ntiles, 4));
    if (const char * v = getenv("LK_PF_ATTN8")) p.attn8 = atoi(v);
    if (const char * v = getenv("LK_PF_KSMOOTH")) p.ksmooth = atoi(v);
    if (const char * v = getenv("LK_PF_A8_NEAR")) p.a8near = atoi(v);
    if (const char * v = getenv("LK_PF_A8_SINK")) p.a8sink = atoi(v);
    if (p.attn8) {
        const size_t S = (size_t) e.max_seq;
        CK(cudaMalloc(&p.q8,  M * LK_HIDDEN));
        CK(cudaMalloc(&p.qsc, M * LK_N_HEADS * 4));
        CK(cudaMalloc(&p.qcs, M * LK_N_HEADS * 4));
        CK(cudaMalloc(&p.k8,  10 * S * LK_N_KV_HEADS * LK_HEAD_DIM));
        p.kstr = (e.max_seq + 3) & ~3;
        CK(cudaMalloc(&p.ksc, (10 * (size_t) p.kstr * LK_N_KV_HEADS + 4) * 4));
        CK(cudaMalloc(&p.kmean, 10 * LK_N_KV_HEADS * LK_HEAD_DIM * 4));
        CK(cudaMemset(p.ksc, 0, (10 * (size_t) p.kstr * LK_N_KV_HEADS + 4) * 4));
        CK(cudaMemset(p.kmean, 0, 10 * LK_N_KV_HEADS * LK_HEAD_DIM * 4));
    }
}

static PfGemm pf_g1(const Engine & e, const LKTensor * t, int N, int bpr) {
    PfGemm g{};
    g.qs[0] = g.qs[1] = g.qs[2] = PQS(e, t);
    g.d[0]  = g.d[1]  = g.d[2]  = PD(e, t);
    g.seg_end[0] = g.seg_end[1] = g.seg_end[2] = N;
    g.N = N; g.bpr = bpr;
    return g;
}

// One chunk of m tokens at positions [pos0, pos0+m).
static void pf_chunk(Engine & e, cudaStream_t st, const int * toks, int m, int pos0) {
    auto & p = e.pf;
    CK(cudaMemcpyAsync(p.tok, toks, (size_t) m * 4, cudaMemcpyHostToDevice, st));
    pfp_b(st, "embed");
    k_pf_embed<<<dim3(LK_HIDDEN / 256, m), 256, 0, st>>>(
        (const uint8_t *)P(e, e.embd, 0), (const uint8_t *)P(e, e.embd, 1),
        (const int8_t *)P(e, e.embd, 2), (const __half *)P(e, e.embd, 3), p.tok, p.x);
    pfp_e(st);
    CK(cudaGetLastError());
    const int max_tiles = m * LK_TOP_K / PF_MOE_BN + LK_N_EXPERTS;
    int conv_i = 0, attn_i = 0;
    bool norm_ready = false;     // this layer's attn_norm was produced by the previous combine
    // LK_PF_DUMP: the residual stream of the chunk's FIRST token before every
    // layer and after the last, in LK_DUMP_LAYERS's format, so the two paths
    // can be compared layer by layer on the same token.
    static FILE * pfd = nullptr;
    static bool pfd_init = false;
    if (!pfd_init) { pfd_init = true; if (const char * dp = getenv("LK_PF_DUMP")) pfd = fopen(dp, "wb"); }
    auto pf_dump = [&]() {
        if (!pfd) return;
        float buf[LK_HIDDEN];
        CK(cudaStreamSynchronize(st));
        CK(cudaMemcpy(buf, p.x, sizeof buf, cudaMemcpyDeviceToHost));
        fwrite(buf, 4, LK_HIDDEN, pfd);
        fflush(pfd);
    };
    for (int il = 0; il < LK_N_LAYERS; ++il) {
        const Layer & l = e.L[il];
        pf_dump();
        // attn_norm: layer 0's is a plain node in the decode graph; every other
        // layer's is folded Phase-4y style, so the consumer applies inv.
        if (!norm_ready) {
            pfp_b(st, "norm");
            k_pf_norm<<<m, 256, 0, st>>>(p.x, P32(e, l.attn_norm), il == 0, p.a, p.inv, nullptr);
            pfp_e(st);
        }
        norm_ready = false;
        const float * mix_scale = (il == 0) ? nullptr : p.inv;
        if (!l.is_attn) {
            float * hist = e.conv_hist + (size_t) conv_i * LK_HIDDEN * 2;
            ++conv_i;
            PfGemm g = pf_g1(e, l.sc_in, 6144, 64);
            g.aqs = p.a.qs; g.asc = p.a.sc; g.ntok = m; g.scale = mix_scale;
            g.out = p.y; g.out_ld = 6144; g.add = 0;
            pfp_b(st, "gemm_in"); pf_gemm_dense(g, st); pfp_e(st);
            dbg_tap(st, il, "in_proj", p.y, 6144);
            pfp_b(st, "conv");
            k_pf_conv<<<dim3(m, 8), 256, 0, st>>>(p.y, P32(e, l.sc_conv), hist, p.a);
            pfp_e(st);
            pfp_b(st, "conv");
            k_pf_conv_hist<<<LK_HIDDEN / 256, 256, 0, st>>>(p.y, m, hist);
            pfp_e(st);
            PfGemm o = pf_g1(e, l.sc_out, LK_HIDDEN, 64);
            o.aqs = p.a.qs; o.asc = p.a.sc; o.ntok = m; o.scale = nullptr;
            o.out = p.x; o.out_ld = LK_HIDDEN; o.add = 1;
            pfp_b(st, "gemm_out"); pf_gemm_dense(o, st); pfp_e(st);
            dbg_tap(st, il, "x_mixer", p.x, 2048);
        } else {
            const int al = attn_i++;
            PfGemm g{};
            g.qs[0] = PQS(e, l.wq); g.d[0] = PD(e, l.wq);
            g.qs[1] = PQS(e, l.wk); g.d[1] = PD(e, l.wk);
            g.qs[2] = PQS(e, l.wv); g.d[2] = PD(e, l.wv);
            g.seg_end[0] = 2048; g.seg_end[1] = 2560; g.seg_end[2] = 3072;
            g.N = 3072; g.bpr = 64;
            g.aqs = p.a.qs; g.asc = p.a.sc; g.ntok = m; g.scale = mix_scale;
            g.out = p.y; g.out_ld = 3072; g.add = 0;
            pfp_b(st, "gemm_qkv"); pf_gemm_dense(g, st); pfp_e(st);
            __half * kc = e.kcache + (size_t) al * e.max_seq * LK_N_KV_HEADS * LK_HEAD_DIM;
            __half * vc = e.vcache + (size_t) al * e.max_seq * LK_N_KV_HEADS * LK_HEAD_DIM;
            pfp_b(st, "qknr");
            k_pf_qknr<<<dim3(m, LK_N_HEADS + LK_N_KV_HEADS), LK_HEAD_DIM, 0, st>>>(
                p.y, P32(e, l.qnorm), P32(e, l.knorm), pos0, p.qh, kc, vc);
            pfp_e(st);
            if (p.attn8) {
                const size_t S = (size_t) e.max_seq;
                int8_t * k8  = p.k8 + (size_t) al * S * LK_N_KV_HEADS * LK_HEAD_DIM;
                float *  ksc = p.ksc + (size_t) al * p.kstr * LK_N_KV_HEADS;
                float *  cm  = p.kmean + al * LK_N_KV_HEADS * LK_HEAD_DIM;
                // Keys written since the int8 copy was last brought up to date
                // (decode steps between two prefills) are quantised too.  A new
                // context (pos0 == 0) re-derives the smoothing vector.
                const int lo = (pos0 == 0) ? 0 : (p.k8_valid < pos0 ? p.k8_valid : pos0);
                pfp_b(st, "attn_q8");
                if (pos0 == 0 && p.ksmooth) k_pf_kmean<<<LK_N_KV_HEADS, 1024, 0, st>>>(kc, m, cm);
                k_pf_k8<<<((pos0 + m - lo) * LK_N_KV_HEADS + 7) / 8, 256, 0, st>>>(
                    kc, lo, pos0 + m, p.ksmooth ? cm : nullptr, p.kstr, k8, ksc);
                k_pf_q8<<<(m * LK_N_HEADS + 7) / 8, 256, 0, st>>>(p.qh, m, p.ksmooth ? cm : nullptr, p.q8, p.qsc, p.qcs);
                pfp_e(st);
                pfp_b(st, "attn");
                k_pf_attn8<<<dim3((m + 31) / 32, LK_N_KV_HEADS), 256, 0, st>>>(
                    p.q8, p.qsc, p.qcs, p.qh, m, pos0, k8, ksc, p.kstr, kc, vc, p.a8near, p.a8sink, p.o);
                pfp_e(st);
                // LK_PF_ATTN_CMP (diagnostic): the fp16 kernel on the same inputs,
                // into p.y (free between qknr and the next GEMM); report the
                // int8 kernel's error relative to it, per layer, on real data.
                static const bool cmp = getenv("LK_PF_ATTN_CMP") != nullptr;
                if (cmp) {
                    k_pf_attn<<<dim3((m + 31) / 32, LK_N_KV_HEADS), 256, 0, st>>>(p.qh, m, pos0, kc, vc, p.y);
                    CK(cudaStreamSynchronize(st));
                    std::vector<float> A((size_t) m * LK_HIDDEN), B((size_t) m * LK_HIDDEN);
                    CK(cudaMemcpy(A.data(), p.o, A.size() * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(B.data(), p.y, B.size() * 4, cudaMemcpyDeviceToHost));
                    double sd = 0, sr = 0, md = 0;
                    for (size_t i = 0; i < A.size(); ++i) {
                        const double d = (double) A[i] - B[i];
                        sd += d * d; sr += (double) B[i] * B[i]; md = std::max(md, fabs(d));
                    }
                    // The error the NEXT step already adds: Q8_1 rounding of the
                    // fp16 kernel's output (k_pf_quant_rows' arithmetic).
                    double sq = 0;
                    for (size_t b = 0; b < B.size(); b += 32) {
                        float am = 0; for (int k = 0; k < 32; ++k) am = std::max(am, fabsf(B[b + k]));
                        const float d = am / 127.0f;
                        for (int k = 0; k < 32; ++k) {
                            const float q = (am == 0.0f) ? 0.0f : roundf(B[b + k] / d) * d;
                            sq += ((double) q - B[b + k]) * ((double) q - B[b + k]);
                        }
                    }
                    fprintf(stderr, "ATTN_CMP pos0=%d layer=%d rel_rms=%.4e max_abs=%.3e q8_floor=%.4e\n",
                            pos0, al, sqrt(sd / sr), md, sqrt(sq / sr));
                }
            } else {
                pfp_b(st, "attn");
                k_pf_attn<<<dim3((m + 31) / 32, LK_N_KV_HEADS), 256, 0, st>>>(p.qh, m, pos0, kc, vc, p.o);
                pfp_e(st);
            }
            pfp_b(st, "quant");
            k_pf_quant_rows<<<dim3(m, 8), 256, 0, st>>>(p.o, LK_HIDDEN, 64, p.a);
            pfp_e(st);
            PfGemm o = pf_g1(e, l.wo, LK_HIDDEN, 64);
            o.aqs = p.a.qs; o.asc = p.a.sc; o.ntok = m; o.scale = nullptr;
            o.out = p.x; o.out_ld = LK_HIDDEN; o.add = 1;
            pfp_b(st, "gemm_out"); pf_gemm_dense(o, st); pfp_e(st);
        }
        // ffn_norm: always Phase 4y; MoE layers also need the f32 router input.
        pfp_b(st, "norm");
        k_pf_norm<<<m, 256, 0, st>>>(p.x, P32(e, l.ffn_norm), 0, p.a, p.inv,
                                     l.is_moe ? p.h32 : nullptr);
        pfp_e(st);
        // The SwiGLU-fused GEMM epilogue is bit-equivalent to GEMM + k_pf_swiglu
        // (test-prefill).  It measured slower with the 3-op GEMM epilogue; with
        // PF_EPI2's 2-op epilogue it is 3% faster end to end (4k: 0.191 vs 0.197 s,
        // 32k: 1.790 vs 1.845 s), so it is the default.  LK_PF_FUSE=0 turns it off.
        static const bool nofuse = getenv("LK_PF_FUSE") && atoi(getenv("LK_PF_FUSE")) == 0;
        if (!l.is_moe) {
            PfGemm g{};
            g.qs[0] = PQS(e, l.ffn_gate); g.d[0] = PD(e, l.ffn_gate);
            g.qs[1] = g.qs[2] = PQS(e, l.ffn_up); g.d[1] = g.d[2] = PD(e, l.ffn_up);
            g.aqs = p.a.qs; g.asc = p.a.sc; g.ntok = m; g.scale = p.inv;
            if (nofuse) {
                g.seg_end[0] = LK_DENSE_INTER; g.seg_end[1] = g.seg_end[2] = 2 * LK_DENSE_INTER;
                g.N = 2 * LK_DENSE_INTER; g.bpr = 64;
                g.out = p.y; g.out_ld = 2 * LK_DENSE_INTER; g.add = 0;
                pfp_b(st, "dense_gu"); pf_gemm_dense(g, st); pfp_e(st);
                dbg_tap(st, il, "gate_up", p.y, 2 * LK_DENSE_INTER);
                pfp_b(st, "swiglu");
                k_pf_swiglu<<<dim3(m, LK_DENSE_INTER / 256), 256, 0, st>>>(
                    p.y, 2 * LK_DENSE_INTER, LK_DENSE_INTER, LK_DENSE_INTER / 32, p.a2);
                pfp_e(st);
            } else {
                // gate|up GEMM with SwiGLU + Q8_1 in its epilogue.
                g.N = LK_DENSE_INTER; g.bpr = 64;
                g.oqs = p.a2.qs; g.osc = p.a2.sc; g.okb = LK_DENSE_INTER / 32;
                pfp_b(st, "dense_gu"); pf_gemm_swiglu(g, 0, st); pfp_e(st);
            }
            PfGemm dn = pf_g1(e, l.ffn_down, LK_HIDDEN, LK_DENSE_INTER / 32);
            dn.aqs = p.a2.qs; dn.asc = p.a2.sc; dn.ntok = m; dn.scale = nullptr;
            dn.out = p.x; dn.out_ld = LK_HIDDEN; dn.add = 1;
            pfp_b(st, "dense_down"); pf_gemm_dense(dn, st); pfp_e(st);
            dbg_tap(st, il, "x_ffn", p.x, 2048);
        } else {
            pfp_b(st, "router");
            if (m <= 256)
                k_pf_router_small<<<dim3(LK_N_EXPERTS, m), 32, 0, st>>>(p.h32, P32(e, l.router), P32(e, l.rbias),
                                                                       p.inv, p.sel, p.sig);
            else
                k_pf_router<<<(m + 63) / 64, 256, 0, st>>>(p.h32, m, P32(e, l.router), P32(e, l.rbias),
                                                           p.inv, p.sel, p.sig);
            pfp_e(st);
            pfp_b(st, "router");
            k_pf_select<<<(m + 7) / 8, 256, 0, st>>>(p.sel, p.sig, m, p.idx, p.w);
            pfp_e(st);
            if (e.d_routelog)   // same [pos][layer][k] layout k_route_log writes
                CK(cudaMemcpy2DAsync(e.d_routelog + ((size_t) pos0 * LK_N_LAYERS + il) * LK_TOP_K,
                                     LK_N_LAYERS * LK_TOP_K * 4, p.idx, LK_TOP_K * 4, LK_TOP_K * 4, m,
                                     cudaMemcpyDeviceToDevice, st));
            CK(cudaMemsetAsync(p.cnt, 0, LK_N_EXPERTS * 4, st));
            const int ns = m * LK_TOP_K;
            pfp_b(st, "moe_plan");
            k_pf_moe_count<<<(ns + 255) / 256, 256, 0, st>>>(p.idx, ns, p.cnt);
            pfp_e(st);
            pfp_b(st, "moe_plan");
            k_pf_moe_plan<<<1, 32, 0, st>>>(p.cnt, p.off, p.tiles, p.ntiles);
            pfp_e(st);
            pfp_b(st, "moe_plan");
            k_pf_moe_scatter<<<LK_N_EXPERTS, 1024, 0, st>>>(p.idx, ns, p.off, p.perm);
            pfp_e(st);
            PfGemm g{};
            g.qs[0] = PQS(e, l.eg); g.d[0] = PD(e, l.eg);
            g.qs[1] = g.qs[2] = PQS(e, l.eu); g.d[1] = g.d[2] = PD(e, l.eu);
            g.bpr = 64;
            g.exp_stride = (long long) LK_MOE_INTER * 64;
            g.aqs = p.a.qs; g.asc = p.a.sc;
            g.perm = p.perm; g.tiles = p.tiles; g.ntiles = p.ntiles; g.a_shift = 2;
            g.scale = p.inv;
            if (nofuse) {
                g.seg_end[0] = LK_MOE_INTER; g.seg_end[1] = g.seg_end[2] = 2 * LK_MOE_INTER;
                g.N = 2 * LK_MOE_INTER;
                g.out = p.y; g.out_ld = 2 * LK_MOE_INTER; g.add = 0;
                pfp_b(st, "moe_gu"); pf_gemm_moe(g, max_tiles, st); pfp_e(st);
                pfp_b(st, "swiglu");
                k_pf_swiglu<<<dim3(ns, LK_MOE_INTER / 256), 256, 0, st>>>(
                    p.y, 2 * LK_MOE_INTER, LK_MOE_INTER, LK_MOE_INTER / 32, p.a2);
                pfp_e(st);
            } else {
                g.N = LK_MOE_INTER;
                g.oqs = p.a2.qs; g.osc = p.a2.sc; g.okb = LK_MOE_INTER / 32;
                pfp_b(st, "moe_gu"); pf_gemm_swiglu(g, max_tiles, st); pfp_e(st);
            }
            PfGemm dn{};
            dn.qs[0] = dn.qs[1] = dn.qs[2] = PQS(e, l.ed);
            dn.d[0]  = dn.d[1]  = dn.d[2]  = PD(e, l.ed);
            dn.seg_end[0] = dn.seg_end[1] = dn.seg_end[2] = LK_HIDDEN;
            dn.N = LK_HIDDEN; dn.bpr = LK_MOE_INTER / 32;
            dn.exp_stride = (long long) LK_HIDDEN * (LK_MOE_INTER / 32);
            dn.aqs = p.a2.qs; dn.asc = p.a2.sc;
            dn.perm = p.perm; dn.tiles = p.tiles; dn.ntiles = p.ntiles; dn.a_shift = 0;
            dn.scale = nullptr;
            dn.out = p.y; dn.out_ld = LK_HIDDEN; dn.add = 0;
            pfp_b(st, "moe_down"); pf_gemm_moe(dn, max_tiles, st); pfp_e(st);
            // The next layer's attn_norm rides on the combine (it reads the row
            // the combine just wrote); only the last layer has no successor.
            if (il + 1 < LK_N_LAYERS) {
                pfp_b(st, "moe_comb");
                k_pf_moe_combine_norm<<<m, 256, 0, st>>>(p.x, p.y, p.w, P32(e, e.L[il + 1].attn_norm),
                                                        p.a, p.inv);
                pfp_e(st);
                norm_ready = true;
            } else {
                pfp_b(st, "moe_comb");
                k_pf_moe_combine<<<dim3(m, LK_HIDDEN / 256), 256, 0, st>>>(p.x, p.y, p.w);
                pfp_e(st);
            }
        }
        if (g_debug_steps) {
            cudaError_t e_ = cudaStreamSynchronize(st);
            if (e_ != cudaSuccess) { fprintf(stderr, "PREFILL FAULT at layer %d: %s\n", il, cudaGetErrorString(e_)); exit(1); }
        }
    }
    pf_dump();
    if (pfd) { fclose(pfd); pfd = nullptr; }
    CK(cudaGetLastError());
    p.k8_valid = pos0 + m;
}

// Positions [pos0, pos0+n).  Leaves d_pos = pos0 + n; does not touch d_token.
static void prefill_batched(Engine & e, cudaStream_t st, const int * toks, int n, int pos0, int mc) {
    pf_alloc(e, mc);
    for (int c = 0; c < n; c += mc) {
        const int m = (n - c < mc) ? (n - c) : mc;
        pf_chunk(e, st, toks + c, m, pos0 + c);
    }
    const int pos = pos0 + n;
    CK(cudaMemcpyAsync(e.d_pos, &pos, 4, cudaMemcpyHostToDevice, st));
    CK(cudaStreamSynchronize(st));     // `pos` is a stack variable
    pfp_report(n);
}

int main(int argc, char ** argv) {
    // Load every kernel when the context is created.  CUDA 12 loads modules
    // lazily by default, so the FIRST batched prefill would pay ~1 ms per new
    // kernel inside the timed region -- several ms on a short prompt, a cost a
    // long-running server never sees.  The decode graph is unaffected (it is
    // captured at startup).  An explicit CUDA_MODULE_LOADING in the
    // environment still wins.
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);
    const char * pack = getenv("LK_PACK");
    const char * mode = (argc > 1) ? argv[1] : "gen";
    if (!pack) { fprintf(stderr, "LK_PACK not set\n"); return 1; }
    // KV-cache capacity, in tokens.  THIS IS THE ONLY THING THAT SIZES THE KV
    // CACHE.  LK_ATTN_MAX_SEQ in kernels.cuh is a compile-time dimension for the
    // split-K scratch and allocates nothing; the comment here used to imply the
    // two were the same thing, which is exactly how a reader concludes that
    // raising LK_ATTN_MAX_SEQ buys capacity.  It does not.
    //
    // Defaults to the model's full trained context (the GGUF says
    // lfm2moe.context_length = 131072) so the engine can be driven exactly as
    // far as llama.cpp can be.  FP16 K and V for that is 2.68 GB.  Overridable
    // to cut the allocation when only short context is being measured.
    const int max_seq = getenv("LK_MAX_SEQ") ? atoi(getenv("LK_MAX_SEQ")) : LK_ATTN_MAX_SEQ;
    if (max_seq > LK_ATTN_MAX_SEQ) {
        fprintf(stderr, "FATAL: LK_MAX_SEQ=%d exceeds the compiled split-K limit %d\n",
                max_seq, LK_ATTN_MAX_SEQ);
        return 1;
    }

    // P0 prefill chunk size.  1 is the historical behaviour and the only value
    // for which the refactor above is bit-identical by construction.  Reported
    // in RESULT so that a measurement can never be attributed to the wrong
    // batch size after the fact.
    const int prefill_b = getenv("LK_PREFILL_B") ? atoi(getenv("LK_PREFILL_B")) : 1;
    if (prefill_b < 1) {
        fprintf(stderr, "FATAL: LK_PREFILL_B=%d must be >= 1\n", prefill_b);
        return 1;
    }

    // Prefill path.  "batch" (the default) ingests all but the last prompt
    // token through prefill_batched in chunks of LK_PREFILL_CHUNK; "seq" is the
    // historical one-decode-step-per-token loop, kept for A/B and as the
    // reference the batched path is gated against.
    const char * pfm = getenv("LK_PREFILL");
    const bool pf_batch = !(pfm && strcmp(pfm, "seq") == 0);
    if (pfm && strcmp(pfm, "seq") != 0 && strcmp(pfm, "batch") != 0) {
        fprintf(stderr, "FATAL: LK_PREFILL=%s must be seq or batch\n", pfm);
        return 1;
    }
    const int pf_chunk_n = getenv("LK_PREFILL_CHUNK") ? atoi(getenv("LK_PREFILL_CHUNK")) : 8192;
    if (pf_chunk_n < 1) { fprintf(stderr, "FATAL: LK_PREFILL_CHUNK must be >= 1\n"); return 1; }

    g_debug_steps = getenv("LK_DEBUG_STEPS") ? atoi(getenv("LK_DEBUG_STEPS")) : 0;
    Engine * e = engine_create(pack, max_seq);
    printf("liquid-kernel engine ready: 40 layers, max_seq=%d\n", max_seq);
#if LK_NORM_AUDIT
    lk_audit_init();
    printf("FP16 RANGE AUDIT BUILD: recording min/max of ds.x and ds.y\n");
#endif

    cudaStream_t st; CK(cudaStreamCreate(&st));

    // --- capture the whole position-independent decode step once ---------
    // LK_NO_GRAPH replays the step directly instead of through a graph.  The
    // dump hooks below use cudaMemcpy/cudaStreamSynchronize, neither of which is
    // legal inside a capture, so graph capture has to be skipped to use them --
    // previously those hooks silently failed at capture time.
    const bool no_graph = getenv("LK_NO_GRAPH") != nullptr;
    cudaGraphExec_t gexec = nullptr;
    if (!no_graph) {
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        forward(*e, st);
        cudaGraph_t graph = nullptr;
        CK(cudaStreamEndCapture(st, &graph));
        CK(cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
        size_t nnode = 0; cudaGraphGetNodes(graph, nullptr, &nnode);
        fprintf(stderr, "captured CUDA graph: %zu nodes\n", nnode);
    }

    auto feed = [&](int tok) { CK(cudaMemcpy(e->d_token, &tok, 4, cudaMemcpyHostToDevice)); };
    // LK_PROFILE swaps graph replay for a direct call so the per-launch
    // events bracket real kernels; a graph replays as one opaque object.
    const bool profile = getenv("LK_PROFILE") != nullptr;
    auto run  = [&]() { if (profile || no_graph) forward(*e, st);
                        else CK(cudaGraphLaunch(gexec, st)); };
    auto take = [&]() { int t = 0; CK(cudaMemcpy(&t, e->d_token, 4, cudaMemcpyDeviceToHost)); return t; };
    PrefillCtx pc{ e, st, gexec, (profile || no_graph) };

    // ONE PLACE THAT KNOWS THE KV CACHE IS FINITE.
    //
    // d_pos is never bounds-checked anywhere in the forward path and kcache and
    // vcache are exactly max_seq long, so ANY mode that feeds more than max_seq
    // tokens writes past the end of them.  That happened first in gen, where it
    // looked exactly like "the model gives up at long context" rather than like
    // a memory bug; it was still open in ppl, which is the mode the perplexity
    // and NLL evidence runs through; and it was open in both argv-fed modes
    // here.  Four copies of one check is how three of them came to be missing,
    // so this is the shared copy that the argv-fed modes call.  ppl keeps its
    // own only because its arithmetic differs by one -- it feeds N-1 tokens in
    // order to score N -- and hiding that off-by-one behind a `reserve` of -1
    // would be the kind of cleverness that loses the bug a second time.
    //
    // `reserve` is the number of extra positions the mode feeds AFTER this
    // vector, which prof does and the others do not.
    auto fit = [&](std::vector<int> & v, long reserve) {
        const long cap = (long) max_seq - reserve;
        if ((long) v.size() > cap) {
            fprintf(stderr,
                    "%s: %zu tokens + %ld reserve exceeds LK_MAX_SEQ %d; using "
                    "the first %ld.\n",
                    mode, v.size(), reserve, max_seq, cap > 0 ? cap : 0L);
            v.resize(cap > 0 ? (size_t) cap : 0);
        }
    };

    if (strcmp(mode, "topk") == 0) {
        std::vector<int> tk;
        for (int i = 2; i < argc; ++i) tk.push_back(atoi(argv[i]));
        // LK_PROMPT_BIN works here too, so the logit gate can run at length.
        if (const char * pb = getenv("LK_PROMPT_BIN")) {
            FILE * pf = fopen(pb, "rb");
            if (!pf) { fprintf(stderr, "LK_PROMPT_BIN: cannot open %s\n", pb); return 1; }
            fseek(pf, 0, SEEK_END); const long nb = ftell(pf); rewind(pf);
            std::vector<int> pbuf((size_t) nb / 4);
            if (fread(pbuf.data(), 4, pbuf.size(), pf) != pbuf.size()) { fprintf(stderr, "short read\n"); return 1; }
            fclose(pf);
            tk.insert(tk.end(), pbuf.begin(), pbuf.end());
        }
        fit(tk, 0);
        if (tk.empty()) { fprintf(stderr, "topk: no tokens\n"); return 1; }
        auto tq0 = std::chrono::steady_clock::now();
        if (pf_batch && tk.size() > 1) {
            prefill_batched(*e, st, tk.data(), (int) tk.size() - 1, 0, pf_chunk_n);
            feed(tk.back()); run();
        } else {
            for (size_t i = 0; i < tk.size(); ++i) { feed(tk[i]); run(); }
        }
        CK(cudaStreamSynchronize(st));
        auto tq1 = std::chrono::steady_clock::now();
        fprintf(stderr, "topk: ingested %zu tokens in %.4f s (%s prefill)\n", tk.size(),
                std::chrono::duration<double>(tq1 - tq0).count(), pf_batch ? "batch" : "seq");
        // LK_VERIFY_PACK: the device weights must still equal the file.  Any
        // stray write from a kernel into d_pack shows up here as the tensor it
        // landed in.
        if (getenv("LK_VERIFY_PACK")) {
            const size_t CH = (size_t) 256 << 20;
            std::vector<unsigned char> buf(CH);
            size_t bad = 0, first = (size_t) -1;
            for (size_t o = 0; o < e->pack_size; o += CH) {
                const size_t n = std::min(CH, e->pack_size - o);
                CK(cudaMemcpy(buf.data(), (const char *) e->d_pack + o, n, cudaMemcpyDeviceToHost));
                const unsigned char * h = (const unsigned char *) e->pack + o;
                for (size_t i = 0; i < n; ++i)
                    if (buf[i] != h[i]) { ++bad; if (first == (size_t) -1) first = o + i; }
            }
            if (!bad) fprintf(stderr, "VERIFY_PACK: device weights identical to the pack\n");
            else {
                const char * who = "?";
                for (int i = 0; i < LK_N_TENSORS; ++i)
                    for (uint32_t pl = 0; pl < LK_TENSORS[i].n_planes; ++pl)
                        if (LK_TENSORS[i].plane[pl] <= first &&
                            (pl + 1 == LK_TENSORS[i].n_planes ? true : first < LK_TENSORS[i].plane[pl + 1]))
                            who = LK_TENSORS[i].name;
                fprintf(stderr, "VERIFY_PACK: %zu bytes differ, first at offset %zu (near %s)\n", bad, first, who);
            }
        }
        // LK_DUMP_KV: the state the prompt left behind -- K and V of every
        // attention layer for every prompt position, then every shortconv
        // layer's two-tap history -- so the batched and sequential paths can be
        // compared layer by layer (tools/pfstate.py).
        if (const char * kvp = getenv("LK_DUMP_KV")) {
            FILE * kf = fopen(kvp, "wb");
            if (!kf) { perror(kvp); return 1; }
            const size_t per = (size_t) tk.size() * LK_N_KV_HEADS * LK_HEAD_DIM;
            std::vector<__half> hb(per);
            for (int al = 0; al < 10; ++al)
                for (int kv = 0; kv < 2; ++kv) {
                    const __half * src = (kv ? e->vcache : e->kcache) + (size_t) al * max_seq * LK_N_KV_HEADS * LK_HEAD_DIM;
                    CK(cudaMemcpy(hb.data(), src, per * 2, cudaMemcpyDeviceToHost));
                    fwrite(hb.data(), 2, per, kf);
                }
            std::vector<float> ch((size_t) 30 * LK_HIDDEN * 2);
            CK(cudaMemcpy(ch.data(), e->conv_hist, ch.size() * 4, cudaMemcpyDeviceToHost));
            fwrite(ch.data(), 4, ch.size(), kf);
            fclose(kf);
        }
        if (e->d_routelog) {
            const size_t cnt = tk.size() * LK_N_LAYERS * LK_TOP_K;
            std::vector<int> buf(cnt);
            CK(cudaMemcpy(buf.data(), e->d_routelog, cnt * 4, cudaMemcpyDeviceToHost));
            FILE * rf = fopen(getenv("LK_ROUTE_LOG"), "wb");
            if (!rf) { perror(getenv("LK_ROUTE_LOG")); return 1; }
            fwrite(buf.data(), 4, cnt, rf);
            fclose(rf);
        }
        std::vector<float> lg(LK_VOCAB);
        CK(cudaMemcpy(lg.data(), e->logits, (size_t)LK_VOCAB * 4, cudaMemcpyDeviceToHost));
        // C1a: dump the FULL 128k logit vector so it can be correlated against
        // llama.cpp's, not merely against its top-10 (which would hide any
        // distribution-wide drift that happens not to cross an argmax).
        if (const char * ld = getenv("LK_LOGIT_DUMP")) {
            FILE * lf = fopen(ld, "wb");
            if (!lf) { perror(ld); return 1; }
            if (fwrite(lg.data(), 4, LK_VOCAB, lf) != (size_t)LK_VOCAB) {
                fprintf(stderr, "short write %s\n", ld); return 1; }
            fclose(lf);
        }
        std::vector<int> ord(LK_VOCAB);
        for (int i = 0; i < LK_VOCAB; ++i) ord[i] = i;
        std::partial_sort(ord.begin(), ord.begin() + 10, ord.end(),
                          [&](int a, int b) { return lg[a] > lg[b]; });
        double m = -1e30;
        for (int i = 0; i < LK_VOCAB; ++i) m = std::max(m, (double)lg[i]);
        double z = 0.0;
        for (int i = 0; i < LK_VOCAB; ++i) z += exp((double)lg[i] - m);
        const double lse = m + log(z);
        printf("topk:");
        for (int i = 0; i < 10; ++i) printf(" %d:%.4f", ord[i], lg[ord[i]] - lse);
        printf("\n  logsumexp=%.4f  top1_logit=%.4f\n", lse, lg[ord[0]]);
        return 0;
    }

    if (strcmp(mode, "ppl") == 0) {
        // Teacher-forced perplexity.  Feed token i, the graph leaves the logits
        // that predict token i+1 in e.logits, so we can score the actual next
        // token without any sampling.  Identical token IDs are fed to the
        // oracle, so the two perplexities are directly comparable.
        const char * tf = (argc > 2) ? argv[2] : nullptr;
        if (!tf) { fprintf(stderr, "usage: %s ppl TOKENS.bin [N]\n", argv[0]); return 1; }
        FILE * f = fopen(tf, "rb");
        if (!f) { perror("tokens"); return 1; }
        fseek(f, 0, SEEK_END); const long nb = ftell(f); fseek(f, 0, SEEK_SET);
        std::vector<int> ids(nb / 4);
        if (fread(ids.data(), 4, ids.size(), f) != ids.size()) { fprintf(stderr, "short read\n"); return 1; }
        fclose(f);
        int N = (int)ids.size();
        if (argc > 3) N = std::min(N, atoi(argv[3]));
        if (N < 2) { fprintf(stderr, "need >= 2 tokens\n"); return 1; }
        // THE SAME UNBOUNDED WRITE AS IN gen, AND IT WAS STILL OPEN HERE.
        //
        // The loop below feeds N-1 tokens one at a time, and each feed advances
        // d_pos, but kcache/vcache are exactly max_seq long and nothing in the
        // forward path checks.  A corpus longer than the KV cache writes past
        // the end of the cache and keeps going: this project's own corpora are
        // 149837 tokens against a 131072 cache, so the last 18765 positions were
        // being written into whatever the allocator placed next -- while still
        // printing a plausible perplexity, because the overflow lands in memory
        // that is read back as if it were real KV.
        //
        // gen got a hard guard first because that is where the corruption was
        // noticed; the identical hazard sat one mode away.  Truncating is the
        // honest fix: it scores a prefix and says so, rather than dying, because
        // the whole point of the NLL curve is the high positions.
        if (N - 1 > max_seq) {
            fprintf(stderr,
                    "ppl: corpus is %d tokens, KV cache holds %d.  Scoring the "
                    "first %d tokens only; raise LK_MAX_SEQ to score further.\n",
                    N, max_seq, max_seq);
            N = max_seq;
        }
        std::vector<float> lg(LK_VOCAB);
        const char * dump = getenv("LK_PPL_DUMP");
        FILE * df = dump ? fopen(dump, "w") : nullptr;
        double nll = 0.0; int cnt = 0; int top1ok = 0;
        auto tpp0 = std::chrono::steady_clock::now();
        if (pf_batch) {
            // Batched: prefill a chunk, then score every position in it.  The
            // final norm is the Phase 4y split (as the decode graph runs it), the
            // lm_head is the decode GEMV itself on each position's activation,
            // and inv is applied on the host -- the same multiply the GEMV would
            // do.  So the logits differ from the sequential mode ONLY through the
            // batched layers, which is exactly what this mode exists to measure.
            pf_alloc(*e, pf_chunk_n);
            std::vector<float> inv(pf_chunk_n);
            for (int c = 0; c + 1 < N; c += pf_chunk_n) {
                const int m = std::min(pf_chunk_n, N - 1 - c);
                pf_chunk(*e, st, ids.data() + c, m, c);
                k_pf_norm<<<m, 256, 0, st>>>(e->pf.x, P32(*e, e->out_norm), 0, e->pf.a, e->pf.inv, nullptr);
                CK(cudaMemcpyAsync(inv.data(), e->pf.inv, (size_t) m * 4, cudaMemcpyDeviceToHost, st));
                for (int t = 0; t < m; ++t) {
                    k_pf_pack_q8_1<<<1, 64, 0, st>>>(e->pf.a, t, e->xq);
                    k_gemv_q6_K_dp4a<<<grid_for(LK_VOCAB), 256, 0, st>>>(
                        (const uint8_t *)P(*e, e->embd, 0), (const uint8_t *)P(*e, e->embd, 1),
                        (const int8_t *)P(*e, e->embd, 2), (const __half *)P(*e, e->embd, 3),
                        e->xq, LK_VOCAB, 8, e->logits, nullptr, LK_EPS);
                    CK(cudaMemcpyAsync(lg.data(), e->logits, (size_t) LK_VOCAB * 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaStreamSynchronize(st));
                    const int i = c + t;
                    double mx = -1e30; int am = 0;
                    for (int v = 0; v < LK_VOCAB; ++v) {
                        lg[v] = lg[v] * inv[t];
                        if (lg[v] > mx) { mx = lg[v]; am = v; }
                    }
                    double z = 0.0;
                    for (int v = 0; v < LK_VOCAB; ++v) z += exp((double)lg[v] - mx);
                    const double lse = mx + log(z);
                    const double this_nll = -((double)lg[ids[i+1]] - lse);
                    if (df) fprintf(df, "%d %.8f %d %d\n", i, this_nll, am, ids[i+1]);
                    nll += this_nll;
                    top1ok += (am == ids[i+1]);
                    ++cnt;
                }
            }
        }
        for (int i = 0; !pf_batch && i + 1 < N; ++i) {
            feed(ids[i]); run();
            CK(cudaStreamSynchronize(st));
            CK(cudaMemcpy(lg.data(), e->logits, (size_t)LK_VOCAB * 4, cudaMemcpyDeviceToHost));
            double m = -1e30; int am = 0;
            for (int v = 0; v < LK_VOCAB; ++v) if (lg[v] > m) { m = lg[v]; am = v; }
            double z = 0.0;
            for (int v = 0; v < LK_VOCAB; ++v) z += exp((double)lg[v] - m);
            const double lse = m + log(z);
            const double this_nll = -((double)lg[ids[i+1]] - lse);
            if (df) fprintf(df, "%d %.8f %d %d\n", i, this_nll, am, ids[i+1]);
            nll += this_nll;
            top1ok += (am == ids[i+1]);
            ++cnt;
        }
        if (df) fclose(df);
        const double ppl = exp(nll / cnt);
        const double ppsec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tpp0).count();
        printf("RESULT nll=%.6f tokens=%d ppl=%.6f top1_acc=%.4f seconds=%.2f prefill_mode=%s\n",
               nll / cnt, cnt, ppl, (double)top1ok / cnt, ppsec, pf_batch ? "batch" : "seq");
        return 0;
    }

    if (strcmp(mode, "prof") == 0) {
        std::vector<int> tk;
        for (int i = 2; i < argc; ++i) tk.push_back(atoi(argv[i]));
        if (tk.empty()) { fprintf(stderr, "usage: %s prof TOKEN_ID...\n", argv[0]); return 1; }
        fit(tk, 4);   // prof feeds the last token 3 more times plus one forward
        lk_prof_init(st);
        for (size_t i = 0; i < tk.size(); ++i) { feed(tk[i]); run(); }
        CK(cudaStreamSynchronize(st));
        for (int i = 0; i < 3; ++i) { feed(tk.back()); forward(*e, st); }
        CK(cudaStreamSynchronize(st));
        g_prof_n = 0;
        g_prof_on = true;
        forward(*e, st);
        CK(cudaStreamSynchronize(st));
        g_prof_on = false;
        lk_prof_report((int) tk.size());
        return 0;
    }

    if (strcmp(mode, "gen") != 0) { fprintf(stderr, "unknown mode %s\n", mode); return 1; }
    std::vector<int> toks;
    for (int i = 2; i < argc; ++i) toks.push_back(atoi(argv[i]));
    // A 131072-token prompt CANNOT be passed through argv: at ~6 bytes of text
    // plus an 8-byte pointer per token that is 1.84 MB against a 2 MB ARG_MAX
    // which also has to hold the environment, so execve fails with E2BIG before
    // main ever runs and the failure looks like "the engine crashed at long
    // context" rather than "the command line was too long".
    //
    // LK_PROMPT_BIN reads the same little-endian int32 token blob that
    // tools/sweep.py and tools/perplexity.py already write, so a long-context
    // run needs no argument list at all.  Appended to any argv tokens.
    if (const char * pb = getenv("LK_PROMPT_BIN")) {
        FILE * pf = fopen(pb, "rb");
        if (!pf) { fprintf(stderr, "LK_PROMPT_BIN: cannot open %s\n", pb); return 1; }
        if (fseek(pf, 0, SEEK_END) != 0) { fprintf(stderr, "LK_PROMPT_BIN: seek\n"); fclose(pf); return 1; }
        const long nb = ftell(pf);
        rewind(pf);
        if (nb <= 0 || (nb % 4) != 0) {
            fprintf(stderr, "LK_PROMPT_BIN: %s is %ld bytes, not a non-empty int32 blob\n", pb, nb);
            fclose(pf); return 1;
        }
        const size_t cnt = (size_t) nb / 4;
        std::vector<int> pbuf(cnt);
        if (fread(pbuf.data(), 4, cnt, pf) != cnt) {
            fprintf(stderr, "LK_PROMPT_BIN: short read on %s\n", pb); fclose(pf); return 1;
        }
        fclose(pf);
        toks.insert(toks.end(), pbuf.begin(), pbuf.end());
    }
    if (toks.empty()) {
        fprintf(stderr, "usage: %s gen TOKEN_ID...   or set LK_PROMPT_BIN=file.bin\n", argv[0]);
        return 1;
    }
    const int ngen = getenv("LK_NGEN") ? atoi(getenv("LK_NGEN")) : 32;
    const bool bench = getenv("LK_BENCH") != nullptr;

    // ---- end of turn ------------------------------------------------------
    //
    // THE ENGINE DID NOT STOP.  It produced the right answer, emitted
    // <|im_end|>, and then kept going: the transcript ran through
    // <|endoftext|> and started the answer over.  Nothing was wrong with the
    // model or the kernel -- the loop simply counted to ngen and never looked
    // at what it had produced.  But a reader sees an answer repeated forever
    // and concludes the quantisation broke the model, so this is the cheapest
    // large gain in PERCEIVED coherence available, and it was worth more than
    // any kernel change in this whole long-context effort.
    //
    // 124900 = <|im_end|>, which is the model's own tokenizer.ggml.eos_token_id
    //          in the GGUF metadata, i.e. the end of a chat turn.
    // 124895 = <|endoftext|>, the document terminator.
    // Both are verified against the running llama-server's /tokenize rather
    // than guessed, and llama.cpp stops on 124900, which is the one that
    // actually occurs first in practice.
    //
    // NOT APPLIED IN BENCH MODE.  Every timing number this project reports is
    // taken over a fixed ngen; if the sequence ends early the loop would time
    // four tokens instead of thirty-two and report a throughput that no
    // workload can reproduce.
    //
    // LK_EOS="" disables the stop; LK_EOS="1,2" replaces the set.
    std::vector<int> eos;
    {
        const char * es = getenv("LK_EOS");
        const char * src = es ? es : "124900,124895";
        const char * p = src;
        while (*p) {
            char * end = nullptr;
            long v = strtol(p, &end, 10);
            if (end == p) break;
            eos.push_back((int) v);
            p = end;
            while (*p == ',' || *p == ' ') ++p;
        }
    }
    auto is_eos = [&eos](int t) {
        for (int x : eos) if (x == t) return true;
        return false;
    };

    // HARD GUARD, AND IT IS NOT DEFENSIVE PROGRAMMING -- IT IS A REAL BUG FOUND
    // BY RUNNING AT THE LIMIT.  Nothing in the forward path bounds-checks d_pos,
    // and kcache/vcache are exactly max_seq long, so a prompt plus a generation
    // that together exceed LK_MAX_SEQ writes past the end of the KV cache.  That
    // does not crash: it corrupts whatever the allocator put next and the model
    // produces blank or repetitive output that reads like the model giving up at
    // long context.  A 131072-token prompt at the default LK_MAX_SEQ of 131072
    // hits it on the FIRST generated token -- exactly the case the long-context
    // work creates -- and it was diagnosed as "the model fails at 131k" until
    // the arithmetic was done.
    if ((long) toks.size() + (long) ngen > (long) max_seq) {
        fprintf(stderr,
                "FATAL: prompt %zu + ngen %d = %ld exceeds LK_MAX_SEQ %d.  The KV "
                "cache is exactly max_seq long and nothing bounds-checks d_pos, so "
                "continuing would write past the end of it.  Raise LK_MAX_SEQ or "
                "lower LK_NGEN.\n",
                toks.size(), ngen, (long) toks.size() + (long) ngen, max_seq);
        return 1;
    }

    // prefill: feed each prompt token; the graph advances d_pos itself
    //
    // TIMED SEPARATELY, AND IT IS NOT A DETAIL.  There is no batched prefill:
    // every prompt token costs one full decode step, so ingesting a 131072-token
    // system prompt costs 131072 sequential steps at whatever the engine does at
    // that context.  llama.cpp evaluates the prompt in parallel batches and is
    // orders of magnitude faster at this.  Reporting only tok_s made the engine
    // look uniformly good at long context while hiding the cost that a serving
    // workload actually pays first.
    auto tp0 = std::chrono::steady_clock::now();
    // P0: chunked, so that a batched path can be dropped in behind
    // prefill_forward without this loop being touched again.  At the default
    // LK_PREFILL_B=1 each chunk is one token and the executed work is identical
    // to the single-token loop this replaces.  The sync count is unchanged too:
    // prefill_forward does not sync, so there is still exactly one sync here.
    const int pfb = pf_batch ? pf_chunk_n : prefill_b;
    int pfb_batches = 0;
    if (pf_batch) {
        // All but the last prompt token in chunks; the last through the graph.
        const int nb = (int) toks.size() - 1;
        if (nb > 0) prefill_batched(*e, st, toks.data(), nb, 0, pf_chunk_n);
        pfb_batches = (nb + pf_chunk_n - 1) / pf_chunk_n;
        prefill_forward(pc, toks.data() + nb, 1);
    } else {
        for (size_t i = 0; i < toks.size(); i += (size_t) pfb) {
            const int nb = ((size_t) pfb < toks.size() - i) ? pfb : (int) (toks.size() - i);
            prefill_forward(pc, toks.data() + i, nb);
            ++pfb_batches;
        }
    }
    CK(cudaStreamSynchronize(st));
    auto tp1 = std::chrono::steady_clock::now();
    const double psec = std::chrono::duration<double>(tp1 - tp0).count();
    int next = take();

    if (!bench) printf("out:");
    auto t0 = std::chrono::steady_clock::now();
    int done = 0;
    bool stopped = false;
    // emit the token produced by the prefill -- dropping it silently shifts the
    // whole trajectory by one and makes parity comparisons meaningless
    if (!bench && is_eos(next)) {
        // The turn ended before a single token was emitted.  Nothing is printed
        // and nothing is counted, which is what llama.cpp does with a stop
        // token too; counting it would make "generated" mean something
        // different in the two engines and quietly break every comparison that
        // divides by it.
        stopped = true;
    } else {
        if (!bench) { printf(" %d", next); fflush(stdout); }
        ++done;
        for (int i = 1; i < ngen; ++i) {
            run();
            CK(cudaStreamSynchronize(st));
            next = take();
            if (!bench && is_eos(next)) { stopped = true; break; }
            if (!bench) { printf(" %d", next); fflush(stdout); }
            ++done;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    if (!bench) printf("\n");
    if (e->d_routelog) {           // G4b: [positions][LK_N_LAYERS][LK_TOP_K] int32
        int endpos = 0;
        CK(cudaMemcpy(&endpos, e->d_pos, 4, cudaMemcpyDeviceToHost));
        const size_t cnt = (size_t) endpos * LK_N_LAYERS * LK_TOP_K;
        std::vector<int> buf(cnt);
        CK(cudaMemcpy(buf.data(), e->d_routelog, cnt * 4, cudaMemcpyDeviceToHost));
        FILE * rf = fopen(getenv("LK_ROUTE_LOG"), "wb");
        if (!rf) { perror(getenv("LK_ROUTE_LOG")); return 1; }
        fwrite(buf.data(), 4, cnt, rf);
        fclose(rf);
        fprintf(stderr, "route log: %d positions x %d layers x %d experts\n",
                endpos, LK_N_LAYERS, LK_TOP_K);
    }
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double tps = done / sec;
    const double BYTES_PER_TOKEN = 1466751488.0;
    printf("RESULT ctx=%d generated=%d seconds=%.4f tok_s=%.2f GB_s=%.1f pct_peak=%.1f "
           "prompt_n=%d prefill_s=%.4f prefill_tok_s=%.2f stopped=%d "
           "prefill_b=%d prefill_batches=%d prefill_mode=%s\n",
           (int)toks.size() + done, done, sec, tps, tps * BYTES_PER_TOKEN / 1e9,
           100.0 * tps * BYTES_PER_TOKEN / 1e9 / 1008.0,
           (int)toks.size(), psec,
           psec > 0.0 ? (double)toks.size() / psec : 0.0, stopped ? 1 : 0,
           pfb, pfb_batches, pf_batch ? "batch" : "seq");

#if LK_NORM_AUDIT
    // ARCHITECTURE.md 9.1 / REBRIEF A6.  Phase 4y's scale invariance is exact in
    // fp32 but block_q8_1 stores BOTH fields as fp16, so it holds only while they
    // stay in range.  A min/max RANGE is not enough to judge it: what decides the
    // risk is how MANY blocks sit below the fp16 normal floor, because there
    // fp16(d) carries only a couple of bits and inv*fp16(d) stops matching
    // fp16(inv*d).  So the distribution is counted, not just the extremes.
    {
        float au[8];
        lk_audit_read(au);
        const double SUBN = 6.103515625e-5;      // fp16 smallest NORMAL
        const double SUBS = 5.9604644775390625e-8; // fp16 smallest subnormal
        const double FMAX = 65504.0;             // fp16 largest finite
        const double n    = au[3];
        printf("\n=== FP16 RANGE AUDIT (%.0f Q8_1 blocks = %.0f per token) ===\n",
               n, n / (double) done);
        printf("  ds.x = d    min %.4e   max %.4e   mean %.4e\n", au[0], au[1],
               n > 0 ? au[6] / n : 0.0);
        printf("  ds.y = SUM  max|.| %.4e\n", au[2]);
        printf("  d   below the fp16 NORMAL floor %.4e : %.4f%% of blocks\n",
               SUBN, 100.0 * au[4] / (n > 0 ? n : 1.0));
        printf("  d   below 1e-6 (<= %.0f subnormal quanta)     : %.4f%% of blocks\n",
               1.0e-6 / SUBS, 100.0 * au[5] / (n > 0 ? n : 1.0));
        printf("  SUM headroom below the fp16 finite max %.0f: %.0fx\n", FMAX, FMAX / au[2]);
        printf("  d   headroom above the fp16 subnormal floor %.3e: %.1fx\n",
               SUBS, au[0] / SUBS);
        if (n <= 0.0f) {
            printf("  VERDICT: NO BLOCKS OBSERVED -- instrument not wired in\n");
        } else if (au[0] <= SUBS) {
            printf("  VERDICT: *** d REACHES ZERO -- fp16 cannot represent it ***\n");
        } else if (au[4] > 0.0f) {
            printf("  VERDICT: SUM is safe (%.0fx); d ENTERS THE SUBNORMAL RANGE for\n"
                   "           %.4f%% of blocks.  d stays representable, but below the\n"
                   "           normal floor inv*fp16(d) and fp16(inv*d) diverge, so the\n"
                   "           scale invariance is APPROXIMATE there, not exact.  The\n"
                   "           open question is the WEIGHT those blocks carry, which a\n"
                   "           range instrument cannot answer -- see REBRIEF A6.\n",
                   FMAX / au[2], 100.0 * au[4] / n);
        } else {
            printf("  VERDICT: SAFE -- both fields stay in the fp16 NORMAL range\n");
        }
    }
#endif
    return 0;
}
