// engine.cu - liquid-kernel Phase 3: a correct, complete, deliberately
// unoptimised LFM2.5-24B-A2B forward pass.  Speed work happens in Phase 4;
// this exists to pass the C1/C2 correctness gates first.
#include "kernels.cuh"
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
    float *  d_apv = nullptr;    // [125] argmax stage-1 block winners
    int *    d_api = nullptr;    // [125] their indices
    int *    d_routelog = nullptr; // [max_seq][LK_N_LAYERS][LK_TOP_K], G4b only
    int *    d_idx = nullptr;
    float *  d_w = nullptr;
};

static inline const void * P(const Engine & e, const LKTensor * t, int plane) {
    return (const char *)e.d_pack + t->plane[plane];
}
static inline const float *   P32(const Engine & e, const LKTensor * t) { return (const float *)P(e, t, 0); }
static inline const uint4 *   PQS(const Engine & e, const LKTensor * t) { return (const uint4 *)P(e, t, 0); }
static inline const __half *  PD (const Engine & e, const LKTensor * t) { return (const __half *)P(e, t, 1); }

static inline int grid_for(int rows) { return (rows + 7) / 8; }   // 8 warps/block

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
    CK(cudaMalloc(&e->d_apv, LK_ARGMAX_BLOCKS * 4));
    CK(cudaMalloc(&e->d_api, LK_ARGMAX_BLOCKS * 4));
    CK(cudaMalloc(&e->d_idx, LK_TOP_K * 4));
    CK(cudaMalloc(&e->d_w, LK_TOP_K * 4));
    CK(cudaMemset(e->conv_hist, 0, (size_t)30 * LK_HIDDEN * 2 * 4));
    CK(cudaMemset(e->kcache, 0, KV * 2));
    CK(cudaMemset(e->vcache, 0, KV * 2));
    return e;
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
        LKP_B("norm_attn");
    // dead outputs: nothing reads the f32 or fp16 form of the PRE-mixer norm
    // (the router consumes the POST-ffn_norm one), so skip 12 KB of stores per
    // call.  The kernel already branches on null.
    k_rmsnorm_q8_1<<<1, 256, LK_HIDDEN * 4, st>>>(e.x, P32(e, l.attn_norm), LK_HIDDEN, LK_EPS,
                                              nullptr, nullptr, e.xq); LKP_E();
        LK_STEP("mixer-norm");

        if (!l.is_attn) {
            float * hist = e.conv_hist + (size_t)conv_i * LK_HIDDEN * 2;
            ++conv_i;
            LKP_B("shortconv_in");
    k_gemv_q4_0<<<LK_GEMV_BLOCKS(6144), 256, 0, st>>>(
                PQS(e, l.sc_in), PD(e, l.sc_in), e.xq, 6144, 64, e.y, 0); LKP_E();
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
            const int ar = (il == 0 && getenv("LK_NO_RESIDUAL")) ? 0 : 1;
            LKP_B("shortconv_out");
    k_gemv_q4_0<<<LK_GEMV_BLOCKS(LK_HIDDEN), 256, 0, st>>>(
                PQS(e, l.sc_out), PD(e, l.sc_out), e.xq, LK_HIDDEN, 64, e.x, ar); LKP_E();
            if (il == 0 && getenv("LK_DUMP_L0X")) {   // immediately after the GEMV
                const char * dp = getenv("LK_DUMP_L0X");
                FILE * df = fopen(dp, "wb");
                float tx[LK_HIDDEN];
                CK(cudaStreamSynchronize(st));
                CK(cudaMemcpy(tx, e.x, LK_HIDDEN * 4, cudaMemcpyDeviceToHost));
                fwrite(tx, 4, LK_HIDDEN, df);
                fclose(df);
            }
            LK_STEP("shortconv");
        } else {
            const int al = attn_i++;
            LKP_B("attn_qkv");
    k_gemv_q4_0_3<<<LK_GEMV_BLOCKS(LK_GEMV_ROWS * (LK_HIDDEN + 1024)), 256, 0, st>>>(
                PQS(e, l.wq), PD(e, l.wq), LK_HIDDEN, e.qv,
                PQS(e, l.wk), PD(e, l.wk), 512, e.kv,
                PQS(e, l.wv), PD(e, l.wv), 512, e.vv,
                e.xq, 64); LKP_E();
            __half * kc = e.kcache + (size_t)al * e.max_seq * LK_N_KV_HEADS * LK_HEAD_DIM;
            __half * vc = e.vcache + (size_t)al * e.max_seq * LK_N_KV_HEADS * LK_HEAD_DIM;
            LKP_B("qk_norm_rope");
    k_qk_norm_rope<<<LK_N_HEADS + LK_N_KV_HEADS, LK_HEAD_DIM, 0, st>>>(
                e.qv, e.kv, e.vv, P32(e, l.qnorm), P32(e, l.knorm),
                e.d_pos, kc, vc); LKP_E();
            LKP_B("attention");
    // LK_ATTN_SERIAL keeps the original single-block kernel reachable for A/B.
    if (getenv("LK_ATTN_SERIAL")) {
        k_attention_decode<<<LK_N_HEADS, LK_ATTN_BLOCK, 0, st>>>(
                    e.qv, kc, vc, e.d_pos, e.tmp, e.xq);
    } else {
        k_attn_split<<<dim3(LK_N_HEADS, LK_ATTN_SPLIT), 256, 0, st>>>(
                    e.qv, kc, vc, e.d_pos, e.attn_part);
        k_attn_combine<<<LK_N_HEADS, LK_HEAD_DIM, 0, st>>>(
                    e.attn_part, e.tmp, e.xq);
    }
    LKP_E();
            
            LKP_B("attn_out");
    k_gemv_q4_0<<<LK_GEMV_BLOCKS(LK_HIDDEN), 256, 0, st>>>(
                PQS(e, l.wo), PD(e, l.wo), e.xq, LK_HIDDEN, 64, e.x, 1); LKP_E();
            LK_STEP("attention");
        }

        // ---- channel mixer --------------------------------------------
        LKP_B("norm_ffn");
    // the f32 form is the router's input and exists only in MoE layers; the
    // fp16 form is never read anywhere in the engine.
    k_rmsnorm_q8_1<<<1, 256, LK_HIDDEN * 4, st>>>(e.x, P32(e, l.ffn_norm), LK_HIDDEN, LK_EPS,
                                              l.is_moe ? e.h : nullptr, nullptr, e.xq); LKP_E();
        LK_STEP("ffn-norm");

        if (!l.is_moe) {
            LKP_B("dense_gate");
    k_gemv_q4_0<<<LK_GEMV_BLOCKS(LK_DENSE_INTER), 256, 0, st>>>(
                PQS(e, l.ffn_gate), PD(e, l.ffn_gate), e.xq, LK_DENSE_INTER, 64, e.y, 0); LKP_E();
            LKP_B("dense_up");
    k_gemv_q4_0<<<LK_GEMV_BLOCKS(LK_DENSE_INTER), 256, 0, st>>>(
                PQS(e, l.ffn_up), PD(e, l.ffn_up), e.xq, LK_DENSE_INTER, 64, e.y + LK_DENSE_INTER, 0); LKP_E();
            LKP_B("dense_swiglu");
    k_swiglu<<<64, 256, 0, st>>>(e.y, e.y + LK_DENSE_INTER, LK_DENSE_INTER, e.inter, e.xq); LKP_E();
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
    k_gemv_q4_0<<<LK_GEMV_BLOCKS(LK_HIDDEN), 256, 0, st>>>(
                PQS(e, l.ffn_down), PD(e, l.ffn_down), e.xq, LK_HIDDEN, 368, e.x, 1); LKP_E();
            LK_STEP("dense-ffn");
        } else {
            // The router consumes the POST-ffn_norm hidden state in FP32:
            // llama.cpp's ggml_rms_norm emits f32 and ffn_gate_inp is F32, so
            // build_lora_mm runs f32 x f32.  (The experts, by contrast, take
            // the Q8_1-quantised activation.)
            LKP_B("router_dot");
    k_router_dot<<<LK_N_EXPERTS, 32, 0, st>>>(e.h, P32(e, l.router), e.d_gate); LKP_E();
            LKP_B("router_top4");
    // one lane per expert: the sigmoids are computed cooperatively, so a
    // 32-thread block would leave experts 32..63 uninitialised.
    k_router_top4<<<1, LK_N_EXPERTS, 0, st>>>(e.d_gate, P32(e, l.rbias),
                                    e.d_idx, e.d_w); LKP_E();
            if (e.d_routelog) {          // G4b instrumentation (off unless asked)
                LKP_B("route_log");
    k_route_log<<<1, 32, 0, st>>>(e.d_idx, e.d_pos, il, e.d_routelog); LKP_E();
            }
            // Phase 4: ONE launch per matrix for all 4 experts (grid.y = expert),
            // replacing 16 launches per MoE layer.  Weights stay on device, so the
            // per-layer cudaStreamSynchronize host round-trip is gone too.
            const int GU = LK_TOP_K * LK_MOE_INTER;
            LKP_B("moe_gate_up");
    k_gemv_q4_0_exp2<<<dim3(grid_for(LK_MOE_INTER), 2 * LK_TOP_K), 256, 0, st>>>(
                PQS(e, l.eg), PD(e, l.eg), PQS(e, l.eu), PD(e, l.eu), e.xq,
                LK_MOE_INTER, 64, LK_MOE_INTER, 0,
                e.d_idx, e.moe_gu, e.moe_gu + GU); LKP_E();
            LKP_B("moe_swiglu");
    k_swiglu<<<48, 256, 0, st>>>(e.moe_gu, e.moe_gu + GU, GU, nullptr, e.xq2); LKP_E();
            
            // down: each expert consumes its OWN 48-block activation slice
            LKP_B("moe_down");
    k_gemv_q4_0_exp<<<dim3(grid_for(LK_HIDDEN), LK_TOP_K), 256, 0, st>>>(
                PQS(e, l.ed), PD(e, l.ed), e.xq2, LK_HIDDEN, 48, LK_HIDDEN, 48,
                e.d_idx, e.moe_down4); LKP_E();
            LKP_B("moe_accum");
    k_moe_accum4<<<8, 256, 0, st>>>(e.x, e.moe_down4, e.d_w, LK_HIDDEN); LKP_E();
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
    LKP_B("norm_final");
    k_rmsnorm_q8_1<<<1, 256, LK_HIDDEN * 4, st>>>(e.x, P32(e, e.out_norm), LK_HIDDEN, LK_EPS,
                                          nullptr, nullptr, e.xq); LKP_E();
    LKP_B("lm_head");
    // LK_Q6K_SCALAR keeps the original scalar kernel reachable so the two can be
    // compared on real weights without a rebuild.
    if (getenv("LK_Q6K_SCALAR"))
        k_gemv_q6_K<<<grid_for(LK_VOCAB), 256, 0, st>>>(
            (const uint8_t *)P(e, e.embd, 0), (const uint8_t *)P(e, e.embd, 1),
            (const int8_t *)P(e, e.embd, 2), (const __half *)P(e, e.embd, 3),
            e.xq, LK_VOCAB, 8, e.logits);
    else
        k_gemv_q6_K_dp4a<<<grid_for(LK_VOCAB), 256, 0, st>>>(
            (const uint8_t *)P(e, e.embd, 0), (const uint8_t *)P(e, e.embd, 1),
            (const int8_t *)P(e, e.embd, 2), (const __half *)P(e, e.embd, 3),
            e.xq, LK_VOCAB, 8, e.logits);
    LKP_E();
    LKP_B("argmax_part");
    k_argmax_part<<<LK_ARGMAX_BLOCKS, 256, 0, st>>>(e.logits, e.d_apv, e.d_api); LKP_E();
    LKP_B("argmax_final");
    k_argmax_final<<<1, 256, 0, st>>>(e.d_apv, e.d_api, LK_ARGMAX_BLOCKS,
                                      e.d_token, e.d_pos); LKP_E();
    CK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    const char * pack = getenv("LK_PACK");
    const char * mode = (argc > 1) ? argv[1] : "gen";
    if (!pack) { fprintf(stderr, "LK_PACK not set\n"); return 1; }
    // The objective is single-stream decode at up to 8k context and
    // LK_ATTN_MAX_SEQ (the shared-memory bound in k_attention_decode) is already
    // 8192, but the KV cache was sized for 2048 -- anything longer addressed
    // past the end of the allocation.  Overridable for experiments.
    const int max_seq = getenv("LK_MAX_SEQ") ? atoi(getenv("LK_MAX_SEQ")) : 8192;

    g_debug_steps = getenv("LK_DEBUG_STEPS") ? atoi(getenv("LK_DEBUG_STEPS")) : 0;
    Engine * e = engine_create(pack, max_seq);
    printf("liquid-kernel engine ready: 40 layers, max_seq=%d\n", max_seq);

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

    if (strcmp(mode, "topk") == 0) {
        std::vector<int> tk;
        for (int i = 2; i < argc; ++i) tk.push_back(atoi(argv[i]));
        for (size_t i = 0; i < tk.size(); ++i) { feed(tk[i]); run(); }
        CK(cudaStreamSynchronize(st));
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
        std::vector<float> lg(LK_VOCAB);
        const char * dump = getenv("LK_PPL_DUMP");
        FILE * df = dump ? fopen(dump, "w") : nullptr;
        double nll = 0.0; int cnt = 0; int top1ok = 0;
        for (int i = 0; i + 1 < N; ++i) {
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
        printf("RESULT nll=%.6f tokens=%d ppl=%.6f top1_acc=%.4f\n",
               nll / cnt, cnt, ppl, (double)top1ok / cnt);
        return 0;
    }

    if (strcmp(mode, "prof") == 0) {
        std::vector<int> tk;
        for (int i = 2; i < argc; ++i) tk.push_back(atoi(argv[i]));
        if (tk.empty()) { fprintf(stderr, "usage: %s prof TOKEN_ID...\n", argv[0]); return 1; }
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
    if (toks.empty()) { fprintf(stderr, "usage: %s gen TOKEN_ID...\n", argv[0]); return 1; }
    const int ngen = getenv("LK_NGEN") ? atoi(getenv("LK_NGEN")) : 32;
    const bool bench = getenv("LK_BENCH") != nullptr;

    // prefill: feed each prompt token; the graph advances d_pos itself
    for (size_t i = 0; i < toks.size(); ++i) { feed(toks[i]); run(); }
    CK(cudaStreamSynchronize(st));
    int next = take();

    if (!bench) printf("out:");
    auto t0 = std::chrono::steady_clock::now();
    int done = 0;
    // emit the token produced by the prefill -- dropping it silently shifts the
    // whole trajectory by one and makes parity comparisons meaningless
    if (!bench) { printf(" %d", next); fflush(stdout); }
    ++done;
    for (int i = 1; i < ngen; ++i) {
        run();
        CK(cudaStreamSynchronize(st));
        next = take();
        if (!bench) { printf(" %d", next); fflush(stdout); }
        ++done;
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
    printf("RESULT ctx=%d generated=%d seconds=%.4f tok_s=%.2f GB_s=%.1f pct_peak=%.1f\n",
           (int)toks.size() + done, done, sec, tps, tps * BYTES_PER_TOKEN / 1e9,
           100.0 * tps * BYTES_PER_TOKEN / 1e9 / 1008.0);
    return 0;
}
