// prefill.cu - batched prefill kernels.  See prefill.cuh for what this path is
// and how it relates, numerically, to the decode path.
#include "prefill.cuh"
#include <cstdio>
#include <cstdlib>

#define PF_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------
__device__ __forceinline__ void pf_cp16(void * smem, const void * gmem, bool pred) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    const int sz = pred ? 16 : 0;              // 0 = zero-fill, nothing is read
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(s), "l"(gmem), "r"(sz) : "memory");
}
__device__ __forceinline__ void pf_cp4(void * smem, const void * gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void pf_cp8(void * smem, const void * gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void pf_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void pf_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

// The int32 -> fp32 conversion runs at 1/8 of FP32 rate on sm_86 (16/clk/SM),
// which would make it the bottleneck of the whole GEMM.  So the mma accumulator
// is SEEDED with the bit pattern of 1.5*2^23: the hardware's integer add then
// leaves 0x4B400000 + sumi, which read as a float is exactly 12582912 + sumi for
// |sumi| < 2^22 (a Q4_0 x Q8_1 block is bounded by 32*15*128 = 61440).  One exact
// FADD recovers (float)sumi.  Same value as a cvt, at full rate.
#define PF_MAGIC_I 0x4B400000
#define PF_MAGIC_F 12582912.0f

// The mma wrappers are deliberately NOT volatile: they read and write only
// registers, and `volatile` would pin every mma in program order, which stops
// the compiler interleaving independent tiles with the FP work that consumes
// them.  (ldmatrix and cp.async stay volatile: they touch shared memory and must
// not move across a barrier.)
__device__ __forceinline__ void pf_mma_s8(int (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "r"(PF_MAGIC_I));
}
__device__ __forceinline__ void pf_mma_f16(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// fp16 accumulate: 2.2x the fp32-accumulate rate on GA102 (87.7 vs 40.0
// TMAC/s measured on this card).  Used for P.V inside ONE 64-key tile only,
// whose partial is then added to the fp32 output accumulator.
__device__ __forceinline__ void pf_mma_f16h(unsigned (&d)[2], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void pf_ldsm4(unsigned (&r)[4], const void * p) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void pf_ldsm4t(unsigned (&r)[4], const void * p) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ float pf_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ unsigned pf_pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned *>(&h);
}

// quantize_q8_1_warp (quant.cuh), operation for operation, writing the planar
// form.  One warp per 32-element block, one element per lane.
__device__ __forceinline__ void pf_quant_warp(const float xi, int8_t * __restrict__ qs,
                                              float2 * __restrict__ sc, int lane) {
    const unsigned am = __reduce_max_sync(0xffffffffu, __float_as_uint(fabsf(xi)));
    float sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = __uint_as_float(am) / 127.0f;
    const int8_t q = (am == 0u) ? (int8_t) 0 : (int8_t) roundf(xi / d);
    qs[lane] = q;
    if (lane == 0) {
        const __half2 ds = __floats2half2_rn(d, sum);
        *sc = make_float2(__low2float(ds), 8.0f * __high2float(ds));
    }
}

// ---------------------------------------------------------------------------
// Q4_0 x Q8_1 tensor-core GEMM.
//
// mma.m16n8k32 consumes exactly one 32-element Q4_0 block per k-step, and the
// Q4_0 nibble layout already IS the int8 A-fragment layout: byte j of a block
// holds element j (low nibble) and element j+16 (high nibble), so word w of the
// 16-byte block gives a0 = w & 0x0F0F0F0F (k = 4w..4w+3) and
// a2 = (w >> 4) & 0x0F0F0F0F (k = 16+4w..).  No shuffling, no dequantisation.
// Nibbles stay UNSIGNED (0..15) and the -8 offset is applied through the
// activation's raw sum, exactly as dot_q4_0_q8_1 does.
//
// Block tile 128 rows x BN tokens, 8 warps as 2 (rows) x 4 (tokens), warp tile
// 64 rows x BN/4 tokens.  A stage holds KB blocks (KB*32 K-elements).
// Every shared-memory row pitch carries 16 B of padding so that the fragment
// reads of 8 rows x 4 words land in 32 distinct banks.
// ---------------------------------------------------------------------------
#ifndef PF_ATTN_PV_F16
#define PF_ATTN_PV_F16 1
#endif
#ifndef PF_ATTN_QK_F16
#define PF_ATTN_QK_F16 0
#endif
#define PF_BM 128
#ifndef PF_ABL
#define PF_ABL 0          // ablation bitmask (bench only): 1 = 1 FADD/value, 2 = no mma, 4 = no copies
#endif
// PF_EPI2 = 1: two FP ops per mma value instead of three.  The magic offset is
// folded into a per-(token, block) constant c = d8 * 1.5*2^23 + 8*sum, so
//   d8*sumi - 8*sum  =  fma(d8, float_bits(magic + sumi), -c)
// with one rounding in c of at most half an ulp of c, i.e. ~0.5 * d8: an error
// of half a unit in sumi -- the same size as the fp16 rounding Q8_1 already
// applies to 8*sum itself (2^-12 * |8*sum| ~ 0.4 * d8 for typical blocks).
#ifndef PF_EPI2
#define PF_EPI2 1
#endif

// KB = 32-element blocks per pipeline stage.  Pitches carry 16 B of padding so
// the fragment reads of 8 rows x 4 words land in 32 distinct banks:
//   A row (KB+1)*16 B -> bank 4r + w;  B row KB*32+16 B -> bank 4n + w;
//   scale row (KB+2) float2 -> the four tokens of a quad hit distinct bank pairs.
template <int BM, int BN, int KB>
struct PfTile {
    static constexpr int A_P   = (KB + 1) * 16;
    static constexpr int B_P   = KB * 32 + 16;
    // float2 per token row: >= KB, 16-byte aligned, and 6 rather than 4 so the
    // four tokens of a quad (rows 2t) land in distinct bank pairs.
    static constexpr int S_P   = (KB + 2 > 6) ? KB + 2 : 6;
    static constexpr int SA    = BM * A_P;
    static constexpr int SD    = BM * KB * 2;
    static constexpr int SB    = BN * B_P;
    static constexpr int SS    = BN * S_P * 8;
    static constexpr int STAGE = SA + SD + SB + SS;
};

// WM = warps along the row dimension (2 or 4); the 8 warps tile the block as
// WM x (8/WM), so a warp owns (128/WM) rows x (BN*WM/8) tokens = MT x NT mma
// tiles.  Wider warp tiles reuse each fragment read across more mma.
// EPI = 0: out (+)= scale * acc, f32.
// EPI = 1: the SwiGLU epilogue.  The block's 128 rows are 64 gate rows and the
//          SAME 64 up rows (qs[0] = gate tensor, qs[1] = up tensor, N = the
//          intermediate width), so silu(gate)*up and its Q8_1 quantisation
//          happen in the epilogue and the f32 gate/up never reach memory.
// BM = rows per block tile (128 or 256).  The GEMM is L2 -> shared-memory
// bandwidth-bound at BM 128 x BN 64 (the copies alone, with no math, run at
// 1.57 TB/s against a measured 1.73 TB/s wall), so a taller tile that shares
// each activation tile across 256 rows is the lever; BM 256 needs 16 warps to
// keep the per-warp tile, and therefore the register budget, unchanged.
template <int BN, int NST, int KB, int MINB, int WM, int EPI, int NTH = 256, int BM = 128>
__global__ void __launch_bounds__(NTH, MINB) k_pf_mmq(const PfGemm g) {
    using T = PfTile<BM, BN, KB>;
    static_assert(EPI == 0 || (WM == 2 && NTH == 256 && BM == 128), "the SwiGLU epilogue pairs warp rows 0-63 with 64-127");
    constexpr int NW = NTH / 32;
    constexpr int WN = NW / WM;
    constexpr int MT = (BM / WM) / 16;          // m16 tiles per warp
    constexpr int NT = (BN / WN) / 8;           // n8 tiles per warp
    static_assert(MT >= 1 && NT >= 1, "warp tile");
    extern __shared__ __align__(16) unsigned char pf_smem[];
    __shared__ int   s_arow[BN];
    __shared__ int   s_orow[BN];
    __shared__ float s_scl[BN];

    int rt, tstart, tcnt, expert = 0;
    if (g.perm) {
        if ((int) blockIdx.y >= *g.ntiles) return;          // block-uniform
        const int4 tl = g.tiles[blockIdx.y];
        expert = tl.x; tstart = tl.y; tcnt = tl.z; rt = blockIdx.x;
    } else {
        // Grouped raster: 8 row tiles sweep every token tile together, so the
        // weights and activations a wave of blocks touches stay inside L2.
        const int RT = EPI ? g.N / 64 : g.N / BM;
        const int TT = (g.ntok + BN - 1) / BN;
        const int L  = blockIdx.x;
        const int GM = 8;
        const int npg   = GM * TT;
        const int grp   = L / npg;
        const int first = grp * GM;
        const int gsz   = (RT - first < GM) ? (RT - first) : GM;
        rt = first + (L % npg) % gsz;
        const int tt = (L % npg) / gsz;
        tstart = tt * BN;
        tcnt = g.ntok - tstart; if (tcnt > BN) tcnt = BN;
    }
    for (int i = threadIdx.x; i < BN; i += NTH) {
        int slot = -1;
        if (i < tcnt) slot = g.perm ? g.perm[tstart + i] : tstart + i;
        const int ar = (slot >= 0) ? (slot >> g.a_shift) : -1;
        s_arow[i] = ar;
        s_orow[i] = slot;
        s_scl[i]  = (ar >= 0 && g.scale) ? g.scale[ar] : 1.0f;
    }
    __syncthreads();

    // Tile rows 0-63 come from wq0, rows 64-127 from wq1.  EPI 0: one segment,
    // contiguous.  EPI 1: gate rows and the matching up rows.
    const int row0 = EPI ? rt * 64 : rt * BM;
    const uint4 * wq0; const uint4 * wq1; const __half * wd0; const __half * wd1;
    {
        const size_t eo = (size_t) expert * g.exp_stride;
        if (EPI) {
            wq0 = g.qs[0] + eo + (size_t) row0 * g.bpr; wd0 = g.d[0] + eo + (size_t) row0 * g.bpr;
            wq1 = g.qs[1] + eo + (size_t) row0 * g.bpr; wd1 = g.d[1] + eo + (size_t) row0 * g.bpr;
        } else {
            const int seg  = (row0 < g.seg_end[0]) ? 0 : (row0 < g.seg_end[1]) ? 1 : 2;
            const int sbase = seg ? g.seg_end[seg - 1] : 0;
            wq0 = g.qs[seg] + eo + (size_t) (row0 - sbase) * g.bpr;
            wd0 = g.d[seg]  + eo + (size_t) (row0 - sbase) * g.bpr;
            wq1 = wq0 + (size_t) 64 * g.bpr; wd1 = wd0 + (size_t) 64 * g.bpr;
        }
    }
    (void) wq1; (void) wd1;

    // Copy assignments.  Every thread copies the SAME chunks of every stage, so
    // its source pointers and shared offsets are computed once here and the
    // stage loop only adds the K offset.  Trip counts are compile-time, so the
    // copies unroll into straight-line LDGSTS with no loop control and no
    // per-chunk address arithmetic or s_arow lookups.
    const int tid = threadIdx.x;
    constexpr int A_CH = BM * KB, A_PER = (A_CH + NTH - 1) / NTH;
    constexpr int B_CH = BN * KB * 2, B_PER = (B_CH + NTH - 1) / NTH;
    constexpr int S_CH = BN * (KB / 2), S_PER = (S_CH + NTH - 1) / NTH;
    const uint4 *  a_src[A_PER]; int a_dst[A_PER];
    const int8_t * b_src[B_PER]; int b_dst[B_PER]; bool b_ok[B_PER];
    const float2 * s_src[S_PER]; int s_dst[S_PER]; bool s_ok[S_PER];
#pragma unroll
    for (int k = 0; k < A_PER; ++k) {
        const int c = tid + k * NTH;
        const int r = (c < A_CH ? c : 0) / KB, kk = c % KB;
        a_src[k] = ((EPI == 0 || r < 64) ? wq0 + (size_t) r * g.bpr : wq1 + (size_t) (r - 64) * g.bpr) + kk;
        a_dst[k] = (c < A_CH) ? r * T::A_P + kk * 16 : -1;
    }
    const __half * d_src = (EPI == 0 || tid < 64) ? wd0 + (size_t) (tid < BM ? tid : 0) * g.bpr
                                                  : wd1 + (size_t) (tid - 64) * g.bpr;
#pragma unroll
    for (int k = 0; k < B_PER; ++k) {
        const int c = tid + k * NTH;
        const int r = (c < B_CH ? c : 0) / (KB * 2), kk = c % (KB * 2);
        const int ar = s_arow[r];
        b_ok[k]  = ar >= 0;
        b_src[k] = g.aqs + (size_t) (ar < 0 ? 0 : ar) * (g.bpr * 32) + kk * 16;
        b_dst[k] = (c < B_CH) ? T::SA + T::SD + r * T::B_P + kk * 16 : -1;
    }
#pragma unroll
    for (int k = 0; k < S_PER; ++k) {
        const int c = tid + k * NTH;
        const int r = (c < S_CH ? c : 0) / (KB / 2), kk = c % (KB / 2);
        const int ar = s_arow[r];
        s_ok[k]  = ar >= 0;
        s_src[k] = g.asc + (size_t) (ar < 0 ? 0 : ar) * g.bpr + kk * 2;
        s_dst[k] = (c < S_CH) ? T::SA + T::SD + T::SB + r * (T::S_P * 8) + kk * 16 : -1;
    }
    auto load = [&](int stage, int kb0) {
        unsigned char * base = pf_smem + stage * T::STAGE;
#pragma unroll
        for (int k = 0; k < A_PER; ++k)
            if (A_CH % NTH == 0 || a_dst[k] >= 0) pf_cp16(base + a_dst[k], a_src[k] + kb0, true);
        if (tid < BM) {
            if (KB * 2 == 16)     pf_cp16(base + T::SA + tid * 16, d_src + kb0, true);
            else if (KB * 2 == 8) pf_cp8(base + T::SA + tid * 8, d_src + kb0);
            else                  pf_cp4(base + T::SA + tid * 4, d_src + kb0);
        }
#pragma unroll
        for (int k = 0; k < B_PER; ++k)
            if (B_CH % NTH == 0 || b_dst[k] >= 0) pf_cp16(base + b_dst[k], b_src[k] + kb0 * 32, b_ok[k]);
#pragma unroll
        for (int k = 0; k < S_PER; ++k)
            if (S_CH % NTH == 0 || s_dst[k] >= 0) pf_cp16(base + s_dst[k], s_src[k] + kb0, s_ok[k]);
    };

    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp % WM, wn = warp / WM;
    const int g8 = lane >> 2, t4 = lane & 3;

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    const int nkb = g.bpr / KB;
#pragma unroll
    for (int s = 0; s < NST - 1; ++s) {
        if (s < nkb) load(s, s * KB);
        pf_commit();
    }
    for (int it = 0; it < nkb; ++it) {
        pf_wait<NST - 2>();
        __syncthreads();
        {
            const int nx = it + NST - 1;
#if PF_ABL & 4       // ablation: no copies after the prologue (math only, results wrong)
            (void) nx;
#else
            if (nx < nkb) load(nx % NST, nx * KB);
#endif
            pf_commit();
        }
        const unsigned char * pA = pf_smem + (it % NST) * T::STAGE;
        const unsigned char * pD = pA + T::SA;
        const unsigned char * pB = pD + T::SD;
        const unsigned char * pS = pB + T::SB;
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) {
            unsigned bf[NT][2];
            float2   s0[NT], s1[NT];
            float    c0[NT], c1[NT];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int nl = wn * (BN / WN) + j * 8 + g8;
                const unsigned char * pb = pB + nl * T::B_P + kb * 32 + t4 * 4;
                bf[j][0] = *reinterpret_cast<const unsigned *>(pb);
                bf[j][1] = *reinterpret_cast<const unsigned *>(pb + 16);
                const int nc = wn * (BN / WN) + j * 8 + 2 * t4;
                s0[j] = reinterpret_cast<const float2 *>(pS + nc * (T::S_P * 8))[kb];
                s1[j] = reinterpret_cast<const float2 *>(pS + (nc + 1) * (T::S_P * 8))[kb];
                c0[j] = fmaf(s0[j].x, PF_MAGIC_F, s0[j].y);
                c1[j] = fmaf(s1[j].x, PF_MAGIC_F, s1[j].y);
            }
            (void) c0; (void) c1;
            // All of this k-step's mma first, then the FP that consumes them: an
            // mma result takes ~30 cycles, and consuming each one straight after
            // its issue (what the natural loop order compiles to) stalls on every
            // tile.  Issuing the MT x NT independent mma back to back hides it.
            unsigned af[MT][4];
            float d0v[MT], d1v[MT];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r0 = wm * (BM / WM) + i * 16 + g8;
                const unsigned w0 = *reinterpret_cast<const unsigned *>(pA + r0 * T::A_P + kb * 16 + t4 * 4);
                const unsigned w1 = *reinterpret_cast<const unsigned *>(pA + (r0 + 8) * T::A_P + kb * 16 + t4 * 4);
                af[i][0] = w0 & 0x0F0F0F0Fu;        af[i][1] = w1 & 0x0F0F0F0Fu;
                af[i][2] = (w0 >> 4) & 0x0F0F0F0Fu; af[i][3] = (w1 >> 4) & 0x0F0F0F0Fu;
                d0v[i] = __half2float(reinterpret_cast<const __half *>(pD + r0 * (KB * 2))[kb]);
                d1v[i] = __half2float(reinterpret_cast<const __half *>(pD + (r0 + 8) * (KB * 2))[kb]);
            }
            int D[MT][NT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j) {
#if PF_ABL & 2       // ablation: no tensor-core instruction (results wrong)
                    D[i][j][0] = (int) (af[i][0] ^ bf[j][0]); D[i][j][1] = (int) (af[i][1] ^ bf[j][1]);
                    D[i][j][2] = (int) (af[i][2] ^ bf[j][0]); D[i][j][3] = (int) (af[i][3] ^ bf[j][1]);
#else
                    pf_mma_s8(D[i][j], af[i], bf[j]);
#endif
                }
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j) {
#if PF_ABL & 1       // ablation: one FADD per value instead of the full term (results wrong)
                    acc[i][j][0] += __int_as_float(D[i][j][0]); acc[i][j][1] += __int_as_float(D[i][j][1]);
                    acc[i][j][2] += __int_as_float(D[i][j][2]); acc[i][j][3] += __int_as_float(D[i][j][3]);
#elif PF_EPI2
                    acc[i][j][0] = fmaf(d0v[i], fmaf(s0[j].x, __int_as_float(D[i][j][0]), -c0[j]), acc[i][j][0]);
                    acc[i][j][1] = fmaf(d0v[i], fmaf(s1[j].x, __int_as_float(D[i][j][1]), -c1[j]), acc[i][j][1]);
                    acc[i][j][2] = fmaf(d1v[i], fmaf(s0[j].x, __int_as_float(D[i][j][2]), -c0[j]), acc[i][j][2]);
                    acc[i][j][3] = fmaf(d1v[i], fmaf(s1[j].x, __int_as_float(D[i][j][3]), -c1[j]), acc[i][j][3]);
#else
                    const float f0 = __int_as_float(D[i][j][0]) - PF_MAGIC_F;
                    const float f1 = __int_as_float(D[i][j][1]) - PF_MAGIC_F;
                    const float f2 = __int_as_float(D[i][j][2]) - PF_MAGIC_F;
                    const float f3 = __int_as_float(D[i][j][3]) - PF_MAGIC_F;
                    // d4 * (d8*sumi - 8*sum), accumulated in block order.
                    acc[i][j][0] = fmaf(d0v[i], fmaf(s0[j].x, f0, -s0[j].y), acc[i][j][0]);
                    acc[i][j][1] = fmaf(d0v[i], fmaf(s1[j].x, f1, -s1[j].y), acc[i][j][1]);
                    acc[i][j][2] = fmaf(d1v[i], fmaf(s0[j].x, f2, -s0[j].y), acc[i][j][2]);
                    acc[i][j][3] = fmaf(d1v[i], fmaf(s1[j].x, f3, -s1[j].y), acc[i][j][3]);
#endif
                }
        }
    }
    pf_wait<0>();

    if constexpr (EPI == 1) {
        // ---- SwiGLU epilogue -------------------------------------------
        // Up values (warps wm == 1) go through shared memory to the warps
        // holding the gate values for the same (row, token).  Then each
        // (token, 32-row half) is one Q8_1 block: its 32 values live in the 8
        // lanes sharing t4, 4 per lane, so amax and sum are three xor-shuffles.
        __syncthreads();                                  // pipeline smem is free
        float  * sU = reinterpret_cast<float *>(pf_smem);              // [64][BN]
        int8_t * sQ = reinterpret_cast<int8_t *>(pf_smem + 64 * BN * 4); // [BN][64]
        float2 * sS = reinterpret_cast<float2 *>(pf_smem + 64 * BN * 5); // [BN][2]
        if (wm == 1) {
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int lr = i * 16 + g8 + ((e & 2) ? 8 : 0);
                        const int n  = wn * (BN / WN) + j * 8 + 2 * t4 + (e & 1);
                        sU[lr * BN + n] = acc[i][j][e] * s_scl[n];
                    }
        }
        __syncthreads();
        if (wm == 0) {
            static_assert(MT == 4, "two Q8_1 blocks of 32 rows per warp");
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int n = wn * (BN / WN) + j * 8 + 2 * t4 + c;
                        float v[4];
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const int ii = 2 * h + (q >> 1);
                            const int rr = (q & 1) ? 8 : 0;
                            const int lr = ii * 16 + g8 + rr;
                            const float gt = acc[ii][j][c + ((q & 1) ? 2 : 0)] * s_scl[n];
                            v[q] = silu_f(gt) * sU[lr * BN + n];
                        }
                        float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
                        float sm = (v[0] + v[1]) + (v[2] + v[3]);
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) {
                            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                            sm += __shfl_xor_sync(0xffffffffu, sm, o);
                        }
                        const float d = am / 127.0f;
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const int lr = (2 * h + (q >> 1)) * 16 + g8 + ((q & 1) ? 8 : 0);
                            sQ[n * 64 + lr] = (am == 0.0f) ? (int8_t) 0 : (int8_t) roundf(v[q] / d);
                        }
                        if (g8 == 0) {
                            const __half2 ds = __floats2half2_rn(d, sm);
                            sS[n * 2 + h] = make_float2(__low2float(ds), 8.0f * __high2float(ds));
                        }
                    }
        }
        __syncthreads();
        for (int c = threadIdx.x; c < BN * 4; c += 256) {
            const int n = c >> 2, k = c & 3;
            const int orow = s_orow[n];
            if (orow < 0) continue;
            *reinterpret_cast<int4 *>(g.oqs + (size_t) orow * (g.okb * 32) + row0 + k * 16) =
                *reinterpret_cast<const int4 *>(sQ + n * 64 + k * 16);
            if (k < 2) g.osc[(size_t) orow * g.okb + row0 / 32 + k] = sS[n * 2 + k];
        }
    } else {
    // Epilogue: thread holds rows (r0, r0+8) x tokens (nc, nc+1) per tile.
#pragma unroll
    for (int i = 0; i < MT; ++i) {
        const int r0 = row0 + wm * (BM / WM) + i * 16 + g8;
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int nc = wn * (BN / WN) + j * 8 + 2 * t4;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int n = nc + (e & 1);
                const int r = r0 + ((e & 2) ? 8 : 0);
                const int orow = s_orow[n];
                if (orow < 0) continue;
                const float a = acc[i][j][e] * s_scl[n];
                float * p = g.out + (size_t) orow * g.out_ld + r;
                *p = g.add ? (*p + a) : a;
            }
        }
    }
    }
}

// ---------------------------------------------------------------------------
// Warp-specialised variant: one PRODUCER warp issues every cp.async, eight
// CONSUMER warps only compute, and they hand stages over through mbarriers
// (full[s]: the producer's copies into stage s have landed; empty[s]: all eight
// consumers are done reading it).  No block-wide barrier in the main loop, so
// a consumer that finishes a stage early starts the next one as soon as its
// data is there, and the copies overlap the math continuously instead of in
// lock-step.  Same tile maths and epilogue as k_pf_mmq (EPI 0).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void pf_mb_init(uint64_t * b, int count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"((unsigned) __cvta_generic_to_shared(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void pf_mb_arrive(uint64_t * b) {
    asm volatile("{ .reg .b64 st; mbarrier.arrive.shared.b64 st, [%0]; }\n" :: "r"((unsigned) __cvta_generic_to_shared(b)) : "memory");
}
__device__ __forceinline__ void pf_mb_cp_arrive(uint64_t * b) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"((unsigned) __cvta_generic_to_shared(b)) : "memory");
}
__device__ __forceinline__ void pf_mb_wait(uint64_t * b, unsigned parity) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(b);
    unsigned done;
    do {
        asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }\n"
                     : "=r"(done) : "r"(a), "r"(parity) : "memory");
    } while (!done);
}

template <int BN, int NST, int KB, int MINB, int NPW>
__global__ void __launch_bounds__(256 + 32 * NPW, MINB) k_pf_mmq_ws(const PfGemm g) {
    constexpr int NTH = 256 + 32 * NPW, PL = 32 * NPW;   // PL producer lanes
    constexpr int BM = 128, WM = 2, WN = 4;
    using T = PfTile<BM, BN, KB>;
    constexpr int MT = (BM / WM) / 16;
    constexpr int NT = (BN / WN) / 8;
    extern __shared__ __align__(16) unsigned char pf_smem[];
    __shared__ int   s_arow[BN];
    __shared__ int   s_orow[BN];
    __shared__ float s_scl[BN];
    __shared__ __align__(8) uint64_t mb_full[NST], mb_empty[NST];

    int rt, tstart, tcnt, expert = 0;
    if (g.perm) {
        if ((int) blockIdx.y >= *g.ntiles) return;
        const int4 tl = g.tiles[blockIdx.y];
        expert = tl.x; tstart = tl.y; tcnt = tl.z; rt = blockIdx.x;
    } else {
        const int RT = g.N / BM;
        const int TT = (g.ntok + BN - 1) / BN;
        const int L  = blockIdx.x;
        const int GM = 8;
        const int npg   = GM * TT;
        const int grp   = L / npg;
        const int first = grp * GM;
        const int gsz   = (RT - first < GM) ? (RT - first) : GM;
        rt = first + (L % npg) % gsz;
        const int tt = (L % npg) / gsz;
        tstart = tt * BN;
        tcnt = g.ntok - tstart; if (tcnt > BN) tcnt = BN;
    }
    for (int i = threadIdx.x; i < BN; i += NTH) {
        int slot = -1;
        if (i < tcnt) slot = g.perm ? g.perm[tstart + i] : tstart + i;
        const int ar = (slot >= 0) ? (slot >> g.a_shift) : -1;
        s_arow[i] = ar;
        s_orow[i] = slot;
        s_scl[i]  = (ar >= 0 && g.scale) ? g.scale[ar] : 1.0f;
    }
    if (threadIdx.x == 0) {
        for (int s = 0; s < NST; ++s) { pf_mb_init(&mb_full[s], PL); pf_mb_init(&mb_empty[s], 8); }
    }
    __syncthreads();

    const int row0 = rt * BM;
    const uint4 * wq; const __half * wd;
    {
        const size_t eo = (size_t) expert * g.exp_stride;
        const int seg  = (row0 < g.seg_end[0]) ? 0 : (row0 < g.seg_end[1]) ? 1 : 2;
        const int sbase = seg ? g.seg_end[seg - 1] : 0;
        wq = g.qs[seg] + eo + (size_t) (row0 - sbase) * g.bpr;
        wd = g.d[seg]  + eo + (size_t) (row0 - sbase) * g.bpr;
    }
    const int nkb = g.bpr / KB;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

    if (warp >= 8) {
        // ---- producers ----
        const int pl = threadIdx.x - 256;
        for (int it = 0; it < nkb; ++it) {
            const int st = it % NST;
            if (it >= NST) pf_mb_wait(&mb_empty[st], ((it / NST) - 1) & 1);
            unsigned char * base = pf_smem + st * T::STAGE;
            const int kb0 = it * KB;
#pragma unroll 4
            for (int c = pl; c < BM * KB; c += PL) {
                const int r = c / KB, kk = c % KB;
                pf_cp16(base + r * T::A_P + kk * 16, wq + (size_t) r * g.bpr + kb0 + kk, true);
            }
#pragma unroll
            for (int r = pl; r < BM; r += PL) {
                if (KB * 2 == 16)     pf_cp16(base + T::SA + r * 16, wd + (size_t) r * g.bpr + kb0, true);
                else if (KB * 2 == 8) pf_cp8(base + T::SA + r * 8, wd + (size_t) r * g.bpr + kb0);
                else                  pf_cp4(base + T::SA + r * 4, wd + (size_t) r * g.bpr + kb0);
            }
#pragma unroll 4
            for (int c = pl; c < BN * KB * 2; c += PL) {
                const int r = c / (KB * 2), kk = c % (KB * 2);
                const int ar = s_arow[r];
                pf_cp16(base + T::SA + T::SD + r * T::B_P + kk * 16,
                        g.aqs + (size_t) (ar < 0 ? 0 : ar) * (g.bpr * 32) + kb0 * 32 + kk * 16, ar >= 0);
            }
#pragma unroll
            for (int c = pl; c < BN * (KB / 2); c += PL) {
                const int r = c / (KB / 2), kk = c % (KB / 2);
                const int ar = s_arow[r];
                pf_cp16(base + T::SA + T::SD + T::SB + r * (T::S_P * 8) + kk * 16,
                        g.asc + (size_t) (ar < 0 ? 0 : ar) * g.bpr + kb0 + kk * 2, ar >= 0);
            }
            pf_mb_cp_arrive(&mb_full[st]);
        }
        pf_wait<0>();
        return;
    }

    // ---- consumers ----
    const int wm = warp % WM, wn = warp / WM;
    const int g8 = lane >> 2, t4 = lane & 3;
    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    for (int it = 0; it < nkb; ++it) {
        const int st = it % NST;
        pf_mb_wait(&mb_full[st], (it / NST) & 1);
        const unsigned char * pA = pf_smem + st * T::STAGE;
        const unsigned char * pD = pA + T::SA;
        const unsigned char * pB = pD + T::SD;
        const unsigned char * pS = pB + T::SB;
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) {
            unsigned bf[NT][2];
            float2   s0[NT], s1[NT];
            float    c0[NT], c1[NT];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int nl = wn * (BN / WN) + j * 8 + g8;
                const unsigned char * pb = pB + nl * T::B_P + kb * 32 + t4 * 4;
                bf[j][0] = *reinterpret_cast<const unsigned *>(pb);
                bf[j][1] = *reinterpret_cast<const unsigned *>(pb + 16);
                const int nc = wn * (BN / WN) + j * 8 + 2 * t4;
                s0[j] = reinterpret_cast<const float2 *>(pS + nc * (T::S_P * 8))[kb];
                s1[j] = reinterpret_cast<const float2 *>(pS + (nc + 1) * (T::S_P * 8))[kb];
                c0[j] = fmaf(s0[j].x, PF_MAGIC_F, s0[j].y);
                c1[j] = fmaf(s1[j].x, PF_MAGIC_F, s1[j].y);
            }
            unsigned af[MT][4];
            float d0v[MT], d1v[MT];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r0 = wm * (BM / WM) + i * 16 + g8;
                const unsigned w0 = *reinterpret_cast<const unsigned *>(pA + r0 * T::A_P + kb * 16 + t4 * 4);
                const unsigned w1 = *reinterpret_cast<const unsigned *>(pA + (r0 + 8) * T::A_P + kb * 16 + t4 * 4);
                af[i][0] = w0 & 0x0F0F0F0Fu;        af[i][1] = w1 & 0x0F0F0F0Fu;
                af[i][2] = (w0 >> 4) & 0x0F0F0F0Fu; af[i][3] = (w1 >> 4) & 0x0F0F0F0Fu;
                d0v[i] = __half2float(reinterpret_cast<const __half *>(pD + r0 * (KB * 2))[kb]);
                d1v[i] = __half2float(reinterpret_cast<const __half *>(pD + (r0 + 8) * (KB * 2))[kb]);
            }
            int D[MT][NT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j) pf_mma_s8(D[i][j], af[i], bf[j]);
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    acc[i][j][0] = fmaf(d0v[i], fmaf(s0[j].x, __int_as_float(D[i][j][0]), -c0[j]), acc[i][j][0]);
                    acc[i][j][1] = fmaf(d0v[i], fmaf(s1[j].x, __int_as_float(D[i][j][1]), -c1[j]), acc[i][j][1]);
                    acc[i][j][2] = fmaf(d1v[i], fmaf(s0[j].x, __int_as_float(D[i][j][2]), -c0[j]), acc[i][j][2]);
                    acc[i][j][3] = fmaf(d1v[i], fmaf(s1[j].x, __int_as_float(D[i][j][3]), -c1[j]), acc[i][j][3]);
                }
        }
        __syncwarp();
        if (lane == 0) pf_mb_arrive(&mb_empty[st]);
    }

#pragma unroll
    for (int i = 0; i < MT; ++i) {
        const int r0 = row0 + wm * (BM / WM) + i * 16 + g8;
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int nc = wn * (BN / WN) + j * 8 + 2 * t4;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int n = nc + (e & 1);
                const int r = r0 + ((e & 2) ? 8 : 0);
                const int orow = s_orow[n];
                if (orow < 0) continue;
                const float a = acc[i][j][e] * s_scl[n];
                float * p = g.out + (size_t) orow * g.out_ld + r;
                *p = g.add ? (*p + a) : a;
            }
        }
    }
}

template <int BN, int NST, int KB, int MINB, int NPW = 1>
static void pf_mmq_ws_launch(const PfGemm & g, dim3 grid, cudaStream_t st) {
    constexpr int SMEM = PfTile<128, BN, KB>::STAGE * NST;
    static bool attr = false;
    if (!attr) {
        PF_CK(cudaFuncSetAttribute(k_pf_mmq_ws<BN, NST, KB, MINB, NPW>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        attr = true;
    }
    k_pf_mmq_ws<BN, NST, KB, MINB, NPW><<<grid, 256 + 32 * NPW, SMEM, st>>>(g);
    PF_CK(cudaGetLastError());
}

template <int BN, int NST, int KB, int MINB, int WM, int EPI = 0, int NTH = 256, int BM = 128>
static void pf_mmq_launch(const PfGemm & g, dim3 grid, cudaStream_t st) {
    constexpr int SMEM = PfTile<BM, BN, KB>::STAGE * NST;
    static_assert(EPI == 0 || SMEM >= 64 * BN * 5 + BN * 16, "SwiGLU epilogue staging must fit");
    static bool attr = false;
    if (!attr) {
        PF_CK(cudaFuncSetAttribute(k_pf_mmq<BN, NST, KB, MINB, WM, EPI, NTH, BM>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        attr = true;
    }
    k_pf_mmq<BN, NST, KB, MINB, WM, EPI, NTH, BM><<<grid, NTH, SMEM, st>>>(g);
    PF_CK(cudaGetLastError());
}

// Launch configurations, selectable at run time for tuning (LK_PF_CFG_DENSE,
// LK_PF_CFG_MOE).  The token tile is the config's BN.
//   0: BN128 NST2 KB4   1 block/SM
//   1: BN64  NST3 KB4   1 block/SM
//   2: BN64  NST2 KB4   2 blocks/SM
//   3: BN64  NST2 KB8   1 block/SM
//   4: BN32  NST3 KB4   2 blocks/SM
//   5: BN64  NST2 KB4   2 blocks/SM  warps 4x2 (32 rows x 32 tokens each)
//   6: BN128 NST2 KB4   1 block/SM   warps 4x2 (32 rows x 64 tokens each)
//   7: BN64  NST3 KB4   1 block/SM   warps 4x2
//   8: BN128 NST2 KB4   1 block/SM   warps 2x4, but reg-capped for 2 blocks
//   9: BN64  NST4 KB4   16 warps 4x4 (32 rows x 16 tokens each)
//  10: BN64  NST3 KB4   16 warps 4x4
//  11: BN128 NST2 KB4   16 warps 4x4 (32 rows x 32 tokens each)
//  12: BN64  NST4 KB4   16 warps 8x2 (16 rows x 32 tokens each)
//  13: BN128 NST4 KB2   1 block/SM, warps 2x4 (half-size stages, deeper pipeline)
//  14: BN128 NST3 KB2   1 block/SM, warps 2x4
//  15: BN64  NST3 KB2   2 blocks/SM, warps 2x4
//  16: BN128 NST4 KB2   16 warps 4x4
//  17: BM256 BN64  NST2 KB4  16 warps 4x4 (64 rows x 16 tokens each, as cfg 2)
//  18: BM256 BN64  NST3 KB2  16 warps 4x4
//  19: BM256 BN128 NST2 KB4  16 warps 4x4 (64 rows x 32 tokens each)
//
// MEASURED (dense 6144x2048x4096 / MoE gate|up 16384 slots, TMAC/s, one 3090 Ti;
// the card's int8 tensor peak measures 158.7):
//   cfg 2   62.5 / 58.7   <- the default for every GEMM
//   cfg 5   60.8 / 54.4    cfg 0  58.3 / 59.2    cfg 6  61.9 / 59.0
//   cfg 9   50.8 / 48.6    cfg 11 58.8 / -       cfg 13 55.6 / -
//   cfg 15  58.3 / 54.9    cfg 17 50.5 / 47.6    cfg 18 46.2 / 44.2
//   cfg 19  13.5 (spills)
// Why the ceiling: with no math at all (copies only) cfg 2 runs at the
// equivalent of 84 TMAC/s, i.e. ~1.57 TB/s from L2 into shared memory against a
// measured 1.73 TB/s wall, and the copy, the mma and the 3-op fp32 epilogue per
// value do not fully overlap.  Taller tiles cut the L2 traffic but put 16 warps
// behind each stage barrier, which costs more than the traffic saves; two
// independent 8-warp blocks per SM overlap one block's copy with the other's
// math.
//
// 2026-10-22, with PF_EPI2 (2 FP ops per value): cfg 2 67.3 / 61.4.  Ablations
// (cfg 2, dense 6144x2048x4096): copies only 84, FP work only 97, mma + 1 FADD
// 113, all three 67 -- each part alone is well above the whole, i.e. the loss
// is in how they overlap.  Producer/consumer (k_pf_mmq_ws, cfgs 20-28: 1, 2 or 4
// copy warps feeding 8 compute warps through mbarriers) was built to fix that
// and MEASURED SLOWER in every shape: 1 copy warp 50-57, 2 warps 62.1, 4 warps
// 57.1 (cfg 20 38-44, 1 block/SM).  Without a bulk-copy engine (TMA is sm_90)
// the copy warps cannot issue requests fast enough, which is why Ampere GEMMs
// (CUTLASS sm80) use the all-threads multistage pipeline this kernel has.
//  20-25: k_pf_mmq_ws, 1 copy warp  (BN, NST, KB, blocks/SM) = (64,4,4,1) (64,2,4,2)
//         (64,3,2,2) (128,4,2,1) (128,3,2,1) (64,6,2,1)
//  26-28: 2 copy warps (64,2,4,2) (64,3,2,2); 4 copy warps (64,4,4,1)
static int pf_cfg_bn(int c) { return (c == 0 || c == 6 || c == 8 || c == 11 || c == 13 || c == 14 || c == 16 || c == 19 || c == 23 || c == 24) ? 128 : (c == 4) ? 32 : 64; }
static int pf_cfg_bm(int c) { return (c >= 17 && c <= 19) ? 256 : 128; }
static void pf_mmq_cfg(int c, const PfGemm & g, dim3 grid, cudaStream_t st) {
    switch (c) {
    case 0: pf_mmq_launch<128, 2, 4, 1, 2>(g, grid, st); break;
    case 1: pf_mmq_launch<64, 3, 4, 1, 2>(g, grid, st); break;
    case 2: pf_mmq_launch<64, 2, 4, 2, 2>(g, grid, st); break;
    case 3: pf_mmq_launch<64, 2, 8, 1, 2>(g, grid, st); break;
    case 4: pf_mmq_launch<32, 3, 4, 2, 2>(g, grid, st); break;
    case 5: pf_mmq_launch<64, 2, 4, 2, 4>(g, grid, st); break;
    case 6: pf_mmq_launch<128, 2, 4, 1, 4>(g, grid, st); break;
    case 7: pf_mmq_launch<64, 3, 4, 1, 4>(g, grid, st); break;
    case 8: pf_mmq_launch<128, 2, 4, 1, 2>(g, grid, st); break;
    case 9: pf_mmq_launch<64, 4, 4, 1, 4, 0, 512>(g, grid, st); break;
    case 10: pf_mmq_launch<64, 3, 4, 1, 4, 0, 512>(g, grid, st); break;
    case 11: pf_mmq_launch<128, 2, 4, 1, 4, 0, 512>(g, grid, st); break;
    case 12: pf_mmq_launch<64, 4, 4, 1, 8, 0, 512>(g, grid, st); break;
    case 13: pf_mmq_launch<128, 4, 2, 1, 2>(g, grid, st); break;
    case 14: pf_mmq_launch<128, 3, 2, 1, 2>(g, grid, st); break;
    case 15: pf_mmq_launch<64, 3, 2, 2, 2>(g, grid, st); break;
    case 16: pf_mmq_launch<128, 4, 2, 1, 4, 0, 512>(g, grid, st); break;
    case 17: pf_mmq_launch<64, 2, 4, 1, 4, 0, 512, 256>(g, grid, st); break;
    case 18: pf_mmq_launch<64, 3, 2, 1, 4, 0, 512, 256>(g, grid, st); break;
    case 19: pf_mmq_launch<128, 2, 4, 1, 4, 0, 512, 256>(g, grid, st); break;
    case 20: pf_mmq_ws_launch<64, 4, 4, 1>(g, grid, st); break;
    case 21: pf_mmq_ws_launch<64, 2, 4, 2>(g, grid, st); break;
    case 22: pf_mmq_ws_launch<64, 3, 2, 2>(g, grid, st); break;
    case 23: pf_mmq_ws_launch<128, 4, 2, 1>(g, grid, st); break;
    case 24: pf_mmq_ws_launch<128, 3, 2, 1>(g, grid, st); break;
    case 25: pf_mmq_ws_launch<64, 6, 2, 1>(g, grid, st); break;
    case 26: pf_mmq_ws_launch<64, 2, 4, 2, 2>(g, grid, st); break;
    case 27: pf_mmq_ws_launch<64, 3, 2, 2, 2>(g, grid, st); break;
    case 28: pf_mmq_ws_launch<64, 4, 4, 1, 4>(g, grid, st); break;
    default: fprintf(stderr, "bad mmq config %d\n", c); exit(1);
    }
}
static int pf_env_cfg(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

void pf_gemm_dense(const PfGemm & g, cudaStream_t st) {
    if (g.N % PF_BM || g.bpr % 8) { fprintf(stderr, "pf_gemm_dense: bad shape N=%d bpr=%d\n", g.N, g.bpr); exit(1); }
    static const int forced = pf_env_cfg("LK_PF_CFG_DENSE", -1);
    int c = (forced >= 0) ? forced : 2;
    // A 256-row tile must not straddle a weight segment; fall back if it would.
    if (pf_cfg_bm(c) == 256 && (g.N % 256 || g.seg_end[0] % 256 || g.seg_end[1] % 256)) c = 2;
    const int BN = pf_cfg_bn(c), BM = pf_cfg_bm(c);
    pf_mmq_cfg(c, g, dim3((g.N / BM) * ((g.ntok + BN - 1) / BN)), st);
}

// The MoE plan tiles each expert's slots in PF_MOE_BN; a config with a smaller
// BN would leave rows uncomputed, so only BN == PF_MOE_BN configs are legal.
void pf_gemm_moe(const PfGemm & g, int max_tiles, cudaStream_t st) {
    if (g.N % PF_BM || g.bpr % 8) { fprintf(stderr, "pf_gemm_moe: bad shape N=%d bpr=%d\n", g.N, g.bpr); exit(1); }
    static const int forced = pf_env_cfg("LK_PF_CFG_MOE", -1);
    const int c = (forced >= 0) ? forced : 2;
    if (pf_cfg_bn(c) != PF_MOE_BN) { fprintf(stderr, "LK_PF_CFG_MOE=%d has BN != %d\n", c, PF_MOE_BN); exit(1); }
    int cc = c;
    if (pf_cfg_bm(cc) == 256 && (g.N % 256 || g.seg_end[0] % 256 || g.seg_end[1] % 256)) cc = 2;
    pf_mmq_cfg(cc, g, dim3(g.N / pf_cfg_bm(cc), max_tiles), st);
}

// SwiGLU-fused gate|up: g.qs[0]/d[0] = gate, g.qs[1]/d[1] = up, g.N = the
// intermediate width; writes Q8_1 activations to g.oqs/g.osc (g.okb blocks per
// row).  Same tiling as cfg 2 (BN 64, 2 blocks/SM).
void pf_gemm_swiglu(const PfGemm & g, int max_tiles, cudaStream_t st) {
    if (g.N % 64 || g.bpr % 8) { fprintf(stderr, "pf_gemm_swiglu: bad shape N=%d bpr=%d\n", g.N, g.bpr); exit(1); }
    static const int v = pf_env_cfg("LK_PF_CFG_SWIGLU", 0);
    const dim3 grid = g.perm ? dim3(g.N / 64, max_tiles) : dim3((g.N / 64) * ((g.ntok + 63) / 64));
    if (v == 1) pf_mmq_launch<64, 3, 4, 1, 2, 1>(g, grid, st);      // no register cap
    else        pf_mmq_launch<64, 2, 4, 2, 2, 1>(g, grid, st);
}

// ---------------------------------------------------------------------------
// Per-token kernels.  Each is the decode kernel's arithmetic for one token,
// applied to a chunk of tokens.
// ---------------------------------------------------------------------------
// grid (8, m) x 256: k_embed_q6_K for token toks[blockIdx.y].
__global__ void k_pf_embed(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                           const int8_t * __restrict__ sc, const __half * __restrict__ d,
                           const int * __restrict__ toks, float * __restrict__ x) {
    const int t   = blockIdx.y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = toks[t];
    const int sb = tid / 256;
    const int e  = tid % 256;
    const size_t blk = (size_t) token * 8 + sb;
    const float dk = __half2float(d[blk]);
    const int q6 = q6_at(ql + blk * 128, qh + blk * 64, e);
    x[(size_t) t * LK_HIDDEN + tid] = dk * (float) sc[blk * 16 + e / 16] * (float) (q6 - 32);
}

// grid m x 256.
__global__ void __launch_bounds__(256) k_pf_norm(const float * __restrict__ X, const float * __restrict__ w,
                                                 int plain, PfAct a, float * __restrict__ inv_out,
                                                 float * __restrict__ h32) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float * x = X + (size_t) t * LK_HIDDEN;
    int8_t * q = a.qs + (size_t) t * LK_HIDDEN;
    float2 * s = a.sc + (size_t) t * (LK_HIDDEN / 32);
    __shared__ float s_sq[256];
    if (plain) {
        // k_rmsnorm_q8_1, n == 2048 path.
        float xv[8], wv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { xv[j] = x[tid + j * 256]; wv[j] = w[tid + j * 256]; }
        float sq = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) sq += xv[j] * xv[j];
        s_sq[tid] = sq;
        __syncthreads();
#pragma unroll
        for (int o = 128; o > 16; o >>= 1) {
            if (tid < o) s_sq[tid] += s_sq[tid + o];
            __syncthreads();
        }
        if (tid < 32) {
            float v = s_sq[tid];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (tid == 0) s_sq[0] = v;
        }
        __syncthreads();
        const float inv = rsqrtf(s_sq[0] / 2048.0f + LK_EPS);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float tv = xv[j] * inv * wv[j];
            const int b = warp + 8 * j;
            pf_quant_warp(tv, q + b * 32, s + b, lane);
        }
        if (tid == 0) inv_out[t] = 1.0f;
        return;
    }
    // Phase 4y: lk_norm_prod_warp for every block, then lk_norm_inv.
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int b = warp + 8 * j;
        const int i = b * 32 + lane;
        const float xv = x[i];
        const float uv = xv * w[i];
        if (h32) h32[(size_t) t * LK_HIDDEN + i] = uv;
        pf_quant_warp(uv, q + b * 32, s + b, lane);
        float sq = xv * xv;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (lane == 0) s_sq[b] = sq;
    }
    __syncthreads();
    if (warp == 0) {
        float sv = s_sq[lane] + s_sq[lane + 32];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sv += __shfl_xor_sync(0xffffffffu, sv, o);
        if (lane == 0) inv_out[t] = rsqrtf(sv * (1.0f / (float) LK_HIDDEN) + LK_EPS);
    }
}

// grid (m, 8) x 256: one warp per (token, 32-channel block).  Y is the in_proj
// output [m][6144] = B | C | x.  hist is the state BEFORE this chunk.
__global__ void __launch_bounds__(256) k_pf_conv(const float * __restrict__ Y, const float * __restrict__ cw,
                                                 const float * __restrict__ hist, PfAct a) {
    const int t = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.y * 8 + warp;
    const int ch = b * 32 + lane;
    const float * yt = Y + (size_t) t * 6144;
    const float bx = yt[ch] * yt[4096 + ch];
    float h0, h1;
    if (t >= 2)      { const float * y2 = yt - 2 * 6144; h0 = y2[ch] * y2[4096 + ch]; }
    else if (t == 1) { h0 = hist[ch * 2 + 1]; }
    else             { h0 = hist[ch * 2]; }
    if (t >= 1)      { const float * y1 = yt - 6144; h1 = y1[ch] * y1[4096 + ch]; }
    else             { h1 = hist[ch * 2 + 1]; }
    const float cw0 = cw[0 + ch * 3];
    const float cw1 = cw[1 + ch * 3];
    const float cw2 = cw[2 + ch * 3];
    const float conv = h0 * cw0 + h1 * cw1 + bx * cw2;
    const float v = yt[2048 + ch] * conv;
    pf_quant_warp(v, a.qs + (size_t) t * LK_HIDDEN + b * 32, a.sc + (size_t) t * 64 + b, lane);
}

// grid 8 x 256.  Runs after k_pf_conv has read the old state.
__global__ void k_pf_conv_hist(const float * __restrict__ Y, int m, float * __restrict__ hist) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const float * yl = Y + (size_t) (m - 1) * 6144;
    const float b1 = yl[ch] * yl[4096 + ch];
    float b0;
    if (m >= 2) { const float * yp = Y + (size_t) (m - 2) * 6144; b0 = yp[ch] * yp[4096 + ch]; }
    else        { b0 = hist[ch * 2 + 1]; }
    hist[ch * 2]     = b0;
    hist[ch * 2 + 1] = b1;
}

// grid (m, 40) x 64: k_qk_norm_rope for token t at position P + t.  Y is the
// q|k|v GEMM output [m][3072].  Q goes out as fp16 pre-scaled by 1/8 (exact),
// ready for the tensor-core attention; K and V go into the cache exactly as the
// decode kernel writes them.
__global__ void k_pf_qknr(const float * __restrict__ Y, const float * __restrict__ qn,
                          const float * __restrict__ kn, int P,
                          __half * __restrict__ qh, __half * __restrict__ kc, __half * __restrict__ vc) {
    const int t = blockIdx.x;
    const int blk = blockIdx.y;
    const int tid = threadIdx.x;
    const float pos = (float) (P + t);
    __shared__ float s_head[LK_HEAD_DIM];
    __shared__ float s_red[LK_HEAD_DIM];
    const bool is_q = (blk < LK_N_HEADS);
    const int head = is_q ? blk : (blk - LK_N_HEADS);
    const float * src = Y + (size_t) t * 3072 + (is_q ? head * LK_HEAD_DIM : 2048 + head * LK_HEAD_DIM);
    const float * nw = is_q ? qn : kn;

    const float v0 = src[tid];
    s_red[tid] = v0 * v0;
    __syncthreads();
    for (int o = LK_HEAD_DIM / 2; o > 0; o >>= 1) {
        if (tid < o) s_red[tid] += s_red[tid + o];
        __syncthreads();
    }
    const float inv = rsqrtf(s_red[0] / (float) LK_HEAD_DIM + LK_EPS);
    s_head[tid] = v0 * inv * nw[tid];
    __syncthreads();
    if (tid < LK_HEAD_DIM / 2) {
        const float invf = powf(LK_ROPE_THETA, -2.0f * (float) tid / (float) LK_HEAD_DIM);
        float c, s;
        sincosf(pos * invf, &s, &c);
        const float x0 = s_head[tid], x1 = s_head[tid + LK_HEAD_DIM / 2];
        s_head[tid]                   = x0 * c - x1 * s;
        s_head[tid + LK_HEAD_DIM / 2] = x0 * s + x1 * c;
    }
    __syncthreads();
    const float r = s_head[tid];
    if (is_q) {
        qh[(size_t) t * 2048 + head * LK_HEAD_DIM + tid] = __float2half(r * LK_ATTN_SCALE);
    } else {
        const size_t off = ((size_t) (P + t) * LK_N_KV_HEADS + head) * LK_HEAD_DIM;
        kc[off + tid] = __float2half(r);
        vc[off + tid] = __float2half(Y[(size_t) t * 3072 + 2560 + head * LK_HEAD_DIM + tid]);
    }
}

// ---------------------------------------------------------------------------
// Causal GQA flash attention, fp16 tensor cores, fp32 accumulation.
//
// grid (ceil(m/32), 8 kv heads) x 256.  A block owns 32 query positions of ONE
// kv head and all four query heads that share it: warp w computes head
// kvh*4 + (w&3) for positions q0 + (w>>2)*16 .. +15.  Every K/V tile it loads
// therefore serves 128 query rows.  Keys are 64 per tile, double-buffered with
// cp.async, rows XOR-swizzled in 16-byte chunks so ldmatrix is conflict-free.
// Scores live in the log2 domain (Q carries the exact 1/8; log2(e) is applied
// in fp32) and the softmax is the usual online one.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 2) k_pf_attn(const __half * __restrict__ qh, int m, int P,
                                                    const __half * __restrict__ kc,
                                                    const __half * __restrict__ vc,
                                                    float * __restrict__ o) {
    __shared__ __align__(128) __half sK[2][64 * 64];
    __shared__ __align__(128) __half sV[2][64 * 64];
    const int qt   = gridDim.x - 1 - blockIdx.x;      // longest rows first
    const int kvh  = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t4 = lane & 3;
    const int head = kvh * 4 + (warp & 3);
    const int q0 = qt * 32 + (warp >> 2) * 16;
    const int qlast = (qt * 32 + 32 < m) ? (qt * 32 + 32) : m;
    const int kend = P + qlast;
    const int nkt = (kend + 63) / 64;

    unsigned qa[4][4];
    {
        const int r0 = q0 + g, r1 = q0 + g + 8;
        const __half * Q0 = qh + (size_t) r0 * 2048 + head * 64;
        const __half * Q1 = qh + (size_t) r1 * 2048 + head * 64;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const int c = kk * 16 + 2 * t4;
            qa[kk][0] = (r0 < m) ? *reinterpret_cast<const unsigned *>(Q0 + c)     : 0u;
            qa[kk][1] = (r1 < m) ? *reinterpret_cast<const unsigned *>(Q1 + c)     : 0u;
            qa[kk][2] = (r0 < m) ? *reinterpret_cast<const unsigned *>(Q0 + c + 8) : 0u;
            qa[kk][3] = (r1 < m) ? *reinterpret_cast<const unsigned *>(Q1 + c + 8) : 0u;
        }
    }
    auto load = [&](int st, int kt) {
#pragma unroll
        for (int i = threadIdx.x; i < 64 * 8 * 2; i += 256) {
            const int isv = i >> 9;
            const int r = (i >> 3) & 63;
            const int c = i & 7;
            const int kpos = kt * 64 + r;
            const bool ok = kpos < kend;
            const __half * src = (isv ? vc : kc) + ((size_t) (ok ? kpos : 0) * LK_N_KV_HEADS + kvh) * 64 + c * 8;
            __half * dst = (isv ? sV[st] : sK[st]) + r * 64 + ((c ^ (r & 7)) * 8);
            pf_cp16(dst, src, ok);
        }
    };

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f; }
    float mrow0 = -INFINITY, mrow1 = -INFINITY, lrow0 = 0.0f, lrow1 = 0.0f;
    const float L2E = 1.4426950408889634f;
    const int qpos0 = P + q0 + g, qpos1 = qpos0 + 8;

    load(0, 0);
    pf_commit();
    for (int kt = 0; kt < nkt; ++kt) {
        if (kt + 1 < nkt) load((kt + 1) & 1, kt + 1);
        pf_commit();
        pf_wait<1>();
        __syncthreads();
        const __half * K = sK[kt & 1];
        const __half * V = sV[kt & 1];

        float s[8][4];
        const int mi = lane >> 3, ri = lane & 7;
#if PF_ATTN_QK_F16
        // EXPERIMENT: scores accumulated in fp16, widened afterwards.
        {
            unsigned sh[8][2];
#pragma unroll
            for (int j = 0; j < 8; ++j) { sh[j][0] = 0u; sh[j][1] = 0u; }
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
                for (int jp = 0; jp < 4; ++jp) {
                    const int key = 16 * jp + 8 * (mi >> 1) + ri;
                    const int ch  = 2 * kk + (mi & 1);
                    unsigned kb[4];
                    pf_ldsm4(kb, K + key * 64 + ((ch ^ (key & 7)) * 8));
                    const unsigned b0[2] = { kb[0], kb[1] };
                    const unsigned b1[2] = { kb[2], kb[3] };
                    pf_mma_f16h(sh[2 * jp],     qa[kk], b0);
                    pf_mma_f16h(sh[2 * jp + 1], qa[kk], b1);
                }
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&sh[j][0]));
                const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&sh[j][1]));
                s[j][0] = lo.x; s[j][1] = lo.y; s[j][2] = hi.x; s[j][3] = hi.y;
            }
        }
#else
#pragma unroll
        for (int j = 0; j < 8; ++j) { s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f; }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int jp = 0; jp < 4; ++jp) {
                const int key = 16 * jp + 8 * (mi >> 1) + ri;
                const int ch  = 2 * kk + (mi & 1);
                unsigned kb[4];
                pf_ldsm4(kb, K + key * 64 + ((ch ^ (key & 7)) * 8));
                const unsigned b0[2] = { kb[0], kb[1] };
                const unsigned b1[2] = { kb[2], kb[3] };
                pf_mma_f16(s[2 * jp],     qa[kk], b0);
                pf_mma_f16(s[2 * jp + 1], qa[kk], b1);
            }
        }
#endif
        const int kbase = kt * 64;
        const bool diag = (kbase + 63 > P + q0);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float v = s[j][e] * L2E;
                if (diag) {
                    const int kpos = kbase + 8 * j + 2 * t4 + (e & 1);
                    const int qp = (e < 2) ? qpos0 : qpos1;
                    if (kpos > qp) v = -INFINITY;
                }
                s[j][e] = v;
            }
        }
        float mx0 = mrow0, mx1 = mrow1;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            mx0 = fmaxf(mx0, fmaxf(s[j][0], s[j][1]));
            mx1 = fmaxf(mx1, fmaxf(s[j][2], s[j][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float al0 = pf_ex2(mrow0 - mx0), al1 = pf_ex2(mrow1 - mx1);
        mrow0 = mx0; mrow1 = mx1;
        lrow0 *= al0; lrow1 *= al1;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s[j][0] = pf_ex2(s[j][0] - mx0); s[j][1] = pf_ex2(s[j][1] - mx0);
            s[j][2] = pf_ex2(s[j][2] - mx1); s[j][3] = pf_ex2(s[j][3] - mx1);
            lrow0 += s[j][0] + s[j][1];
            lrow1 += s[j][2] + s[j][3];
            acc[j][0] *= al0; acc[j][1] *= al0; acc[j][2] *= al1; acc[j][3] *= al1;
        }
#if PF_ATTN_PV_F16
        // This tile's P.V in fp16 accumulators (P <= 1 against the running
        // max, |V| is O(1) in this model, 64 terms), then into the fp32 output.
        unsigned pv[8][2];
#pragma unroll
        for (int j = 0; j < 8; ++j) { pv[j][0] = 0u; pv[j][1] = 0u; }
#endif
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const unsigned pa[4] = { pf_pack_h2(s[2 * kk][0], s[2 * kk][1]),
                                     pf_pack_h2(s[2 * kk][2], s[2 * kk][3]),
                                     pf_pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                     pf_pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3]) };
#pragma unroll
            for (int jdp = 0; jdp < 4; ++jdp) {
                const int key = 16 * kk + 8 * (mi & 1) + ri;
                const int ch  = 2 * jdp + (mi >> 1);
                unsigned vb[4];
                pf_ldsm4t(vb, V + key * 64 + ((ch ^ (key & 7)) * 8));
                const unsigned b0[2] = { vb[0], vb[1] };
                const unsigned b1[2] = { vb[2], vb[3] };
#if PF_ATTN_PV_F16
                pf_mma_f16h(pv[2 * jdp],     pa, b0);
                pf_mma_f16h(pv[2 * jdp + 1], pa, b1);
#else
                pf_mma_f16(acc[2 * jdp],     pa, b0);
                pf_mma_f16(acc[2 * jdp + 1], pa, b1);
#endif
            }
        }
#if PF_ATTN_PV_F16
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&pv[j][0]));
            const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&pv[j][1]));
            acc[j][0] += lo.x; acc[j][1] += lo.y; acc[j][2] += hi.x; acc[j][3] += hi.y;
        }
#endif
        __syncthreads();
    }
    pf_wait<0>();
    lrow0 += __shfl_xor_sync(0xffffffffu, lrow0, 1);
    lrow0 += __shfl_xor_sync(0xffffffffu, lrow0, 2);
    lrow1 += __shfl_xor_sync(0xffffffffu, lrow1, 1);
    lrow1 += __shfl_xor_sync(0xffffffffu, lrow1, 2);
    const float il0 = 1.0f / lrow0, il1 = 1.0f / lrow1;
    const int r0 = q0 + g, r1 = r0 + 8;
#pragma unroll
    for (int jd = 0; jd < 8; ++jd) {
        const int c = head * 64 + 8 * jd + 2 * t4;
        if (r0 < m) *reinterpret_cast<float2 *>(o + (size_t) r0 * 2048 + c) = make_float2(acc[jd][0] * il0, acc[jd][1] * il0);
        if (r1 < m) *reinterpret_cast<float2 *>(o + (size_t) r1 * 2048 + c) = make_float2(acc[jd][2] * il1, acc[jd][3] * il1);
    }
}

// ---------------------------------------------------------------------------
// INT8 attention scores (the SageAttention recipe).
//
// The fp32-accumulate fp16 mma that scores Q.K runs at 40 TMAC/s on GA102; the
// int8 mma runs at 158.7.  So Q and K are quantised to int8, one scale per
// (token, head) row of 64, and the score is sumi * sq * sk, exact in int32.
// Accumulating the scores in fp16 instead (PF_ATTN_QK_F16) was 1.18x faster but
// cost corpus NLL; int8 is both faster and, per element, a cleaner error
// (uniform, unbiased, one rounding per input instead of a rounding per partial
// sum).
//
// K is SMOOTHED first: a fixed per-(layer, kv head) vector c is subtracted from
// every key.  q.(k - c) = q.k - q.c, and q.c is the same for every key a query
// sees, so softmax is unchanged EXACTLY; but the int8 grid now spans k's spread
// around c instead of its offset, which is where RMS-normed keys put most of
// their magnitude.  c is the mean key of the first prefill chunk (k_pf_kmean)
// and stays fixed for the rest of the context.  (Measured on this model's
// real keys it did not help -- RMS-normed, RoPE'd keys carry little common
// offset -- so the engine leaves it off; the machinery stays for A/B.)
//
// P.V is unchanged (fp16 P, fp16 V, PF_ATTN_PV_F16), and V and the fp16 K cache
// the decode path reads are untouched: the int8 K lives in a prefill-only copy.
// ---------------------------------------------------------------------------

// grid (ceil(rows*heads/8)) x 256: one warp per 64-element row.  Row r of src
// (pitch 64 halves) -> dst row r (64 bytes), scale[r] = mul * amax/127.  With
// cm != null, the row's head (r % heads) mean cm[head*64..] is subtracted first.
__device__ __forceinline__ void pf_q8_row(const float v0, const float v1, int8_t * dst, float * scl,
                                          float mul, int lane) {
    float am = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float d = am / 127.0f;
    const float id = (am > 0.0f) ? 127.0f / am : 0.0f;
    char2 q;
    q.x = (signed char) __float2int_rn(v0 * id);
    q.y = (signed char) __float2int_rn(v1 * id);
    reinterpret_cast<char2 *>(dst)[lane] = q;
    if (lane == 0) *scl = d * mul;
}
// Q: qh [m][2048] fp16 (already * 1/8) -> q8 [m][2048], qsc [m][32] = d * log2(e),
// and qcs [m][32] = q.c * log2(e), the shift smoothing puts on every int8 score,
// which the fp16-scored tiles subtract to stay on the same softmax.
__global__ void __launch_bounds__(256) k_pf_q8(const __half * __restrict__ qh, int m,
                                               const float * __restrict__ cm,
                                               int8_t * __restrict__ q8, float * __restrict__ qsc,
                                               float * __restrict__ qcs) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= m * LK_N_HEADS) return;
    const float2 v = __half22float2(reinterpret_cast<const __half2 *>(qh + (size_t) r * 64)[lane]);
    pf_q8_row(v.x, v.y, q8 + (size_t) r * 64, qsc + r, 1.4426950408889634f, lane);
    float qc = 0.0f;
    if (cm) {
        const float * c = cm + ((r % LK_N_HEADS) / 4) * 64;
        qc = v.x * c[2 * lane] + v.y * c[2 * lane + 1];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) qc += __shfl_xor_sync(0xffffffffu, qc, o);
    }
    if (lane == 0) qcs[r] = qc * 1.4426950408889634f;
}
// K: positions [p0, p1) of one layer's fp16 cache kc [pos][8][64] -> k8 (same
// layout, int8) and ksc [8][kstride] (head-major so a key tile's scales are one
// contiguous run).  cm = [8][64] smoothing vector, or null.
__global__ void __launch_bounds__(256) k_pf_k8(const __half * __restrict__ kc, int p0, int p1,
                                               const float * __restrict__ cm, int kstride,
                                               int8_t * __restrict__ k8, float * __restrict__ ksc) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int pos = p0 + r / LK_N_KV_HEADS, h = r % LK_N_KV_HEADS;
    if (pos >= p1) return;
    const size_t row = (size_t) pos * LK_N_KV_HEADS + h;
    float2 v = __half22float2(reinterpret_cast<const __half2 *>(kc + row * 64)[lane]);
    if (cm) { v.x -= cm[h * 64 + 2 * lane]; v.y -= cm[h * 64 + 2 * lane + 1]; }
    pf_q8_row(v.x, v.y, k8 + row * 64, ksc + (size_t) h * kstride + pos, 1.0f, lane);
}
// grid 8 (kv heads) x 1024: cm[h][d] = mean over positions [0, n) of kc.  Fixed
// summation order (16 strided partials, then a fixed tree), so deterministic.
__global__ void __launch_bounds__(1024) k_pf_kmean(const __half * __restrict__ kc, int n,
                                                   float * __restrict__ cm) {
    __shared__ float part[16][64];
    const int h = blockIdx.x, d = threadIdx.x & 63, g = threadIdx.x >> 6;
    float s = 0.0f;
    for (int p = g; p < n; p += 16) s += __half2float(kc[((size_t) p * LK_N_KV_HEADS + h) * 64 + d]);
    part[g][d] = s;
    __syncthreads();
    if (g == 0) {
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) t += part[i][d];
        cm[h * 64 + d] = t / (float) n;
    }
}

__device__ __forceinline__ void pf_mma_s8acc(int (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Same tiling, causal masking, online softmax and P.V as k_pf_attn; only the
// score product differs.  K tile rows are 64 bytes = 4 16-byte chunks, chunk c
// of key r stored at c ^ ((r >> 1) & 3) so each ldmatrix phase (8 consecutive
// keys, one logical chunk) hits 8 distinct 16-byte bank groups.
//
// Tiles where attention is sharpest are scored EXACTLY as k_pf_attn does (fp16
// Q.K, fp32 accumulation): every key tile within `near` positions of the
// block's first query, and, with `sink`, key tile 0 (the attention sink).  At
// long context these are a handful of tiles out of thousands, so they cost
// nothing, and they are where an int8 score error moves the most probability.
__global__ void __launch_bounds__(256, 2) k_pf_attn8(const int8_t * __restrict__ q8,
                                                     const float * __restrict__ qsc,
                                                     const float * __restrict__ qcs,
                                                     const __half * __restrict__ qh, int m, int P,
                                                     const int8_t * __restrict__ k8,
                                                     const float * __restrict__ ksc, int kstride,
                                                     const __half * __restrict__ kc,
                                                     const __half * __restrict__ vc,
                                                     int near, int sink,
                                                     float * __restrict__ o) {
    __shared__ __align__(128) int8_t sK[2][64 * 64];
    __shared__ __align__(128) __half sKh[2][64 * 64];
    __shared__ __align__(128) __half sV[2][64 * 64];
    __shared__ __align__(16)  float  sS[2][64];
    const int qt   = gridDim.x - 1 - blockIdx.x;
    const int kvh  = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t4 = lane & 3;
    const int head = kvh * 4 + (warp & 3);
    const int q0 = qt * 32 + (warp >> 2) * 16;
    const int qlast = (qt * 32 + 32 < m) ? (qt * 32 + 32) : m;
    const int kend = P + qlast;
    const int nkt = (kend + 63) / 64;
    const float * ks_h = ksc + (size_t) kvh * kstride;

    unsigned qa[2][4];
    float rq0, rq1;
    {
        const int r0 = q0 + g, r1 = q0 + g + 8;
        const int8_t * Q0 = q8 + (size_t) r0 * 2048 + head * 64;
        const int8_t * Q1 = q8 + (size_t) r1 * 2048 + head * 64;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const int c = ks * 32 + 4 * t4;
            qa[ks][0] = (r0 < m) ? *reinterpret_cast<const unsigned *>(Q0 + c)      : 0u;
            qa[ks][1] = (r1 < m) ? *reinterpret_cast<const unsigned *>(Q1 + c)      : 0u;
            qa[ks][2] = (r0 < m) ? *reinterpret_cast<const unsigned *>(Q0 + c + 16) : 0u;
            qa[ks][3] = (r1 < m) ? *reinterpret_cast<const unsigned *>(Q1 + c + 16) : 0u;
        }
        rq0 = (r0 < m) ? qsc[(size_t) r0 * 32 + head] : 0.0f;
        rq1 = (r1 < m) ? qsc[(size_t) r1 * 32 + head] : 0.0f;
    }
    const float qc0 = (q0 + g < m) ? qcs[(size_t) (q0 + g) * 32 + head] : 0.0f;
    const float qc1 = (q0 + g + 8 < m) ? qcs[(size_t) (q0 + g + 8) * 32 + head] : 0.0f;
    const int near_lo = P + qt * 32 - near;          // block-uniform
    auto exact = [&](int kt) { return (kt * 64 + 64 > near_lo) || (sink && kt == 0); };
    auto load = [&](int st, int kt) {
        if (exact(kt)) {
            for (int i = threadIdx.x; i < 512; i += 256) {
                const int r = i >> 3, c = i & 7;
                const int kpos = kt * 64 + r;
                const bool ok = kpos < kend;
                pf_cp16(sKh[st] + r * 64 + ((c ^ (r & 7)) * 8),
                        kc + ((size_t) (ok ? kpos : 0) * LK_N_KV_HEADS + kvh) * 64 + c * 8, ok);
            }
        }
        for (int i = threadIdx.x + (exact(kt) ? 256 : 0); i < 256 + 512 + 16; i += 256) {
            if (i < 256) {
                const int r = i >> 2, c = i & 3;
                const int kpos = kt * 64 + r;
                const bool ok = kpos < kend;
                pf_cp16(sK[st] + r * 64 + ((c ^ ((r >> 1) & 3)) * 16),
                        k8 + ((size_t) (ok ? kpos : 0) * LK_N_KV_HEADS + kvh) * 64 + c * 16, ok);
            } else if (i < 768) {
                const int j = i - 256;
                const int r = j >> 3, c = j & 7;
                const int kpos = kt * 64 + r;
                const bool ok = kpos < kend;
                pf_cp16(sV[st] + r * 64 + ((c ^ (r & 7)) * 8),
                        vc + ((size_t) (ok ? kpos : 0) * LK_N_KV_HEADS + kvh) * 64 + c * 8, ok);
            } else {
                const int c = i - 768;
                const int kpos = kt * 64 + c * 4;
                const bool ok = kpos < kend;
                pf_cp16(sS[st] + c * 4, ks_h + (ok ? kpos : 0), ok);
            }
        }
    };

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f; }
    float mrow0 = -INFINITY, mrow1 = -INFINITY, lrow0 = 0.0f, lrow1 = 0.0f;
    const int qpos0 = P + q0 + g, qpos1 = qpos0 + 8;

    load(0, 0);
    pf_commit();
    for (int kt = 0; kt < nkt; ++kt) {
        if (kt + 1 < nkt) load((kt + 1) & 1, kt + 1);
        pf_commit();
        pf_wait<1>();
        __syncthreads();
        const int8_t * K = sK[kt & 1];
        const __half * V = sV[kt & 1];
        const float *  S = sS[kt & 1];
        const int mi = lane >> 3, ri = lane & 7;

        float s[8][4];
        if (exact(kt)) {
            // k_pf_attn's score, minus the smoothing shift q.c.
            const __half * Kh = sKh[kt & 1];
            const int r0 = q0 + g, r1 = q0 + g + 8;
            const __half * Q0 = qh + (size_t) r0 * 2048 + head * 64;
            const __half * Q1 = qh + (size_t) r1 * 2048 + head * 64;
#pragma unroll
            for (int j = 0; j < 8; ++j) { s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f; }
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const int c = kk * 16 + 2 * t4;
                const unsigned qf[4] = {
                    (r0 < m) ? *reinterpret_cast<const unsigned *>(Q0 + c)     : 0u,
                    (r1 < m) ? *reinterpret_cast<const unsigned *>(Q1 + c)     : 0u,
                    (r0 < m) ? *reinterpret_cast<const unsigned *>(Q0 + c + 8) : 0u,
                    (r1 < m) ? *reinterpret_cast<const unsigned *>(Q1 + c + 8) : 0u };
#pragma unroll
                for (int jp = 0; jp < 4; ++jp) {
                    const int key = 16 * jp + 8 * (mi >> 1) + ri;
                    const int ch  = 2 * kk + (mi & 1);
                    unsigned kb[4];
                    pf_ldsm4(kb, Kh + key * 64 + ((ch ^ (key & 7)) * 8));
                    const unsigned b0[2] = { kb[0], kb[1] };
                    const unsigned b1[2] = { kb[2], kb[3] };
                    pf_mma_f16(s[2 * jp],     qf, b0);
                    pf_mma_f16(s[2 * jp + 1], qf, b1);
                }
            }
            const float L2E = 1.4426950408889634f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                s[j][0] = fmaf(s[j][0], L2E, -qc0); s[j][1] = fmaf(s[j][1], L2E, -qc0);
                s[j][2] = fmaf(s[j][2], L2E, -qc1); s[j][3] = fmaf(s[j][3], L2E, -qc1);
            }
        } else {
            int sd[8][4];
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
                for (int jp = 0; jp < 4; ++jp) {
                    const int key = 16 * jp + 8 * (mi >> 1) + ri;
                    const int ch  = 2 * ks + (mi & 1);
                    unsigned kb[4];
                    pf_ldsm4(kb, K + key * 64 + ((ch ^ ((key >> 1) & 3)) * 16));
                    const unsigned b0[2] = { kb[0], kb[1] };
                    const unsigned b1[2] = { kb[2], kb[3] };
                    if (ks == 0) { pf_mma_s8(sd[2 * jp], qa[0], b0);    pf_mma_s8(sd[2 * jp + 1], qa[0], b1); }
                    else         { pf_mma_s8acc(sd[2 * jp], qa[1], b0); pf_mma_s8acc(sd[2 * jp + 1], qa[1], b1); }
                }
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float2 sk = *reinterpret_cast<const float2 *>(S + 8 * j + 2 * t4);
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    s[j][e] = (__int_as_float(sd[j][e]) - PF_MAGIC_F) * ((e & 1) ? sk.y : sk.x) * ((e < 2) ? rq0 : rq1);
            }
        }
        const int kbase = kt * 64;
        const bool diag = (kbase + 63 > P + q0);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float v = s[j][e];
                if (diag) {
                    const int kpos = kbase + 8 * j + 2 * t4 + (e & 1);
                    const int qp = (e < 2) ? qpos0 : qpos1;
                    if (kpos > qp) v = -INFINITY;
                }
                s[j][e] = v;
            }
        }
        float mx0 = mrow0, mx1 = mrow1;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            mx0 = fmaxf(mx0, fmaxf(s[j][0], s[j][1]));
            mx1 = fmaxf(mx1, fmaxf(s[j][2], s[j][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float al0 = pf_ex2(mrow0 - mx0), al1 = pf_ex2(mrow1 - mx1);
        mrow0 = mx0; mrow1 = mx1;
        lrow0 *= al0; lrow1 *= al1;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s[j][0] = pf_ex2(s[j][0] - mx0); s[j][1] = pf_ex2(s[j][1] - mx0);
            s[j][2] = pf_ex2(s[j][2] - mx1); s[j][3] = pf_ex2(s[j][3] - mx1);
            lrow0 += s[j][0] + s[j][1];
            lrow1 += s[j][2] + s[j][3];
            acc[j][0] *= al0; acc[j][1] *= al0; acc[j][2] *= al1; acc[j][3] *= al1;
        }
        unsigned pv[8][2];
#pragma unroll
        for (int j = 0; j < 8; ++j) { pv[j][0] = 0u; pv[j][1] = 0u; }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const unsigned pa[4] = { pf_pack_h2(s[2 * kk][0], s[2 * kk][1]),
                                     pf_pack_h2(s[2 * kk][2], s[2 * kk][3]),
                                     pf_pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                     pf_pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3]) };
#pragma unroll
            for (int jdp = 0; jdp < 4; ++jdp) {
                const int key = 16 * kk + 8 * (mi & 1) + ri;
                const int ch  = 2 * jdp + (mi >> 1);
                unsigned vb[4];
                pf_ldsm4t(vb, V + key * 64 + ((ch ^ (key & 7)) * 8));
                const unsigned b0[2] = { vb[0], vb[1] };
                const unsigned b1[2] = { vb[2], vb[3] };
                pf_mma_f16h(pv[2 * jdp],     pa, b0);
                pf_mma_f16h(pv[2 * jdp + 1], pa, b1);
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&pv[j][0]));
            const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&pv[j][1]));
            acc[j][0] += lo.x; acc[j][1] += lo.y; acc[j][2] += hi.x; acc[j][3] += hi.y;
        }
        __syncthreads();
    }
    pf_wait<0>();
    lrow0 += __shfl_xor_sync(0xffffffffu, lrow0, 1);
    lrow0 += __shfl_xor_sync(0xffffffffu, lrow0, 2);
    lrow1 += __shfl_xor_sync(0xffffffffu, lrow1, 1);
    lrow1 += __shfl_xor_sync(0xffffffffu, lrow1, 2);
    const float il0 = 1.0f / lrow0, il1 = 1.0f / lrow1;
    const int r0 = q0 + g, r1 = r0 + 8;
#pragma unroll
    for (int jd = 0; jd < 8; ++jd) {
        const int c = head * 64 + 8 * jd + 2 * t4;
        if (r0 < m) *reinterpret_cast<float2 *>(o + (size_t) r0 * 2048 + c) = make_float2(acc[jd][0] * il0, acc[jd][1] * il0);
        if (r1 < m) *reinterpret_cast<float2 *>(o + (size_t) r1 * 2048 + c) = make_float2(acc[jd][2] * il1, acc[jd][3] * il1);
    }
}

// grid (rows, kb/8) x 256: one warp per (row, block).
__global__ void __launch_bounds__(256) k_pf_quant_rows(const float * __restrict__ X, int ld, int kb, PfAct a) {
    const int r = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.y * 8 + warp;
    if (b >= kb) return;                               // warp-uniform
    const float v = X[(size_t) r * ld + b * 32 + lane];
    pf_quant_warp(v, a.qs + (size_t) r * kb * 32 + b * 32, a.sc + (size_t) r * kb + b, lane);
}

// grid (rows, ceil(kb/8)) x 256: k_swiglu's v = silu(gate) * up, quantised.
__global__ void __launch_bounds__(256) k_pf_swiglu(const float * __restrict__ Y, int ld, int up_off,
                                                   int kb, PfAct a) {
    const int r = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.y * 8 + warp;
    if (b >= kb) return;                               // warp-uniform
    const int i = b * 32 + lane;
    const float * y = Y + (size_t) r * ld;
    const float v = silu_f(y[i]) * y[up_off + i];
    pf_quant_warp(v, a.qs + (size_t) r * kb * 32 + b * 32, a.sc + (size_t) r * kb + b, lane);
}

// Router logits for 64 tokens x 64 experts per block, then the sigmoid pair
// that k_router_top4 derives.  Register-blocked: each thread owns 4 tokens x 4
// experts, so every k costs two LDS.128 for sixteen FMA.  (The first version
// gave each thread 8 tokens of one expert and was LDS-bound at 2.6 TFLOP/s; a
// 32-token block with 2x4 per thread launches twice the blocks and measured
// SLOWER, 6.6 vs 5.0 ms per 4096-token chunk, because it streams W twice.)
// The dot runs in a different order from k_router_dot (it is a tiled fp32
// GEMM), so the logits agree to reassociation; the selection is the decode
// path's own (k_pf_select).
#define PF_RT_K 32
__global__ void __launch_bounds__(256) k_pf_router(const float * __restrict__ H, int m,
                                                   const float * __restrict__ Wr,
                                                   const float * __restrict__ bias,
                                                   const float * __restrict__ inv,
                                                   float * __restrict__ sel, float * __restrict__ sig) {
    __shared__ __align__(16) float sH[PF_RT_K][68];   // [k][token]
    __shared__ __align__(16) float sW[PF_RT_K][68];   // [k][expert]
    const int tid = threadIdx.x;
    const int te = tid & 15, tt = tid >> 4;           // experts 4te.., tokens 4tt..
    const int t0 = blockIdx.x * 64;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < LK_HIDDEN; k0 += PF_RT_K) {
#pragma unroll
        for (int q = 0; q < (64 * PF_RT_K) / 256; ++q) {
            const int i = tid + q * 256;
            const int r = i / PF_RT_K, c = i % PF_RT_K;   // row = token / expert, c = k
            const int tk = t0 + r;
            sH[c][r] = (tk < m) ? H[(size_t) tk * LK_HIDDEN + k0 + c] : 0.0f;
            sW[c][r] = Wr[(size_t) r * LK_HIDDEN + k0 + c];
        }
        __syncthreads();
#pragma unroll 8
        for (int c = 0; c < PF_RT_K; ++c) {
            const float4 h = *reinterpret_cast<const float4 *>(&sH[c][tt * 4]);
            const float4 w = *reinterpret_cast<const float4 *>(&sW[c][te * 4]);
            const float hv[4] = { h.x, h.y, h.z, h.w }, wv[4] = { w.x, w.y, w.z, w.w };
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(hv[i], wv[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int tk = t0 + tt * 4 + i;
        if (tk >= m) continue;
        float sg[4], sl[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int e = te * 4 + j;
            const float l = acc[i][j] * inv[tk];
            const float gv = 1.0f / (1.0f + expf(-l));
            sg[j] = gv;
            sl[j] = gv + bias[e];
        }
        *reinterpret_cast<float4 *>(sig + (size_t) tk * 64 + te * 4) = make_float4(sg[0], sg[1], sg[2], sg[3]);
        *reinterpret_cast<float4 *>(sel + (size_t) tk * 64 + te * 4) = make_float4(sl[0], sl[1], sl[2], sl[3]);
    }
}

// Short prompts: one warp per (expert, token), k_router_dot's arithmetic
// verbatim -- the same lane-strided chain, the same butterfly, the same
// multiply by inv -- so for a given input row the logits are BIT-IDENTICAL to
// the decode router.  Each warp re-reads its expert's 8 KB row from L2, which
// is nothing at prefill sizes below ~256 tokens and is why the tiled kernel
// above takes over beyond that: there a single 64-token block is the whole
// grid, and the router alone cost 4.6 ms per 64-token prompt.
__global__ void __launch_bounds__(32) k_pf_router_small(const float * __restrict__ H,
                                                        const float * __restrict__ Wr,
                                                        const float * __restrict__ bias,
                                                        const float * __restrict__ inv,
                                                        float * __restrict__ sel, float * __restrict__ sig) {
    const int e = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const float * h = H + (size_t) t * LK_HIDDEN;
    const float * row = Wr + (size_t) e * LK_HIDDEN;
    float acc = 0.0f;
    for (int base = 0; base < LK_HIDDEN; base += 32 * 16) {
        float v[16], hv[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) v[j]  = row[base + j * 32 + lane];
#pragma unroll
        for (int j = 0; j < 16; ++j) hv[j] = h[base + j * 32 + lane];
#pragma unroll
        for (int j = 0; j < 16; ++j) acc += v[j] * hv[j];
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        const float l = acc * inv[t];
        const float gv = 1.0f / (1.0f + expf(-l));
        sig[(size_t) t * 64 + e] = gv;
        sel[(size_t) t * 64 + e] = gv + bias[e];
    }
}

// One warp per token: lk_router_select, the exact top-4 the decode path runs.
__global__ void k_pf_select(const float * __restrict__ sel, const float * __restrict__ sig, int m,
                            int * __restrict__ idx, float * __restrict__ w) {
    const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (t >= m) return;                                // warp-uniform
    int ord[LK_TOP_K];
    float wv[LK_TOP_K];
    lk_router_select(sel + (size_t) t * 64, sig + (size_t) t * 64, lane, ord, wv);
    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < LK_TOP_K; ++k) { idx[t * 4 + k] = ord[k]; w[t * 4 + k] = wv[k]; }
    }
}

__global__ void k_pf_moe_count(const int * __restrict__ idx, int n, int * __restrict__ cnt) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < n) atomicAdd(&cnt[idx[s]], 1);
}

// One thread: 64 experts is nothing to scan.
__global__ void k_pf_moe_plan(const int * __restrict__ cnt, int * __restrict__ off,
                              int4 * __restrict__ tiles, int * __restrict__ ntiles) {
    if (threadIdx.x != 0) return;
    int o = 0, nt = 0;
    for (int e = 0; e < LK_N_EXPERTS; ++e) {
        const int c = cnt[e];
        off[e] = o;
        for (int s = 0; s < c; s += PF_MOE_BN) {
            tiles[nt++] = make_int4(e, o + s, (c - s < PF_MOE_BN) ? (c - s) : PF_MOE_BN, 0);
        }
        o += c;
    }
    *ntiles = nt;
}

// grid 64 x 1024: block e places expert e's slots in increasing slot order, so
// the permutation (and therefore every GEMM result) is deterministic.
__global__ void __launch_bounds__(1024) k_pf_moe_scatter(const int * __restrict__ idx, int n,
                                                         const int * __restrict__ off,
                                                         int * __restrict__ perm) {
    const int e = blockIdx.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    __shared__ int s_w[32];
    __shared__ int s_base;
    if (tid == 0) s_base = off[e];
    __syncthreads();
    for (int s0 = 0; s0 < n; s0 += 1024) {
        const int s = s0 + tid;
        const bool f = (s < n) && (idx[s] == e);
        const unsigned bal = __ballot_sync(0xffffffffu, f);
        if (lane == 0) s_w[warp] = __popc(bal);
        __syncthreads();
        int before = 0;
        for (int i = 0; i < warp; ++i) before += s_w[i];
        const int base = s_base;
        if (f) perm[base + before + __popc(bal & ((1u << lane) - 1u))] = s;
        __syncthreads();
        if (tid == 0) { int tot = 0; for (int i = 0; i < 32; ++i) tot += s_w[i]; s_base = base + tot; }
        __syncthreads();
    }
}

// grid m x 256: k_pf_moe_combine followed by k_pf_norm's Phase 4y path for the
// NEXT layer's attn_norm, in one pass over the token's row.  Thread tid owns
// elements tid + 256j, which is exactly the element k_pf_norm's warp (tid>>5)
// handles as lane (tid&31) of block (tid>>5) + 8j -- so the arithmetic is the
// two kernels' own, unchanged, and the residual row is read once instead of
// three times.
__global__ void __launch_bounds__(256) k_pf_moe_combine_norm(float * __restrict__ X, const float * __restrict__ D,
                                                             const float * __restrict__ w,
                                                             const float * __restrict__ nw, PfAct a,
                                                             float * __restrict__ inv_out) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    __shared__ float s_sq[64];
    float wk[LK_TOP_K];
#pragma unroll
    for (int k = 0; k < LK_TOP_K; ++k) wk[k] = w[t * 4 + k];
    float * x = X + (size_t) t * LK_HIDDEN;
    int8_t * q = a.qs + (size_t) t * LK_HIDDEN;
    float2 * sc = a.sc + (size_t) t * (LK_HIDDEN / 32);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int i = tid + 256 * j;
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < LK_TOP_K; ++k) acc += wk[k] * D[(size_t) (t * 4 + k) * LK_HIDDEN + i];
        const float xv = x[i] + acc;
        x[i] = xv;
        const int b = warp + 8 * j;
        const float uv = xv * nw[i];
        pf_quant_warp(uv, q + b * 32, sc + b, lane);
        float sq = xv * xv;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (lane == 0) s_sq[b] = sq;
    }
    __syncthreads();
    if (warp == 0) {
        float sv = s_sq[lane] + s_sq[lane + 32];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sv += __shfl_xor_sync(0xffffffffu, sv, o);
        if (lane == 0) inv_out[t] = rsqrtf(sv * (1.0f / (float) LK_HIDDEN) + LK_EPS);
    }
}

// grid (m, 8) x 256: k_moe_down_accum's combine.  D is [m*4][2048], one row per
// slot; the four experts are added in rank order from a = 0, as decode does.
__global__ void k_pf_moe_combine(float * __restrict__ X, const float * __restrict__ D,
                                 const float * __restrict__ w) {
    const int t = blockIdx.x;
    const int n = blockIdx.y * blockDim.x + threadIdx.x;
    float a = 0.0f;
#pragma unroll
    for (int k = 0; k < LK_TOP_K; ++k)
        a += w[t * 4 + k] * D[(size_t) (t * 4 + k) * LK_HIDDEN + n];
    float * xp = X + (size_t) t * LK_HIDDEN + n;
    *xp = *xp + a;
}

// One thread per block: planar row t back into block_q8_1, so a decode GEMV can
// consume a batched activation.  Both fields are fp16 values already, so the
// re-pack is exact (sc.y / 8 undoes an exact *8).
__global__ void k_pf_pack_q8_1(PfAct a, int t, block_q8_1 * __restrict__ y) {
    const int b = threadIdx.x;
    const float2 s = a.sc[(size_t) t * 64 + b];
    y[b].ds = __floats2half2_rn(s.x, s.y * 0.125f);
    const int4 * src = reinterpret_cast<const int4 *>(a.qs + (size_t) t * LK_HIDDEN + b * 32);
    const int4 v0 = src[0], v1 = src[1];
    int * dq = reinterpret_cast<int *>(y[b].qs);
    dq[0] = v0.x; dq[1] = v0.y; dq[2] = v0.z; dq[3] = v0.w;
    dq[4] = v1.x; dq[5] = v1.y; dq[6] = v1.z; dq[7] = v1.w;
}
