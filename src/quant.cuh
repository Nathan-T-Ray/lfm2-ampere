// quant.cuh - block formats, Q8_1 activation quantisation, and integer dp4a
// dot products, numerically matching llama.cpp's ggml-cuda backend.
//
// ALL weight data is read from the PLANAR lfm25.pack produced by tools/repack.py:
//   Q4_0 : qs plane (16 B/block) + d plane (2 B/block)
//   Q6_K : ql (128 B) + qh (64 B) + scales (16 B) + d (2 B) planes
// Planar splitting is what makes every access 16-byte aligned; in the native
// interleaved layout block_q6_K is 210 B, so only block 0 of each 2048-wide row
// is 16-byte aligned.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK4_0 32
#define QK8_1 32
#define QK_K  256

// ---------------------------------------------------------------------------
// activation block, identical to ggml's block_q8_1
//   ds.x = d   = amax/127
//   ds.y = SUM = the RAW sum of the 32 activations (NOT d*sum).  This is the
//          ggml-cuda convention and dot_q4_0_q8_1's correction term uses it as
//          a proxy for d8*sum(q8); replicating it exactly keeps us bit-aligned
//          with the oracle.
// ---------------------------------------------------------------------------
struct block_q8_1 {
    __half2  ds;
    int8_t   qs[32];
};
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 must be 36 bytes");

// ---------------------------------------------------------------------------
// FP16 RANGE AUDIT (ARCHITECTURE.md 9.1 / REBRIEF A6).
//
// Phase 4y carries a scale invariance through this struct: with
// ds.x = d = amax/127 and ds.y = SUM = the raw 32-element sum, both fields are
// LINEAR in a positive scalar applied to the activations, and `qs` is invariant
// -- so dot(inv*u) = inv*dot(u) and the RMSNorm's 1/rms can be applied once per
// output row instead of once per element.
//
// That argument is exact in fp32.  The FIELDS ARE fp16, so it holds only while
// both stay in range:
//
//     d   underflows to subnormal below 6.104e-5 and overflows to inf above 65504
//     SUM overflows to inf above 65504
//
// Subnormal d loses mantissa bits; SUM at 65504 with 32 addends of order 1e3 is
// reachable in principle.  Neither is visible in a benchmark or a loss curve --
// they would show up as a slowly degrading accuracy that looks like a precision
// limit.  So it is MEASURED over the real corpus rather than argued.
//
// For non-negative floats the IEEE bit pattern orders the same way the
// corresponding unsigned integer does, so atomicMin/Max on the reinterpreted
// bits is a valid float min/max.  The hook is a hook rather than an inline body
// so the counter lives in exactly one translation unit.
// ---------------------------------------------------------------------------
#if LK_NORM_AUDIT
__device__ void lk_audit_hook(float d, float s);
#define LK_AUDIT_D(d, s) lk_audit_hook((d), (s))
#else
#define LK_AUDIT_D(d, s) ((void) 0)
#endif

// __dp4a is overloaded: (int,int,int) does SIGNED int8 x int8, whereas
// (unsigned,unsigned,unsigned) does unsigned u8 x u8.  Our Q8_1 activations are
// int8 and go negative, so the SIGNED overload is mandatory -- the unsigned one
// silently reinterprets every negative value as 255.  This mirrors ggml's
// ggml_cuda_dp4a, which is also (int,int,int).
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __dp4a(a, b, c);
}

// ---------------------------------------------------------------------------
// Warp-cooperative Q8_1 quantisation: one warp per 32-element block, one
// element per lane.  Mirrors ggml-cuda's quantize_q8_1 exactly.
// ---------------------------------------------------------------------------
// Interleaved form for two independent 32-element blocks: the same per-block
// operations in the same order, but the two reduction chains sit adjacent so
// their latencies overlap instead of serialising.  Each block still performs
// the same five shfl_xor steps in the same sequence, so every value produced
// is bit-identical to calling the single-block form twice.
__device__ __forceinline__ void quantize_q8_1_warp2(const float x0, const float x1,
                                                    block_q8_1 * __restrict__ y0,
                                                    block_q8_1 * __restrict__ y1, int lane) {
    const unsigned a0 = __reduce_max_sync(0xffffffffu, __float_as_uint(fabsf(x0)));
    const unsigned a1 = __reduce_max_sync(0xffffffffu, __float_as_uint(fabsf(x1)));
    float s0 = x0, s1 = x1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, o);
        s1 += __shfl_xor_sync(0xffffffffu, s1, o);
    }
    const float d0 = __uint_as_float(a0) / 127.0f;
    const float d1 = __uint_as_float(a1) / 127.0f;
    y0->qs[lane] = (a0 == 0u) ? (int8_t) 0 : (int8_t) roundf(x0 / d0);
    y1->qs[lane] = (a1 == 0u) ? (int8_t) 0 : (int8_t) roundf(x1 / d1);
    if (lane == 0) {
        y0->ds = __floats2half2_rn(d0, s0);
        y1->ds = __floats2half2_rn(d1, s1);
        LK_AUDIT_D(d0, s0);
        LK_AUDIT_D(d1, s1);
    }
}

__device__ __forceinline__ void quantize_q8_1_warp(const float xi, block_q8_1 * y, int lane) {
    // A maximum is exact and independent of reduction order, so the hardware
    // reduction gives a bit-identical answer.  For non-negative floats the IEEE
    // bit pattern orders the same way unsigned integers do, so comparing the
    // reinterpreted bits is the same comparison.  This shortens the dependency
    // chain from ten sequential shuffles to six, which matters here because a
    // single block runs 8 warps with nothing to hide the latency behind.
    const unsigned am = __reduce_max_sync(0xffffffffu, __float_as_uint(fabsf(xi)));
    float sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = __uint_as_float(am) / 127.0f;
    const int8_t q = (am == 0u) ? (int8_t)0 : (int8_t)roundf(xi / d);
    y->qs[lane] = q;
    if (lane == 0) { y->ds = __floats2half2_rn(d, sum); LK_AUDIT_D(d, sum); }
}

// ---------------------------------------------------------------------------
// Q4_0 (planar) x Q8_1 dot for one 32-element block.
//   value = d4 * (d8*sumi - 8*sum_raw)
// Nibble layout: byte j of qs holds element j in its LOW nibble and element
// j+16 in its HIGH nibble.  So for uint32 word w (bytes j = 4w..4w+3):
//   low  nibbles -> elements 4w    .. 4w+3
//   high nibbles -> elements 16+4w .. 19+4w
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dot_q4_0_q8_1(const uint4 * __restrict__ qs,
                                               const __half * __restrict__ d,
                                               int blk,
                                               const block_q8_1 * __restrict__ x) {
    const uint4 q = __ldcs(qs + blk);
    const float d4 = __half2float(d[blk]);
    const int32_t * xq = reinterpret_cast<const int32_t *>(x->qs);
    const uint32_t * qw = reinterpret_cast<const uint32_t *>(&q);

    int sumi = 0;
#ifndef LK_LOADONLY
    // LK_LOADONLY builds a diagnostic variant that issues the same memory
    // traffic but skips the arithmetic.  Comparing the two isolates whether a
    // kernel is bound by instructions or by bandwidth.  Results are wrong; this
    // exists only to answer that question.
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const uint32_t v = qw[w];
        sumi = dp4a((int)(v & 0x0F0F0F0Fu),        xq[w],     sumi);  // elements 4w..4w+3
        sumi = dp4a((int)((v >> 4) & 0x0F0F0F0Fu), xq[4 + w], sumi);  // elements 16+4w..
    }
#else
    sumi = (int)(qw[0] & 0xFF) + (int)(qw[3] & 0xFF);
#endif
    const float d8  = __low2float(x->ds);
    const float sum = __high2float(x->ds);
    return d4 * (d8 * (float)sumi - 8.0f * sum);
}

// ---------------------------------------------------------------------------
// Q6_K (planar) x Q8_1 dot for one 256-element super-block.
//   value_i = d * sc[i/16] * (q6_i - 32),  q6 in 0..63, sc is int8.
// One super-block spans 8 Q8_1 activation blocks (256 = 8 x 32).
//
// q6_at() encodes the bit mapping directly and is transcriptionally identical
// to the mapping in tools/gguf.py, which was verified BIT-EXACT against
// gguf-py's authoritative dequantiser.  Deliberately scalar: correctness first,
// dp4a later (Phase 4).  Q6_K is only the tied lm_head + token lookup, both
// memory-bound rather than ALU-bound.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int q6_at(const uint8_t * __restrict__ QL,
                                     const uint8_t * __restrict__ QH, int idx) {
    // The 256 elements are processed as two 128-element halves.  For the second
    // half the ql window advances by 64 bytes and the qh window by 32 bytes --
    // exactly the n//2 / n//4 slicing in tools/gguf.py's dequantiser.
    const int n      = (idx >= 128) ? 128 : 0;
    const int r      = idx - n;
    const int qlbase = n >> 1;   // 0 or 64
    const int qhbase = n >> 2;   // 0 or 32
    if (r < 32) {
        const int l = r;
        return (QL[qlbase + l] & 0x0F) | (((QH[qhbase + l] >> 0) & 3) << 4);
    } else if (r < 64) {
        const int l = r - 32;
        return (QL[qlbase + l + 32] & 0x0F) | (((QH[qhbase + l] >> 2) & 3) << 4);
    } else if (r < 96) {
        const int l = r - 64;
        return (QL[qlbase + l] >> 4) | (((QH[qhbase + l] >> 4) & 3) << 4);
    } else {
        const int l = r - 96;
        return (QL[qlbase + l + 32] >> 4) | (((QH[qhbase + l] >> 6) & 3) << 4);
    }
}

__device__ __forceinline__ float dot_q6_K_q8_1(const uint8_t * __restrict__ ql,
                                               const uint8_t * __restrict__ qh,
                                               const int8_t  * __restrict__ scales,
                                               const __half  * __restrict__ d,
                                               int blk,
                                               const block_q8_1 * __restrict__ x) {
    const uint8_t * QL = ql + (size_t)blk * 128;
    const uint8_t * QH = qh + (size_t)blk * 64;
    const int8_t  * SC = scales + (size_t)blk * 16;
    const float dk = __half2float(d[blk]);

    float sumf = 0.0f;
#pragma unroll
    for (int nb = 0; nb < 8; ++nb) {                 // 8 Q8_1 blocks of 32
        const block_q8_1 * xb = x + nb;
        const float d8 = __low2float(xb->ds);
#pragma unroll
        for (int h = 0; h < 2; ++h) {                // two 16-groups, one int8 scale each
            const float sc = (float)SC[nb * 2 + h];
            float partial = 0.0f;
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const int q6 = q6_at(QL, QH, nb * 32 + h * 16 + e);
                partial += (float)(q6 - 32) * (float)xb->qs[h * 16 + e];
            }
            sumf += d8 * partial * sc;
        }
    }
    return dk * sumf;
}

// ---------------------------------------------------------------------------
// Warp-cooperative Q6_K x Q8_1 dot for ONE 256-weight super-block.
//
// The scalar dot_q6_K_q8_1 above is correct but leaves 24 of 32 lanes idle when
// a warp owns a whole row (sbpr = 8 superblocks).  Here all 32 lanes cooperate
// on a single super-block: lane L handles elements L, L+32, ..., L+224, and the
// 8 Q8_1 activation blocks line up exactly with those groups of 32.  Caller
// reduces across the warp once, after accumulating over all super-blocks.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dot_q6_K_q8_1_warp(const uint8_t * __restrict__ QL,
                                                    const uint8_t * __restrict__ QH,
                                                    const int8_t  * __restrict__ SC,
                                                    float dk,
                                                    const block_q8_1 * __restrict__ x,
                                                    int lane) {
    float acc = 0.0f;
#pragma unroll
    for (int g = 0; g < 8; ++g) {               // 8 groups of 32 = 256 elements
        const int e = g * 32 + lane;
        const float d8 = __low2float(x[g].ds);
        const int q6 = q6_at(QL, QH, e);
        acc += d8 * (float)SC[e >> 4] * (float)(q6 - 32) * (float)x[g].qs[lane];
    }
    return dk * acc;
}
