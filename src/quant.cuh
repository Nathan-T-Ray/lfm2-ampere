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
    if (lane == 0) y->ds = __floats2half2_rn(d, sum);
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
