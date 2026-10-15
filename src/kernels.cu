#include "kernels.cuh"
#include <cstdio>

// ---------------------------------------------------------------------------
__global__ void k_rmsnorm(const float * __restrict__ x, const float * __restrict__ w,
                          int n, float eps,
                          float * __restrict__ y32, __half * __restrict__ y16) {
    extern __shared__ float s_sq[];
    float sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i];
        sq += v * v;
    }
    s_sq[threadIdx.x] = sq;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int)threadIdx.x < o) s_sq[threadIdx.x] += s_sq[threadIdx.x + o];
        __syncthreads();
    }
    const float inv = rsqrtf(s_sq[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i] * inv * w[i];
        if (y32) y32[i] = v;
        if (y16) y16[i] = __float2half(v);
    }
}

__global__ void k_rmsnorm_q8_1(const float * __restrict__ x, const float * __restrict__ w,
                               int n, float eps, float * __restrict__ y32,
                               __half * __restrict__ y16, block_q8_1 * __restrict__ yq8) {
    __shared__ float s_sq[256];
    extern __shared__ float s_h[];          // generic path only
    const int tid = threadIdx.x;

    if (n == 2048) {
        // Every caller passes LK_HIDDEN, so the shape is a compile-time constant.
        //
        // x and w are each loaded ONCE and held in registers.  The reduction
        // forces a barrier between the sum of squares and the normalise, and the
        // previous version re-issued all sixteen loads across that barrier --
        // a second HBM round trip for data already in flight.  The summation
        // order is untouched: the same eight strided terms per thread, then the
        // same tree.
        float xv[8], wv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { xv[j] = x[tid + j * 256]; wv[j] = w[tid + j * 256]; }
        float sq = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) sq += xv[j] * xv[j];
        s_sq[tid] = sq;
        __syncthreads();
        // Tree levels with stride >= 32 pair threads in different warps and need
        // shared memory.  The remaining five (16, 8, 4, 2, 1) pair lanes inside
        // warp 0 and are exactly a shfl_xor butterfly producing the same value
        // in lane 0 by the same pairings -- five fewer barriers.
#pragma unroll
        for (int o = 128; o > 16; o >>= 1) {   // o = 128, 64, 32
            if ((int) tid < o) s_sq[tid] += s_sq[tid + o];
            __syncthreads();
        }
        if (tid < 32) {
            float v = s_sq[tid];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (tid == 0) s_sq[0] = v;
        }
        __syncthreads();
        const float inv = rsqrtf(s_sq[0] / 2048.0f + eps);

        // Normalise AND quantise in one pass.  Warp `w` owns elements
        // 32w + 256j .. +31, which is exactly Q8_1 block 8j + w, so the
        // quantisation needs no shared staging at all -- the old version wrote
        // 8 KB to shared, synchronised, and read it straight back.
        const int warp = tid >> 5;
        const int lane = tid & 31;
        // Two blocks at a time, with their reduction chains written adjacently.
        //
        // Each Q8_1 block needs a 5-step shfl_xor sum and there are eight
        // independent blocks per warp.  Issued one at a time those five steps
        // are a serial ~125-cycle dependency chain, eight times over: measured
        // at 1.07 us of this kernel's 2.11 us of execution, the single largest
        // item in the 81 norms that together cost 220 us per token.
        // Two blocks at a time, with their reduction chains written adjacently.
        //
        // Each Q8_1 block needs a 5-step shfl_xor sum and there are eight
        // independent blocks per warp.  Issued one at a time those five steps
        // are a serial ~125-cycle dependency chain, eight times over: measured
        // at 1.07 us of this kernel's 2.11 us of execution, the largest single
        // item in the 81 norms that cost 220 us per token.
        //
        // Two at a time measures best (1.07 -> 0.86 us).  Four and eight were
        // tried and are worse -- the extra live registers cost more than the
        // overlap gains.  Each block still runs the same five steps in the same
        // order from the same starting value, so this is bit-identical.
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            const int i0 = tid + j * 256, i1 = tid + (j + 1) * 256;
            const float t0 = xv[j] * inv * wv[j];
            const float t1 = xv[j + 1] * inv * wv[j + 1];
            if (y32) { y32[i0] = t0; y32[i1] = t1; }
            if (y16) { y16[i0] = __float2half(t0); y16[i1] = __float2half(t1); }
            if (yq8) quantize_q8_1_warp2(t0, t1, yq8 + warp + 8 * j, yq8 + warp + 8 * j + 8, lane);
        }
        return;
    }

    // ---- generic fallback, unchanged ---------------------------------------
    float sq = 0.0f;
    for (int i = tid; i < n; i += 256) { const float v = x[i]; sq += v * v; }
    s_sq[tid] = sq;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if ((int) tid < o) s_sq[tid] += s_sq[tid + o];
        __syncthreads();
    }
    const float inv = rsqrtf(s_sq[0] / (float) n + eps);
    for (int i = tid; i < n; i += 256) {
        const float v = x[i] * inv * w[i];
        if (y32) y32[i] = v;
        if (y16) y16[i] = __float2half(v);
        s_h[i] = v;
    }
    __syncthreads();
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nwarp = blockDim.x >> 5;
    for (int b = warp; b < n / 32; b += nwarp) {
        quantize_q8_1_warp(s_h[b * 32 + lane], yq8 + b, lane);
    }
}

__global__ void k_quantize_q8_1(const float * __restrict__ x,
                                block_q8_1 * __restrict__ y, int nblocks) {
    const int b = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (b >= nblocks) return;
    quantize_q8_1_warp(x[b * 32 + lane], y + b, lane);
}

// ---------------------------------------------------------------------------
__global__ void k_gemv_q4_0(const uint4 * __restrict__ qs, const __half * __restrict__ d,
                            const block_q8_1 * __restrict__ xq,
                            int out_rows, int bpr, float * __restrict__ y, int add_residual) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = warp * LK_GEMV_ROWS;
    if (row0 >= out_rows) return;

    float acc[LK_GEMV_ROWS];
#pragma unroll
    for (int r = 0; r < LK_GEMV_ROWS; ++r) acc[r] = 0.0f;

    for (int k = lane; k < bpr; k += 32) {
        const block_q8_1 * xk = xq + k;
#pragma unroll
        for (int r = 0; r < LK_GEMV_ROWS; ++r) {
            const int row = row0 + r;
            if (row < out_rows) {
                acc[r] += dot_q4_0_q8_1(qs + (size_t)row * bpr + k, d + (size_t)row * bpr + k, 0, xk);
            }
        }
    }
    // row0 is warp-uniform, so these reductions are reached by every lane
#pragma unroll
    for (int r = 0; r < LK_GEMV_ROWS; ++r) {
        const int row = row0 + r;
        if (row < out_rows) {
            const float a = warp_sum(acc[r]);
            if (lane == 0) y[row] = add_residual ? (y[row] + a) : a;
        }
    }
}

// ---------------------------------------------------------------------------
__global__ void k_gemv_q6_K(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                            const int8_t * __restrict__ sc, const __half * __restrict__ d,
                            const block_q8_1 * __restrict__ xq,
                            int out_rows, int sbpr, float * __restrict__ y) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= out_rows) return;
    // 8 super-blocks per 2048-dim row.  The loop bound is a runtime argument,
    // so nvcc cannot unroll it and the warp pays a full ~500-cycle HBM round
    // trip per super-block -- 8 serialised stalls per row, which measured as
    // 511 GB/s against a 956 GB/s achievable ceiling on a kernel that is 15% of
    // all bytes moved.  Specialising the only value this model ever passes lets
    // the eight independent loads issue together.  Unrolling does not reorder
    // any addition, so the result is bit-identical.
    float acc = 0.0f;
    if (sbpr == 8) {
#pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            const size_t blk = (size_t)warp * 8 + sb;
            acc += dot_q6_K_q8_1_warp(ql + blk * 128, qh + blk * 64, sc + blk * 16,
                                      __half2float(d[blk]), xq + (size_t)sb * 8, lane);
        }
    } else {
        for (int sb = 0; sb < sbpr; ++sb) {
            const size_t blk = (size_t)warp * sbpr + sb;
            acc += dot_q6_K_q8_1_warp(ql + blk * 128, qh + blk * 64, sc + blk * 16,
                                      __half2float(d[blk]), xq + (size_t)sb * 8, lane);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[warp] = acc;
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Q6_K lm_head, dp4a path.
//
// The scalar version issues FOUR 1-BYTE loads per element (ql, qh, the scale and
// the activation) plus three int->float converts and three multiplies, ~17
// instructions per element.  At 128000 rows x 64 elements per lane that is
// ~139M warp-instructions, i.e. ~294 us of pure issue on this part -- which is
// what the 511 GB/s (versus a 956 GB/s ceiling) actually measures.  Unrolling
// the superblock loop changed nothing precisely because it does not change the
// instruction count.
//
// Q6_K is built for this.  Within a 128-element half, ql byte b holds the low
// nibble of element b and the high nibble of element b+64, and qh byte b holds
// the top two bits of elements b, b+32, b+64 and b+96 at shifts 0, 2, 4, 6.
// So one 32-bit load of ql plus one of qh yields four packed 6-bit values, and
// one dp4a consumes them.  That is 2 loads and ~6 ALU ops per 4 elements instead
// of 4 loads and ~13.
//
// Lane assignment: lane L owns the 8 CONSECUTIVE elements 8L..8L+7 of the
// superblock.  Those always fall inside one 32-element Q8_1 block (index L/4)
// and one 16-element scale group (index L/2), so each lane needs exactly one
// activation scale, one weight scale, and two dp4a groups.  Every byte offset
// this produces is a multiple of 4, so both arrays load as aligned 32-bit words.
//
// The reduction ORDER differs from the scalar kernel, so the result is not
// bit-identical -- it is closer to exact, since the integer accumulation is
// exact and only one float multiply per 8 elements rounds.
__global__ void k_gemv_q6_K_dp4a(const uint8_t * __restrict__ ql1, const uint8_t * __restrict__ qh1,
                                 const int8_t * __restrict__ sc1, const __half * __restrict__ d1,
                                 const block_q8_1 * __restrict__ xq,
                                 int out_rows, int sbpr, float * __restrict__ y) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= out_rows) return;

    // ---- per-lane geometry: constant across all superblocks -----------------
    const int s        = 8 * (lane & 15);   // element offset inside the 128-half
    const int hi_nib   = (s >= 64);         // high or low nibble of ql
    const int k        = s >> 5;            // 0..3 -> qh bit pair
    const int ql_idx   = ((lane >= 16) ? 64 : 0) + (hi_nib ? s - 64 : s);
    const int qh_idx   = ((lane >= 16) ? 32 : 0) + s - 32 * k;
    const int qh_sh    = 2 * k;
    const int u_off    = 8 * (lane & 3);    // byte offset inside the Q8_1 block
    const int blk      = lane >> 2;         // which of the 8 Q8_1 blocks
    const int sci      = lane >> 1;         // which of the 16 scales

    float acc = 0.0f;
    for (int sb = 0; sb < sbpr; ++sb) {
        const size_t sblk = (size_t) warp * sbpr + sb;
        const uint8_t * QL = ql1 + sblk * 128;
        const uint8_t * QH = qh1 + sblk * 64;
        const block_q8_1 * X = xq + (size_t) sb * 8 + blk;

        const int   scv = (int) sc1[sblk * 16 + sci];
        const float d8  = __low2float(X->ds);
        const uint32_t ql0 = *(const uint32_t *) (QL + ql_idx);
        const uint32_t ql4 = *(const uint32_t *) (QL + ql_idx + 4);
        const uint32_t qh0 = *(const uint32_t *) (QH + qh_idx);
        const uint32_t qh4 = *(const uint32_t *) (QH + qh_idx + 4);

        int a = 0;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const uint32_t qw = t ? ql4 : ql0;
            const uint32_t hw = t ? qh4 : qh0;
            const uint32_t vil = hi_nib ? ((qw >> 4) & 0x0F0F0F0Fu) : (qw & 0x0F0F0F0Fu);
            const uint32_t vih = ((hw >> qh_sh) & 0x03030303u) << 4;
            const int vi = __vsubss4((int) (vil | vih), 0x20202020);
            const int u  = *(const int32_t *) (X->qs + u_off + t * 4);
            a = __dp4a(vi, u, a);
        }
        acc += __half2float(d1[sblk]) * (d8 * (float) (a * scv));
    }
    acc = warp_sum(acc);
    if (lane == 0) y[warp] = acc;
}

// Produces the SwiGLU activation and, optionally, its Q8_1 form.
//
// Every caller feeds the result straight into k_quantize_q8_1, which costs a
// whole graph node (0.6 us of pure dispatch, measured) to move 24 KB.  Here a
// warp already owns 32 consecutive outputs, which is exactly one Q8_1 block, so
// the quantisation is free apart from the reduction.  n is always a multiple of
// 32 and the block size is too, so no warp is ever partially active and the
// full-mask shuffles inside quantize_q8_1_warp stay well defined.
__global__ void k_swiglu(const float * __restrict__ gate, const float * __restrict__ up,
                         int n, float * __restrict__ out, block_q8_1 * __restrict__ yq8) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float v = silu_f(gate[i]) * up[i];
        if (out) out[i] = v;
        if (yq8) quantize_q8_1_warp(v, yq8 + (i >> 5), threadIdx.x & 31);
    }
}

// ---------------------------------------------------------------------------
__global__ void k_shortconv(const float * __restrict__ bcx, int t,
                            const float * __restrict__ conv_w, float * __restrict__ hist,
                            float * __restrict__ y, block_q8_1 * __restrict__ yq8) {
    // t selects which token of the bcx buffer we are processing
    const float * B = bcx;
    const float * C = bcx + LK_HIDDEN;
    const float * X = bcx + 2 * LK_HIDDEN;
    for (int ch = blockIdx.x * blockDim.x + threadIdx.x; ch < LK_HIDDEN;
         ch += gridDim.x * blockDim.x) {
        const float bx = B[ch] * X[ch];
        float * h = hist + ch * 2;
        const float h0 = h[0], h1 = h[1];
        const float cw0 = conv_w[0 + ch * 3];
        const float cw1 = conv_w[1 + ch * 3];
        const float cw2 = conv_w[2 + ch * 3];
        const float conv = h0 * cw0 + h1 * cw1 + bx * cw2;
        h[0] = h1;
        h[1] = bx;
        const float v = C[ch] * conv;
        if (y) y[ch] = v;
        if (yq8) quantize_q8_1_warp(v, yq8 + (ch >> 5), threadIdx.x & 31);
    }
    (void)t;
}

// ---------------------------------------------------------------------------
// MoE router: logits[j] = dot(h, gate_inp[j]) for the 64 experts, then
// sigmoid, top-4 by (sigmoid + bias), weights from the UNBIASED sigmoid,
// renormalised.  Matches llama.cpp's SIGMOID gating path exactly.
//
// Originally this was ONE block of 64 threads, each computing a whole
// 2048-element dot serially -- i.e. 2048 dependent loads per thread, 38 times
// per token, on a GPU with 84 SMs sitting idle.  Now one WARP per expert
// (64 warps = 2048 threads), lanes striding the row, with a warp reduction.
// The sigmoid/top-4 logic is unchanged and still runs on one thread, so the
// result is identical.
// The router's dot products, one warp per expert.
//
// This is the SAME arithmetic as the single-block version it replaces: lane L
// accumulates i = L, L+32, ... in increasing order and is reduced by the same
// shfl_xor butterfly, so all 64 logits come out bit-identical.  Only the
// scheduling changes.  The original ran ONE 512-thread block, i.e. 16 warps of
// memory-level parallelism for a 512 KB read; at ~500 cycles of HBM latency
// that caps a block near 45 GB/s, and it measured 23.6 us per layer -- 20% of
// the whole decode step, for 1.4% of the bytes.  Sixty-four single-warp blocks
// put the same read on 64 different SMs, and batching the loads keeps several
// in flight per warp.
#define LK_ROUTER_UNROLL 16
__global__ void __launch_bounds__(32) k_router_dot(
        const float * __restrict__ h, const float * __restrict__ gate_inp,
        float * __restrict__ logits) {
    const int e = blockIdx.x;
    const int lane = threadIdx.x;
    const float * row = gate_inp + (size_t)e * LK_HIDDEN;
    float acc = 0.0f;
    for (int base = 0; base < LK_HIDDEN; base += 32 * LK_ROUTER_UNROLL) {
        float v[LK_ROUTER_UNROLL], hv[LK_ROUTER_UNROLL];
#pragma unroll
        for (int j = 0; j < LK_ROUTER_UNROLL; ++j) v[j]  = row[base + j * 32 + lane];
#pragma unroll
        for (int j = 0; j < LK_ROUTER_UNROLL; ++j) hv[j] = h[base + j * 32 + lane];
#pragma unroll
        for (int j = 0; j < LK_ROUTER_UNROLL; ++j) acc += v[j] * hv[j];
    }
    acc = warp_sum(acc);
    if (lane == 0) logits[e] = acc;
}

// The selection half: sigmoid + bias ranking, top-4, renormalised.  Byte for
// byte the original body, except that the 64 logits now arrive from global
// memory instead of a block-local array.
__global__ void k_router_top4(const float * __restrict__ logits,
                              const float * __restrict__ bias,
                              int * __restrict__ idx_out, float * __restrict__ w_out) {
    __shared__ float s_sel[LK_N_EXPERTS];
    __shared__ float s_sig[LK_N_EXPERTS];
    const int t = threadIdx.x;
    // Sigmoids are computed one per lane.  A single lane computing all 64 made
    // this kernel cost 15.3 us: one active lane issues one instruction per cycle
    // and stalls for the full latency of every dependent step inside expf.
    if (t < LK_N_EXPERTS) {
        const float g = 1.0f / (1.0f + expf(-logits[t]));
        s_sig[t] = g;
        s_sel[t] = g + bias[t];
    }
    __syncthreads();
    if (t >= 32) return;

    // Selection, one warp.
    //
    // Walking 64 experts through a register insertion in ONE lane is a 64-deep
    // serial dependency: each iteration's p0..p3 depend on the previous one, so
    // the chain cannot be pipelined and measured 2.78 us of pure latency per
    // layer, 106 us per token, for zero bytes of memory traffic.
    //
    // Four rounds of "global argmax over what is left" is the same answer: the
    // insertion maintains the list sorted descending, so round r recovers
    // exactly p_r.  Ties go to the lower expert index.  Each round is five
    // shfl_xor steps carrying a (value, index) pair.
    static_assert(LK_TOP_K == 4, "four selection rounds assume top-4");
    const int lane = t;
    float v0 = s_sel[lane],      v1 = s_sel[lane + 32];
    const int i0 = lane,         i1 = lane + 32;
    int ord[LK_TOP_K];
#pragma unroll
    for (int r = 0; r < LK_TOP_K; ++r) {
        float mv = (v0 >= v1) ? v0 : v1;
        int   mi = (v0 >= v1) ? i0 : i1;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, mv, o);
            const int   oi = __shfl_xor_sync(0xffffffffu, mi, o);
            if (ov > mv || (ov == mv && oi < mi)) { mv = ov; mi = oi; }
        }
        ord[r] = mi;                       // the butterfly leaves all lanes equal
        if (i0 == mi) { v0 = -1e30f; }
        if (i1 == mi) { v1 = -1e30f; }
    }
    if (lane != 0) return;
    float wsum = 0.0f;
    float wv[LK_TOP_K];
#pragma unroll
    for (int k = 0; k < LK_TOP_K; ++k) {
        wv[k] = s_sig[ord[k]];             // unbiased sigmoid, computed above
        wsum += wv[k];
    }
    if (wsum < LK_TOPK_GUARD) wsum = LK_TOPK_GUARD;
#pragma unroll
    for (int k = 0; k < LK_TOP_K; ++k) { idx_out[k] = ord[k]; w_out[k] = wv[k] / wsum; }
}

// ---------------------------------------------------------------------------
__global__ void k_attention_decode(const float * __restrict__ q,
                                   const __half * __restrict__ kc,
                                   const __half * __restrict__ vc,
                                   const int * __restrict__ d_pos, float * __restrict__ out,
                                   block_q8_1 * __restrict__ yq8) {
    const int seq_len = *d_pos + 1;
    const int head = blockIdx.x;
    const int kv_head = head / (LK_N_HEADS / LK_N_KV_HEADS);
    __shared__ float s_q[LK_HEAD_DIM];
    __shared__ float s_score[LK_ATTN_MAX_SEQ];
    if (threadIdx.x < LK_HEAD_DIM) s_q[threadIdx.x] = q[head * LK_HEAD_DIM + threadIdx.x];
    __syncthreads();

    if (seq_len > LK_ATTN_MAX_SEQ) {          // host asserts this; belt-and-braces
        if (threadIdx.x == 0 && blockIdx.x == 0)
            printf("FATAL: seq_len %d exceeds LK_ATTN_MAX_SEQ %d\n", seq_len, LK_ATTN_MAX_SEQ);
        return;
    }
    const int lim = seq_len;
    float m = -1e30f;
    for (int t = threadIdx.x; t < lim; t += blockDim.x) {
        const __half * k = kc + ((size_t)t * LK_N_KV_HEADS + kv_head) * LK_HEAD_DIM;
        float acc = 0.0f;
        // One token's 64 dims are 128 contiguous bytes = one cache line; load it
        // as 8 x uint4 rather than 64 x 2-byte loads.  Same dims, same order.
#pragma unroll
        for (int q = 0; q < LK_HEAD_DIM / 8; ++q) {
            const uint4 w = ((const uint4 *) k)[q];
            const __half2 * h2 = (const __half2 *) &w;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int d = q * 8 + i * 2;
                acc += s_q[d]     * __half2float(__low2half(h2[i]));
                acc += s_q[d + 1] * __half2float(__high2half(h2[i]));
            }
        }
        acc *= LK_ATTN_SCALE;
        s_score[t] = acc;
        m = fmaxf(m, acc);
    }
    // block max
    // Sized by the block, which is deliberately much wider than LK_HEAD_DIM:
    // see LK_ATTN_BLOCK.
    __shared__ float s_m[LK_ATTN_BLOCK];
    s_m[threadIdx.x] = m;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int)threadIdx.x < o) s_m[threadIdx.x] = fmaxf(s_m[threadIdx.x], s_m[threadIdx.x + o]);
        __syncthreads();
    }
    const float mglobal = s_m[0];
    __syncthreads();

    float lsum = 0.0f;
    for (int t = threadIdx.x; t < lim; t += blockDim.x) {
        const float p = __expf(s_score[t] - mglobal);
        s_score[t] = p;
        lsum += p;
    }
    s_m[threadIdx.x] = lsum;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int)threadIdx.x < o) s_m[threadIdx.x] += s_m[threadIdx.x + o];
        __syncthreads();
    }
    const float inv_l = 1.0f / s_m[0];
    __syncthreads();

    for (int d = threadIdx.x; d < LK_HEAD_DIM; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < lim; ++t) {
            const __half * v = vc + ((size_t)t * LK_N_KV_HEADS + kv_head) * LK_HEAD_DIM;
            acc += s_score[t] * __half2float(v[d]);
        }
        const float o = acc * inv_l;
        if (out) out[head * LK_HEAD_DIM + d] = o;
        // LK_HEAD_DIM is 64, so threads 0..31 and 32..63 of each block hold the
        // two Q8_1 blocks belonging to this head.  Both warps are fully active.
        if (yq8) quantize_q8_1_warp(o, yq8 + 2 * head + (threadIdx.x >> 5), threadIdx.x & 31);
    }
}

// ---------------------------------------------------------------------------
__global__ void k_embed_q6_K(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                             const int8_t * __restrict__ sc, const __half * __restrict__ d,
                             const int * __restrict__ d_token, float * __restrict__ x) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= LK_HIDDEN) return;
    const int token = *d_token;
    const int sb = tid / 256;              // super-block index within the row
    const int e  = tid % 256;              // element within the super-block
    const size_t blk = (size_t)token * 8 + sb;
    const float dk = __half2float(d[blk]);
    const int q6 = q6_at(ql + blk * 128, qh + blk * 64, e);
    x[tid] = dk * (float)sc[blk * 16 + e / 16] * (float)(q6 - 32);
}

// ---------------------------------------------------------------------------
__global__ void k_qk_norm_rope(float * __restrict__ qf, float * __restrict__ kf,
                               const float * __restrict__ vf,
                               const float * __restrict__ qn, const float * __restrict__ kn,
                               const int * __restrict__ d_pos,
                               __half * __restrict__ kc, __half * __restrict__ vc) {
    const float pos = (float)(*d_pos);
    __shared__ float s_head[LK_HEAD_DIM];
    __shared__ float s_red[LK_HEAD_DIM];
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;
    const bool is_q = (blk < LK_N_HEADS);
    const int head = is_q ? blk : (blk - LK_N_HEADS);
    float * dst = is_q ? (qf + (size_t)head * LK_HEAD_DIM)
                       : (kf + (size_t)head * LK_HEAD_DIM);
    const float * nw = is_q ? qn : kn;

    const float v0 = dst[tid];
    s_red[tid] = v0 * v0;
    __syncthreads();
    for (int o = LK_HEAD_DIM / 2; o > 0; o >>= 1) {
        if (tid < o) s_red[tid] += s_red[tid + o];
        __syncthreads();
    }
    const float inv = rsqrtf(s_red[0] / (float)LK_HEAD_DIM + LK_EPS);
    s_head[tid] = v0 * inv * nw[tid];
    __syncthreads();

    if (tid < LK_HEAD_DIM / 2) {           // NEOX half-split rotation
        const float invf = powf(LK_ROPE_THETA, -2.0f * (float)tid / (float)LK_HEAD_DIM);
        float c, s;
        sincosf(pos * invf, &s, &c);
        const float x0 = s_head[tid], x1 = s_head[tid + LK_HEAD_DIM / 2];
        s_head[tid]                   = x0 * c - x1 * s;
        s_head[tid + LK_HEAD_DIM / 2] = x0 * s + x1 * c;
    }
    __syncthreads();
    dst[tid] = s_head[tid];

    if (!is_q) {
        const size_t off = ((size_t)(*d_pos) * LK_N_KV_HEADS + head) * LK_HEAD_DIM;
        kc[off + tid] = __float2half(s_head[tid]);
        vc[off + tid] = __float2half(vf[(size_t)head * LK_HEAD_DIM + tid]);
    }
}

// ---------------------------------------------------------------------------
__global__ void k_accum_scaled(float * __restrict__ dst, const float * __restrict__ src,
                               float w, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        dst[i] += w * src[i];
}

// ---------------------------------------------------------------------------
// Two-stage argmax.
//
// The single-block scan kept exactly one load in flight per thread: 256 threads
// is 8 warps is 1 KB outstanding, which at ~500 cycles of HBM latency works out
// at ~2.8 GB/s -- and it measured 145 us to read 512 KB, 4% of the step for
// 0.03% of the bytes.  Stage 1 spreads the read over 125 blocks; stage 2
// reduces the 125 block winners.  Each thread owns a CONTIGUOUS run of
// elements and every level of the reduction breaks ties toward the lower index,
// so the answer is still the lowest index attaining the maximum.
#define LK_ARGMAX_BLOCKS      125
#define LK_ARGMAX_PER_THREAD  4

// Gate G4b: record which experts the router actually chose.
//
// Phase 5 speculative decoding is only worth building if verifying K tokens in
// one pass touches far fewer than 4K experts per layer -- the whole win is
// reading each expert's weights once and applying them to several tokens.  That
// depends entirely on how skewed the routing is in practice, which cannot be
// argued from the architecture and has to be measured.
//
// Written from inside the CUDA graph, indexed by the device-side position, so
// the capture stays position-independent like everything else in the step.
__global__ void k_route_log(const int * __restrict__ idx, const int * __restrict__ d_pos,
                            int il, int * __restrict__ log) {
    if (threadIdx.x >= LK_TOP_K) return;
    const size_t p = (size_t) (*d_pos);
    log[(p * LK_N_LAYERS + il) * LK_TOP_K + threadIdx.x] = idx[threadIdx.x];
}

__global__ void k_argmax_part(const float * __restrict__ v,
                              float * __restrict__ pv, int * __restrict__ pi) {
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = tid * LK_ARGMAX_PER_THREAD;
    float best = -1e30f; int bi = base;
#pragma unroll
    for (int j = 0; j < LK_ARGMAX_PER_THREAD; ++j) {
        const float t = v[base + j];
        if (t > best) { best = t; bi = base + j; }
    }
    __shared__ float sv[256];
    __shared__ int   si[256];
    sv[threadIdx.x] = best; si[threadIdx.x] = bi;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int)threadIdx.x < o && sv[threadIdx.x + o] > sv[threadIdx.x]) {
            sv[threadIdx.x] = sv[threadIdx.x + o];
            si[threadIdx.x] = si[threadIdx.x + o];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { pv[blockIdx.x] = sv[0]; pi[blockIdx.x] = si[0]; }
}

__global__ void k_argmax_final(const float * __restrict__ pv, const int * __restrict__ pi,
                               int n, int * __restrict__ d_token, int * __restrict__ d_pos) {
    __shared__ float sv[256];
    __shared__ int   si[256];
    const int tid = threadIdx.x;
    float best = -1e30f; int bi = 0;
    for (int i = tid; i < n; i += blockDim.x)
        if (pv[i] > best) { best = pv[i]; bi = pi[i]; }
    sv[tid] = best; si[tid] = bi;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (tid < o && sv[tid + o] > sv[tid]) { sv[tid] = sv[tid + o]; si[tid] = si[tid + o]; }
        __syncthreads();
    }
    if (tid == 0) { *d_token = si[0]; atomicAdd(d_pos, 1); }
}

// ---------------------------------------------------------------------------
// Gate and up projections for all 4 experts, in ONE launch.
//
// They are the same kernel applied to two adjacent matrices (ffn_gate_exps and
// ffn_up_exps) with the same shape, and each launch costs a whole graph node --
// 0.6 us of dispatch, measured.  Folding them into grid.y = 2*LK_TOP_K removes
// 38 nodes per token.  The per-expert arithmetic is untouched: blockIdx.y & 3
// still selects the expert and the row/activation indexing is identical, so the
// result is bit-for-bit what two launches produced.
// Three Q4_0 GEMVs in one launch, for q/k/v.
//
// GEMV efficiency tracks the number of WARPS in flight, not bytes per warp:
// measured 512 warps -> 358 GB/s, 2048 -> ~720, 6144 -> 852, 128000 -> 939.
// attn_k and attn_v are 512 rows each, so as separate launches they leave most
// SMs with under one block and run at a third of the ceiling.  One flat grid
// over all three matrices puts 3072 warps in flight at once and removes two
// graph nodes per layer.
//
// blockIdx.x covers the concatenated row space; a warp is assigned to whichever
// matrix its row falls in.  Each row's dot is computed exactly as
// k_gemv_q4_0 computes it -- same j loop, same warp_sum -- so the result is
// bit-identical.
__global__ void k_gemv_q4_0_3(const uint4 * __restrict__ qs0, const __half * __restrict__ d0,
                              int rows0, float * __restrict__ y0,
                              const uint4 * __restrict__ qs1, const __half * __restrict__ d1,
                              int rows1, float * __restrict__ y1,
                              const uint4 * __restrict__ qs2, const __half * __restrict__ d2,
                              int rows2, float * __restrict__ y2,
                              const block_q8_1 * __restrict__ xq, int bpr) {
    const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const uint4  * qs; const __half * d; float * y; int row;
    if (gw < rows0) {
        row = gw;            qs = qs0; d = d0; y = y0;
    } else if (gw < rows0 + rows1) {
        row = gw - rows0;    qs = qs1; d = d1; y = y1;
    } else {
        row = gw - rows0 - rows1; qs = qs2; d = d2; y = y2;
    }
    const uint4  * qrow = qs + (size_t) row * bpr;
    const __half * drow = d  + (size_t) row * bpr;
    float acc = 0.0f;
    for (int j = lane; j < bpr; j += 32) acc += dot_q4_0_q8_1(qrow, drow, j, xq + j);
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

__global__ void k_gemv_q4_0_exp2(const uint4 * __restrict__ qs_a, const __half * __restrict__ d_a,
                                 const uint4 * __restrict__ qs_b, const __half * __restrict__ d_b,
                                 const block_q8_1 * __restrict__ xq,
                                 int out_rows, int bpr, int rows_per_expert,
                                 int xq_stride,
                                 const int * __restrict__ d_idx,
                                 float * __restrict__ y_a, float * __restrict__ y_b) {
    const int half = blockIdx.y >> 2;               // 0 = gate, 1 = up
    const int k    = blockIdx.y & (LK_TOP_K - 1);   // which of the 4 experts
    const uint4  * qs = half ? qs_b : qs_a;
    const __half * d  = half ? d_b  : d_a;
    float        * y  = half ? y_b  : y_a;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= out_rows) return;
    const size_t base = (size_t) d_idx[k] * rows_per_expert * bpr;
    const uint4  * qrow = qs + base + (size_t) warp * bpr;
    const __half * drow = d  + base + (size_t) warp * bpr;
    float acc = 0.0f;
    const block_q8_1 * xq_k = xq + (size_t) k * xq_stride;
    for (int j = lane; j < bpr; j += 32) acc += dot_q4_0_q8_1(qrow, drow, j, xq_k + j);
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) k * out_rows + warp] = acc;
}

__global__ void k_gemv_q4_0_exp(const uint4 * __restrict__ qs, const __half * __restrict__ d,
                                const block_q8_1 * __restrict__ xq,
                                int out_rows, int bpr, int rows_per_expert,
                                int xq_stride,
                                const int * __restrict__ d_idx,
                                float * __restrict__ y) {
    const int k = blockIdx.y;                       // which of the 4 experts
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= out_rows) return;
    const size_t base = (size_t)d_idx[k] * rows_per_expert * bpr;
    const uint4 * qrow = qs + base + (size_t)warp * bpr;
    const __half * drow = d + base + (size_t)warp * bpr;
    float acc = 0.0f;
    const block_q8_1 * xq_k = xq + (size_t)k * xq_stride;
    for (int j = lane; j < bpr; j += 32) acc += dot_q4_0_q8_1(qrow, drow, j, xq_k + j);
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t)k * out_rows + warp] = acc;
}

// ---------------------------------------------------------------------------
__global__ void k_moe_accum4(float * __restrict__ dst, const float * __restrict__ down4,
                             const float * __restrict__ d_w, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float a = 0.0f;
#pragma unroll
        for (int k = 0; k < LK_TOP_K; ++k) a += d_w[k] * down4[(size_t)k * n + i];
        dst[i] += a;
    }
}


// ---------------------------------------------------------------------------
// Split-K attention (FlashDecoding shape).
//
// One block per query head means 32 blocks on 84 SMs, and each block scans the
// WHOLE context twice -- once for scores, once for the weighted V sum.  At 8k
// context that measured 476 us per layer at 34 GB/s against a 956 GB/s ceiling,
// 58% of the entire decode step.  Three cheaper fixes were tried and falsified:
// batching the V loads, transposing the KV cache (worse -- the token-major
// layout was already coalesced), and widening the block to 1024 threads
// (neutral).  Ablations put the score loop at 33.7 ns per context token and the
// V loop at 24.9, so both halves need the parallelism, not just one.
//
// So partition the context: 32 heads x LK_ATTN_SPLIT blocks, each owning one
// contiguous slice, each producing a partial (max, denom, weighted-V vector).
// A second kernel combines them in a fixed order.  Because the slices are
// contiguous and the combine is ordered, the only difference from the serial
// kernel is where the rounding happens -- which does change the result, and is
// the reason this is measured against the parity gate rather than assumed.
#define LK_ATTN_CHUNK ((LK_ATTN_MAX_SEQ + LK_ATTN_SPLIT - 1) / LK_ATTN_SPLIT)

__global__ void k_attn_split(const float * __restrict__ q,
                             const __half * __restrict__ kc, const __half * __restrict__ vc,
                             const int * __restrict__ d_pos, float * __restrict__ part) {
    const int seq_len = *d_pos + 1;
    const int head = blockIdx.x;
    const int sp   = blockIdx.y;
    const int kv_head = head / (LK_N_HEADS / LK_N_KV_HEADS);
    __shared__ float s_q[LK_HEAD_DIM];
    __shared__ float s_score[LK_ATTN_CHUNK];
    __shared__ float s_m[256];
    if (threadIdx.x < LK_HEAD_DIM) s_q[threadIdx.x] = q[head * LK_HEAD_DIM + threadIdx.x];
    __syncthreads();

    float * P = part + ((size_t) head * LK_ATTN_SPLIT + sp) * LK_ATTN_PARTW;
    const int chunk = (seq_len + LK_ATTN_SPLIT - 1) / LK_ATTN_SPLIT;
    const int t0 = sp * chunk;
    const int t1 = (t0 + chunk < seq_len) ? (t0 + chunk) : seq_len;
    if (t0 >= t1) {                       // this slice owns no tokens
        if (threadIdx.x == 0) { P[0] = -1e30f; P[1] = 0.0f; }
        for (int d = threadIdx.x; d < LK_HEAD_DIM; d += blockDim.x) P[2 + d] = 0.0f;
        return;
    }
    const int n = t1 - t0;
    float m = -1e30f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const __half * k = kc + ((size_t)(t0 + j) * LK_N_KV_HEADS + kv_head) * LK_HEAD_DIM;
        float acc = 0.0f;
        // The 64 dims of one token are 128 contiguous bytes -- exactly one cache
        // line -- and the loop below used to fetch them with 64 separate 2-byte
        // loads.  At a fixed dim the 32 lanes of a warp are on 32 different
        // lines (one token each), so that is 32 lines touched per instruction,
        // 64 times over: 2048 L1 accesses per warp per 32 tokens where 256 will
        // do.  Loading the line as 8 x uint4 cuts the instruction count 8x.
        //
        // Dims are still accumulated d = 0,1,2,...,63 in that exact order, so
        // the result is bit-identical.  k is 128-byte aligned (one token's row),
        // so the uint4 loads are aligned.
#pragma unroll
        for (int q = 0; q < LK_HEAD_DIM / 8; ++q) {
            const uint4 w = ((const uint4 *) k)[q];
            const __half2 * h2 = (const __half2 *) &w;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int d = q * 8 + i * 2;
                acc += s_q[d]     * __half2float(__low2half(h2[i]));
                acc += s_q[d + 1] * __half2float(__high2half(h2[i]));
            }
        }
        acc *= LK_ATTN_SCALE;
        s_score[j] = acc;
        m = fmaxf(m, acc);
    }
    s_m[threadIdx.x] = m;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) s_m[threadIdx.x] = fmaxf(s_m[threadIdx.x], s_m[threadIdx.x + o]);
        __syncthreads();
    }
    const float mglobal = s_m[0];
    __syncthreads();

    float lsum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float p = __expf(s_score[j] - mglobal);
        s_score[j] = p;
        lsum += p;
    }
    s_m[threadIdx.x] = lsum;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) s_m[threadIdx.x] += s_m[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) { P[0] = mglobal; P[1] = s_m[0]; }

    // NOTE: batching the V-scan loads was tried twice (8 and 32 tokens per
    // thread) and is WORSE both times -- 334 -> 302 tok/s at 8k context.  The
    // 64 threads do read each line cooperatively, so coalescing was never the
    // problem, and the extra live registers cost more than the added overlap
    // buys.  The score loop above is a different case: there each lane is on
    // its OWN line, which is why 8 x uint4 helped and 64 x 2-byte loads did not.
    for (int d = threadIdx.x; d < LK_HEAD_DIM; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < n; ++t) {
            const __half * v = vc + ((size_t)(t0 + t) * LK_N_KV_HEADS + kv_head) * LK_HEAD_DIM;
            acc += s_score[t] * __half2float(v[d]);
        }
        P[2 + d] = acc;                   // unnormalised partial
    }
}

__global__ void k_attn_combine(const float * __restrict__ part, float * __restrict__ out,
                               block_q8_1 * __restrict__ yq8) {
    const int head = blockIdx.x;
    const int d = threadIdx.x;            // 64 threads, one per head dim
    const float * P = part + (size_t) head * LK_ATTN_SPLIT * LK_ATTN_PARTW;
    __shared__ float s_m[LK_ATTN_SPLIT];
    __shared__ float s_l[LK_ATTN_SPLIT];
    if (d < LK_ATTN_SPLIT) {
        s_m[d] = P[d * LK_ATTN_PARTW + 0];
        s_l[d] = P[d * LK_ATTN_PARTW + 1];
    }
    __syncthreads();
    float mf = -1e30f;
    for (int s = 0; s < LK_ATTN_SPLIT; ++s) mf = fmaxf(mf, s_m[s]);
    float lf = 0.0f, acc = 0.0f;
    for (int s = 0; s < LK_ATTN_SPLIT; ++s) {
        const float w = __expf(s_m[s] - mf);
        lf  += s_l[s] * w;
        acc += P[s * LK_ATTN_PARTW + 2 + d] * w;
    }
    // DIAGNOSTIC: reproduce the serial kernel exactly by using
    // acc * (1/l) rather than acc / l.
    const float o = acc * (1.0f / lf);
    if (out) out[head * LK_HEAD_DIM + d] = o;
    if (yq8) quantize_q8_1_warp(o, yq8 + 2 * head + (d >> 5), d & 31);
}
