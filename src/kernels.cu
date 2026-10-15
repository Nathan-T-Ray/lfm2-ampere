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
    extern __shared__ float s_h[];          // n floats, normalised values
    const int tid = threadIdx.x;

    // ---- phase 1: EXACTLY the original k_rmsnorm order (256 threads, tree) --
    // The summation order is untouched; only the loads are batched.  With one
    // load in flight per thread a 256-thread block has 8 warps x 128 B = 1 KB
    // outstanding, which at ~500 cycles of HBM latency is ~2.8 GB/s.  These
    // norms run 81 times per token and measured ~5 us each (3.5 GB/s); issuing
    // all eight loads before accumulating fills the memory pipeline without
    // changing a single floating-point operation.
    float sq = 0.0f;
    if (n == 2048) {
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = x[tid + j * 256];
#pragma unroll
        for (int j = 0; j < 8; ++j) sq += v[j] * v[j];
    } else {
        for (int i = tid; i < n; i += 256) { const float v = x[i]; sq += v * v; }
    }
    s_sq[tid] = sq;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if ((int)tid < o) s_sq[tid] += s_sq[tid + o];
        __syncthreads();
    }
    const float inv = rsqrtf(s_sq[0] / (float)n + eps);
    if (n == 2048) {                     // same arithmetic, both loads batched
        float v[8], wv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { v[j] = x[tid + j * 256]; wv[j] = w[tid + j * 256]; }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int i = tid + j * 256;
            const float t = v[j] * inv * wv[j];
            if (y32) y32[i] = t;
            if (y16) y16[i] = __float2half(t);
            s_h[i] = t;
        }
    } else {
        for (int i = tid; i < n; i += 256) {
            const float v = x[i] * inv * w[i];
            if (y32) y32[i] = v;
            if (y16) y16[i] = __float2half(v);
            s_h[i] = v;
        }
    }
    __syncthreads();

    // ---- phase 2: warp-cooperative Q8_1 over consecutive 32-element blocks --
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
__global__ void k_swiglu(const float * __restrict__ gate, const float * __restrict__ up,
                         int n, float * __restrict__ out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        out[i] = silu_f(gate[i]) * up[i];
}

// ---------------------------------------------------------------------------
__global__ void k_shortconv(const float * __restrict__ bcx, int t,
                            const float * __restrict__ conv_w, float * __restrict__ hist,
                            float * __restrict__ y) {
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
        y[ch] = C[ch] * conv;
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
#define LK_ROUTER_UNROLL 8
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
    // Split the sigmoids across lanes.  Thread 0 computing all 64 of them made
    // this kernel cost 15.3 us: a single active lane issues one instruction per
    // cycle and stalls for the full latency of every dependent step inside
    // expf, so 64 serial ~200-cycle chains is ~10 us of pure stall.  The values
    // produced are identical (same expf, same inputs), so the ranking below is
    // bit-for-bit the same decision.
    if (t < LK_N_EXPERTS) {
        const float g = 1.0f / (1.0f + expf(-logits[t]));
        s_sig[t] = g;
        s_sel[t] = g + bias[t];
    }
    __syncthreads();
    if (t != 0) return;
    const float * sel = s_sel;
    // Same insertion as the original nested loop, unrolled by hand.  With a
    // variable inner trip count nvcc spills prob[]/idx[] to local memory and
    // every shift becomes a dependent local load/store pair; in a single active
    // lane that measured ~5 us per layer.  The explicit registers make the
    // whole selection a handful of cycles.
    static_assert(LK_TOP_K == 4, "unrolled selection assumes top-4");
    float p0 = -1e30f, p1 = -1e30f, p2 = -1e30f, p3 = -1e30f;
    int   i0 = -1,     i1 = -1,     i2 = -1,     i3 = -1;
    for (int j = 0; j < LK_N_EXPERTS; ++j) {
        const float v = sel[j];
        if      (v > p0) { p3 = p2; i3 = i2; p2 = p1; i2 = i1; p1 = p0; i1 = i0; p0 = v; i0 = j; }
        else if (v > p1) { p3 = p2; i3 = i2; p2 = p1; i2 = i1; p1 = v;  i1 = j; }
        else if (v > p2) { p3 = p2; i3 = i2; p2 = v;  i2 = j; }
        else if (v > p3) { p3 = v;  i3 = j; }
    }
    const int idx[LK_TOP_K] = { i0, i1, i2, i3 };
    float wsum = 0.0f;
    float wv[LK_TOP_K];
    for (int k = 0; k < LK_TOP_K; ++k) {
        wv[k] = s_sig[idx[k]];      // unbiased sigmoid, already computed above
        wsum += wv[k];
    }
    if (wsum < LK_TOPK_GUARD) wsum = LK_TOPK_GUARD;
    for (int k = 0; k < LK_TOP_K; ++k) { idx_out[k] = idx[k]; w_out[k] = wv[k] / wsum; }
}

// ---------------------------------------------------------------------------
__global__ void k_attention_decode(const float * __restrict__ q,
                                   const __half * __restrict__ kc,
                                   const __half * __restrict__ vc,
                                   const int * __restrict__ d_pos, float * __restrict__ out) {
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
        for (int d = 0; d < LK_HEAD_DIM; ++d) acc += s_q[d] * __half2float(k[d]);
        acc *= LK_ATTN_SCALE;
        s_score[t] = acc;
        m = fmaxf(m, acc);
    }
    // block max
    __shared__ float s_m[256];
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
        out[head * LK_HEAD_DIM + d] = acc * inv_l;
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
