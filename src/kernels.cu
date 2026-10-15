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
    if (warp >= out_rows) return;
    const uint4 * qrow = qs + (size_t)warp * bpr;
    const __half * drow = d + (size_t)warp * bpr;
    float acc = 0.0f;
    for (int k = lane; k < bpr; k += 32)
        acc += dot_q4_0_q8_1(qrow, drow, k, xq + k);
    acc = warp_sum(acc);
    if (lane == 0) y[warp] = add_residual ? (y[warp] + acc) : acc;
}

// ---------------------------------------------------------------------------
__global__ void k_gemv_q6_K(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                            const int8_t * __restrict__ sc, const __half * __restrict__ d,
                            const block_q8_1 * __restrict__ xq,
                            int out_rows, int sbpr, float * __restrict__ y) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= out_rows) return;
    float acc = 0.0f;
    for (int sb = 0; sb < sbpr; ++sb) {          // 8 super-blocks per 2048-dim row
        const size_t blk = (size_t)warp * sbpr + sb;
        acc += dot_q6_K_q8_1_warp(ql + blk * 128, qh + blk * 64, sc + blk * 16,
                                  __half2float(d[blk]), xq + (size_t)sb * 8, lane);
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
__global__ void k_router_top4(const float * __restrict__ h, const float * __restrict__ gate_inp,
                              const float * __restrict__ bias,
                              int * __restrict__ idx_out, float * __restrict__ w_out) {
    __shared__ float s_logit[LK_N_EXPERTS];
    __shared__ float s_sel[LK_N_EXPERTS];
    for (int j = threadIdx.x; j < LK_N_EXPERTS; j += blockDim.x) {
        float acc = 0.0f;
        const float * row = gate_inp + (size_t)j * LK_HIDDEN;
        for (int i = 0; i < LK_HIDDEN; ++i) acc += row[i] * h[i];
        s_logit[j] = acc;
        s_sel[j] = 1.0f / (1.0f + __expf(-acc)) + bias[j];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float prob[LK_TOP_K];
        int idx[LK_TOP_K];
        for (int k = 0; k < LK_TOP_K; ++k) { idx[k] = -1; prob[k] = -1e30f; }
        for (int j = 0; j < LK_N_EXPERTS; ++j) {
            for (int k = 0; k < LK_TOP_K; ++k) {
                if (s_sel[j] > prob[k]) {
                    for (int m = LK_TOP_K - 1; m > k; --m) { prob[m] = prob[m-1]; idx[m] = idx[m-1]; }
                    prob[k] = s_sel[j]; idx[k] = j;
                    break;
                }
            }
        }
        // weights come from the UNBIASED sigmoid
        float wsum = 0.0f;
        float wv[LK_TOP_K];
        for (int k = 0; k < LK_TOP_K; ++k) {
            wv[k] = 1.0f / (1.0f + __expf(-s_logit[idx[k]]));
            wsum += wv[k];
        }
        if (wsum < LK_TOPK_GUARD) wsum = LK_TOPK_GUARD;
        for (int k = 0; k < LK_TOP_K; ++k) { idx_out[k] = idx[k]; w_out[k] = wv[k] / wsum; }
    }
}

// ---------------------------------------------------------------------------
__global__ void k_attention_decode(const float * __restrict__ q,
                                   const __half * __restrict__ kc,
                                   const __half * __restrict__ vc,
                                   int seq_len, float * __restrict__ out) {
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
                             int token, float * __restrict__ x) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= LK_HIDDEN) return;
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
                               float pos, __half * __restrict__ kc, __half * __restrict__ vc,
                               int pos_idx) {
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
        const size_t off = ((size_t)pos_idx * LK_N_KV_HEADS + head) * LK_HEAD_DIM;
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
__global__ void k_argmax(const float * __restrict__ v, int n, int * __restrict__ out) {
    __shared__ float s_v[256];
    __shared__ int   s_i[256];
    float best = -1e30f; int bi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (v[i] > best) { best = v[i]; bi = i; }
    s_v[threadIdx.x] = best; s_i[threadIdx.x] = bi;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int)threadIdx.x < o && s_v[threadIdx.x + o] > s_v[threadIdx.x]) {
            s_v[threadIdx.x] = s_v[threadIdx.x + o];
            s_i[threadIdx.x] = s_i[threadIdx.x + o];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = s_i[0];
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
