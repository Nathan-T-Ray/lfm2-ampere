// kernels.cuh - forward-pass kernels for liquid-kernel (Phase 3: correctness
// first, straightforward implementations; Phase 4 optimises to the roofline).
#pragma once
#include "quant.cuh"

// Planar weight views, matching tools/repack.py's plane geometry.
struct QT4 { const uint4 * qs; const __half * d; };                       // Q4_0
struct QT6 { const uint8_t * ql; const uint8_t * qh;
             const int8_t * scales; const __half * d; };                  // Q6_K

// Architecture constants (from the GGUF metadata / llama.cpp lfm2.cpp)
#define LK_HIDDEN      2048
#define LK_N_LAYERS    40
#define LK_N_HEADS     32
#define LK_HEAD_DIM    64
#define LK_N_KV_HEADS  8
#define LK_N_EXPERTS   64
#define LK_TOP_K       4
#define LK_MOE_INTER   1536
#define LK_DENSE_INTER 11776
#define LK_VOCAB       128000
#define LK_EPS         1e-5f
#define LK_ROPE_THETA  10000000.0f
#define LK_ATTN_SCALE  0.125f          // 1/sqrt(64), exact
#define LK_TOPK_GUARD  6.103515625e-5f // llama.cpp's clamp on the top-k weight sum

// ---- device helpers -------------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float silu_f(float v) { return v / (1.0f + __expf(-v)); }

// NEOX (half-split) RoPE, head_dim 64, theta 1e7.  Verified from llama.cpp:
// LLAMA_ROPE_TYPE_NEOX -> rotate_pairs(n_dims, n_dims/2) gives
//   out[i]      = x[i]*cos - x[i+32]*sin
//   out[i+32]   = x[i]*sin + x[i+32]*cos
// i.e. dims d and d+32 rotate together, NOT adjacent pairs.
__device__ __forceinline__ void rope_neox_head(float * h, float pos) {
#pragma unroll
    for (int i = 0; i < LK_HEAD_DIM / 2; ++i) {
        const float inv = powf(LK_ROPE_THETA, -2.0f * (float)i / (float)LK_HEAD_DIM);
        const float ang = pos * inv;
        float c, s;
        // NB: precise sincosf, NOT __sincosf.  The fast intrinsic has poor
        // argument reduction and at pos*theta ~ 1e4 it drifts ~1e-3, which is
        // enough to move logits.  llama.cpp's rope also uses precise sincosf.
        sincosf(ang, &s, &c);
        const float x0 = h[i], x1 = h[i + LK_HEAD_DIM / 2];
        h[i]                    = x0 * c - x1 * s;
        h[i + LK_HEAD_DIM / 2]  = x0 * s + x1 * c;
    }
}

// ---- kernels --------------------------------------------------------------
// y = x * rsqrt(mean(x^2) + eps) * w.  Optionally also emit an fp16 copy (the
// MoE router consumes the POST-ffn_norm hidden state) and Q8_1 blocks for the
// following GEMV.
__global__ void k_rmsnorm(const float * __restrict__ x, const float * __restrict__ w,
                          int n, float eps,
                          float * __restrict__ y32, __half * __restrict__ y16);

// Quantise n floats to Q8_1, one warp per 32-element block.  Kept separate from
// k_rmsnorm so the quantiser is a single, independently tested primitive.
__global__ void k_quantize_q8_1(const float * __restrict__ x,
                                block_q8_1 * __restrict__ y, int nblocks);

// Attention scratch capacity (scores live in shared memory in Phase 3).  The
// host asserts seq_len <= this; a proper Split-K online softmax lands in
// Phase 4 to lift the limit to the full 131072 context.
#define LK_ATTN_MAX_SEQ 8192

// Generic Q4_0 x Q8_1 GEMV: one warp per output row.
__global__ void k_gemv_q4_0(const uint4 * __restrict__ qs, const __half * __restrict__ d,
                            const block_q8_1 * __restrict__ xq,
                            int out_rows, int bpr, float * __restrict__ y, int add_residual);

// Generic Q6_K x Q8_1 GEMV: one warp per output row; each super-block consumes
// 8 consecutive Q8_1 blocks.
__global__ void k_gemv_q6_K(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                            const int8_t * __restrict__ sc, const __half * __restrict__ d,
                            const block_q8_1 * __restrict__ xq,
                            int out_rows, int sbpr, float * __restrict__ y);

// SwiGLU: out[i] = silu(gate[i]) * up[i]
__global__ void k_swiglu(const float * __restrict__ gate, const float * __restrict__ up,
                         int n, float * __restrict__ out);

// ShortConv one-token step.  semantics (verified against llama.cpp lfm2.cpp):
//   [B|C|x] = in_proj(h);  bx = B*x;
//   conv[t] = hist0*w[0] + hist1*w[1] + bx*w[2];  y = C * conv[t];  out = out_proj(y)
// w is [3, 2048] F32 with the tap axis FASTEST: w[tap + ch*3].
// hist is [2048][2] per layer: hist[ch*2+0] = older, hist[ch*2+1] = newer.
__global__ void k_shortconv(const float * __restrict__ bcx, int t,
                            const float * __restrict__ conv_w, float * __restrict__ hist,
                            float * __restrict__ y);

// MoE router: sigmoid gate, top-4 selected by (sigmoid + bias), weights taken
// from the UNBIASED sigmoid and renormalised.  Exactly llama.cpp's
// LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID path.
__global__ void k_router_top4(const float * __restrict__ h, const float * __restrict__ gate_inp,
                              const float * __restrict__ bias,
                              int * __restrict__ idx_out, float * __restrict__ w_out);

// Single-query decode attention over an fp16 KV cache.
__global__ void k_attention_decode(const float * __restrict__ q,     // [32*64]
                                   const __half * __restrict__ kc,   // [max_seq, 8, 64]
                                   const __half * __restrict__ vc,   // [max_seq, 8, 64]
                                   int seq_len, float * __restrict__ out);

// ---- additional forward kernels (Phase 3) ---------------------------------
// Q6_K embedding-row lookup: dequantise row `token` into x[2048].
__global__ void k_embed_q6_K(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                             const int8_t * __restrict__ sc, const __half * __restrict__ d,
                             int token, float * __restrict__ x);

// Per-head Q/K RMSNorm(64) followed by NEOX RoPE, then write K and V into the
// fp16 KV cache at slot `pos_idx`.  grid = (32 Q heads + 8 KV heads),
// block = 64 threads.  Q is normalised+rotated in place; K likewise and then
// cached; V is cached unnormalised and unrotated (it gets neither).
__global__ void k_qk_norm_rope(float * __restrict__ qf, float * __restrict__ kf,
                               const float * __restrict__ vf,
                               const float * __restrict__ qn, const float * __restrict__ kn,
                               float pos, __half * __restrict__ kc, __half * __restrict__ vc,
                               int pos_idx);

// dst[i] += w * src[i]
__global__ void k_accum_scaled(float * __restrict__ dst, const float * __restrict__ src,
                               float w, int n);

__global__ void k_argmax(const float * __restrict__ v, int n, int * __restrict__ out);
