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

// Fused RMSNorm + Q8_1 quantiser, BIT-IDENTICAL to running k_rmsnorm then
// k_quantize_q8_1.  The norm half keeps the original 256-thread loop order and
// the original shared-memory tree, so y32/y16 are bit-for-bit what they were;
// the normalised values are then re-read from shared memory by a
// warp-cooperative quantiser.  Saves one of the ~120 launches per token that
// this pair costs, without reducing parallelism (the earlier attempt dropped to
// 256 threads for both halves and lost 10%).
__global__ void k_rmsnorm_q8_1(const float * __restrict__ x, const float * __restrict__ w,
                               int n, float eps, float * __restrict__ y32,
                               __half * __restrict__ y16, block_q8_1 * __restrict__ yq8);

// Quantise n floats to Q8_1, one warp per 32-element block.  Kept separate from
// k_rmsnorm so the quantiser is a single, independently tested primitive.
__global__ void k_quantize_q8_1(const float * __restrict__ x,
                                block_q8_1 * __restrict__ y, int nblocks);

// Attention scratch capacity (scores live in shared memory in Phase 3).  The
// host asserts seq_len <= this; a proper Split-K online softmax lands in
// Phase 4 to lift the limit to the full 131072 context.
#define LK_ATTN_MAX_SEQ 8192
// k_attention_decode runs ONE block per query head, and every one of its loops
// is limited by outstanding memory requests rather than by bandwidth or issue
// slots: the score loop gives each thread all 64 dims of a single token, so a
// warp holds exactly one cache line in flight and the block stalls near
// 2.4 GB/s per SM -- which is the 3.7 GB/s the profile actually showed at 8k
// context.  Outstanding requests scale with the WARP count, so the block is
// widened well beyond the 64 lanes the output loop needs.

// Rows computed per warp by the general Q4_0 GEMV.  Four rows in flight gives
// 4x the memory-level parallelism per lane and amortises each shared xq block
// read across four weight rows.
#define LK_GEMV_ROWS 1
// blocks of 256 threads (8 warps) needed to cover `rows` output rows
#define LK_GEMV_BLOCKS(rows) ((((rows) / LK_GEMV_ROWS) + 7) / 8)

// Generic Q4_0 x Q8_1 GEMV: LK_GEMV_ROWS output rows per warp.
__global__ void k_gemv_q4_0(const uint4 * __restrict__ qs, const __half * __restrict__ d,
                            const block_q8_1 * __restrict__ xq,
                            int out_rows, int bpr, float * __restrict__ y, int add_residual);

// Generic Q6_K x Q8_1 GEMV: one warp per output row; each super-block consumes
// 8 consecutive Q8_1 blocks.
__global__ void k_gemv_q6_K(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                            const int8_t * __restrict__ sc, const __half * __restrict__ d,
                            const block_q8_1 * __restrict__ xq,
                            int out_rows, int sbpr, float * __restrict__ y);
__global__ void k_gemv_q6_K_dp4a(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                                 const int8_t * __restrict__ sc, const __half * __restrict__ d,
                                 const block_q8_1 * __restrict__ xq,
                                 int out_rows, int sbpr, float * __restrict__ y);

// SwiGLU: out[i] = silu(gate[i]) * up[i]
__global__ void k_swiglu(const float * __restrict__ gate, const float * __restrict__ up,
                         int n, float * __restrict__ out, block_q8_1 * __restrict__ yq8);

// ShortConv one-token step.  semantics (verified against llama.cpp lfm2.cpp):
//   [B|C|x] = in_proj(h);  bx = B*x;
//   conv[t] = hist0*w[0] + hist1*w[1] + bx*w[2];  y = C * conv[t];  out = out_proj(y)
// w is [3, 2048] F32 with the tap axis FASTEST: w[tap + ch*3].
// hist is [2048][2] per layer: hist[ch*2+0] = older, hist[ch*2+1] = newer.
__global__ void k_shortconv(const float * __restrict__ bcx, int t,
                            const float * __restrict__ conv_w, float * __restrict__ hist,
                            float * __restrict__ y, block_q8_1 * __restrict__ yq8);

// MoE router: sigmoid gate, top-4 selected by (sigmoid + bias), weights taken
// from the UNBIASED sigmoid and renormalised.  Exactly llama.cpp's
// LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID path.
__global__ void __launch_bounds__(32) k_router_dot(
        const float * __restrict__ h, const float * __restrict__ gate_inp,
        float * __restrict__ logits);
__global__ void k_router_top4(const float * __restrict__ logits,
                              const float * __restrict__ bias,
                              int * __restrict__ idx_out, float * __restrict__ w_out);

// Single-query decode attention over an fp16 KV cache.
// The block width was tried at 1024 (with launch bounds) and measured NEUTRAL:
// attention at 8k context was 466 us per layer against 476 at 256 threads, and
// short context got slightly worse.  Widening the block does not help because
// the two hot loops want opposite things -- the score loop is coalesced at
// token-major and the V loop needs a token split -- so the remaining lever is
// split-K, not more warps.  Kept at 256 so the reduction order is unchanged.
#define LK_ATTN_BLOCK   256
__global__ void k_attention_decode(const float * __restrict__ q,     // [32*64]
                                   const __half * __restrict__ kc,   // [max_seq, 8, 64]
                                   const __half * __restrict__ vc,   // [max_seq, 8, 64]
                                   const int * __restrict__ d_pos, float * __restrict__ out,
                                   block_q8_1 * __restrict__ yq8);

// ---- additional forward kernels (Phase 3) ---------------------------------
// Q6_K embedding-row lookup: dequantise row `token` into x[2048].
__global__ void k_embed_q6_K(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                             const int8_t * __restrict__ sc, const __half * __restrict__ d,
                             const int * __restrict__ d_token, float * __restrict__ x);

// Per-head Q/K RMSNorm(64) followed by NEOX RoPE, then write K and V into the
// fp16 KV cache at slot `pos_idx`.  grid = (32 Q heads + 8 KV heads),
// block = 64 threads.  Q is normalised+rotated in place; K likewise and then
// cached; V is cached unnormalised and unrotated (it gets neither).
__global__ void k_qk_norm_rope(float * __restrict__ qf, float * __restrict__ kf,
                               const float * __restrict__ vf,
                               const float * __restrict__ qn, const float * __restrict__ kn,
                               const int * __restrict__ d_pos,
                               __half * __restrict__ kc, __half * __restrict__ vc);

// dst[i] += w * src[i]
__global__ void k_accum_scaled(float * __restrict__ dst, const float * __restrict__ src,
                               float w, int n);

// Writes the argmax into d_token and bumps d_pos, so a captured graph is
// position-independent and the whole decode loop can stay on device.
#define LK_ARGMAX_BLOCKS      125
#define LK_ARGMAX_PER_THREAD  4
__global__ void k_route_log(const int * __restrict__ idx, const int * __restrict__ d_pos,
                            int il, int * __restrict__ log);

__global__ void k_argmax_part(const float * __restrict__ v,
                              float * __restrict__ pv, int * __restrict__ pi);
__global__ void k_argmax_final(const float * __restrict__ pv, const int * __restrict__ pi,
                               int n, int * __restrict__ d_token, int * __restrict__ d_pos);

// ---- Phase 4: expert-batched MoE -----------------------------------------
// One launch serves all 4 selected experts (grid.y = expert slot), so the MoE
// layer drops from 16 launches to 3. Expert k's weights start at row
// (d_idx[k] * rows_per_expert) and its output lands at y + k*out_rows.
__global__ void k_gemv_q4_0_exp2(const uint4 * __restrict__ qs_a, const __half * __restrict__ d_a,
                                 const uint4 * __restrict__ qs_b, const __half * __restrict__ d_b,
                                 const block_q8_1 * __restrict__ xq,
                                 int out_rows, int bpr, int rows_per_expert,
                                 int xq_stride,
                                 const int * __restrict__ d_idx,
                                 float * __restrict__ y_a, float * __restrict__ y_b);
__global__ void k_gemv_q4_0_exp(const uint4 * __restrict__ qs, const __half * __restrict__ d,
                                const block_q8_1 * __restrict__ xq,
                                int out_rows, int bpr, int rows_per_expert,
                                int xq_stride,   // 0 = all experts share the same
                                                 // activation; bpr = per-expert
                                const int * __restrict__ d_idx,
                                float * __restrict__ y);

// dst[i] += sum_k w[k] * down4[k*n + i]  -- keeps the router weights on device,
// which removes the per-MoE-layer cudaStreamSynchronize host round-trip.
__global__ void k_moe_accum4(float * __restrict__ dst, const float * __restrict__ down4,
                             const float * __restrict__ d_w, int n);
