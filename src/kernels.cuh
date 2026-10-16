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

// ---------------------------------------------------------------------------
// Phase 4y: an alternative launch SHAPE for the three GEMVs that produce the
// residual stream (shortconv_out, attn_out, dense_down).
//
// A Q8_1 activation block is 32 CONSECUTIVE channels.  A warp owns one output
// row, and at the default shape a block owns 8 consecutive rows, so a Q8_1
// block spans FOUR blocks -- which means those GEMVs cannot quantise their own
// output without a cross-block barrier.  With 1024 threads a block owns
// 32 consecutive rows, which is exactly one Q8_1 block.
//
// The warp->row map is unchanged (warp = global warp id, row0 = warp*ROWS), so
// the arithmetic is bit-identical; only blockDim and the grid change.  The same
// 2048 warps result, but as 64 blocks of 1024 instead of 256 blocks of 256:
// 64 SMs carry one block each instead of 256 blocks spreading 3 per SM.
//
// This is a FALSIFICATION TEST and defaults off.  LK_GEMV_ROWS=4 also yields 64
// blocks for a 2048-row GEMV and measured WORSE, twice -- but it differs in
// warps per block (8 vs 32), so it does not settle this.  Only a measurement
// does, and it must come before any fusion is built on the shape.
#ifndef LK_GEMV_BIGBLOCK
#define LK_GEMV_BIGBLOCK 0
#endif
#if LK_GEMV_BIGBLOCK
#define LK_GEMV_WBLOCKS(rows) ((rows) / 32)
#define LK_GEMV_WTHREADS 1024
#else
#define LK_GEMV_WBLOCKS(rows) LK_GEMV_BLOCKS(rows)
#define LK_GEMV_WTHREADS 256
#endif

// Phase 4y, in two independent halves so they can be measured separately.
//
// ACCUM folds attn_norm(L) for L >= 3 into k_moe_accum4(L-1).  Free: that
// kernel's warp w already owns exactly Q8_1 block w, so the epilogue is pure
// warp arithmetic with no barrier and no counter.
//
// GEMV folds the remaining 42 sites (the 40 ffn_norms, whose producer is
// shortconv_out/attn_out, and attn_norm for L = 1,2, whose producer is
// dense_down).  All three of those are 2048-row GEMVs, so a block owns only 8
// rows at the default shape and a Q8_1 block spans four blocks -- this half
// requires LK_GEMV_BIGBLOCK.
#ifndef LK_NORM_PROD_ACCUM
#define LK_NORM_PROD_ACCUM 0
#endif
#ifndef LK_NORM_PROD_GEMV
#define LK_NORM_PROD_GEMV 0
#endif
#if LK_NORM_PROD_GEMV && !LK_GEMV_BIGBLOCK
#error "LK_NORM_PROD_GEMV needs LK_GEMV_BIGBLOCK: a block must own a whole Q8_1 block"
#endif
// Speculative L2 prefetch.
//
// ~20% of the token has the memory subsystem idle: 0.268 ms of launch gaps
// (nothing running, so nothing can issue a load) and 0.221 ms of small kernels
// executing with memory going unused.  The second half IS fillable -- a kernel
// is running and can issue loads nobody consumes -- and the prize is bounded by
// the window, not by the traffic: filling an idle window of T seconds saves at
// most T, whatever is fetched during it.
//
// This changes NO arithmetic.  Nothing reads the prefetched lines here; it only
// asks the memory system to start the transfer early so the next kernel finds
// them in L2 instead of in DRAM.  tools/ab.py must therefore stay 7/7, which
// makes this the cheapest remaining idea to prove correct.
// Two mechanisms, because they have different priority in the memory system.
//   prefetch.global.L2 is a HINT.  It costs no register and blocks nothing,
//     but the memory controller is free to deprioritise it below demand loads,
//     which would mean it never actually uses the idle window.
//   ld.global is a DEMAND access.  It gets normal priority; the thread issues
//     it and never reads the destination register, so no scoreboard stall.
#ifndef LK_PF_LOAD
#define LK_PF_LOAD 0
#endif
__device__ __forceinline__ void lk_prefetch_l2(const void * __restrict__ p) {
#if LK_PF_LOAD
#pragma nv_diag_suppress 550
    unsigned int dummy;
    asm volatile("ld.global.b32 %0, [%1];" : "=r"(dummy) : "l"(p) : "memory");
    (void)dummy;   // never read: no scoreboard stall, but see the note above
#else
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
#endif
}

// LK_ABLATE_SMALL: empty the NON-streaming kernels, keeping their grids and
// block sizes, to separate their real EXECUTION from their launch cost (which
// tests/bench_shapes.cu measures separately at 0.268 ms for all 343 nodes).
//
// Value selects a group; 5 is all of them.
//   1 router_dot + router_top4      (76 nodes)
//   2 swiglu                        (38 nodes)
//   3 shortconv (the 3-tap conv)    (30 nodes)
//   4 the rest                      (14 nodes: qk_norm_rope, embed, argmax,
//                                    the one remaining rmsnorm)
//   5 all four groups
//
// Garbage propagates downstream, which is fine: the surviving kernels still
// read the same bytes from the same places, so timing stays valid even though
// the numbers do not.  Measurement only -- never ship this on.
#ifndef LK_ABLATE_SMALL
#define LK_ABLATE_SMALL 0
#endif
#define LK_ABL_PICK(grp) ((LK_ABLATE_SMALL == (grp) || LK_ABLATE_SMALL == 5) ? 1 : 0)
#if LK_ABLATE_SMALL == 1 || LK_ABLATE_SMALL == 5
#define LK_ABL_ROUTER return
#else
#define LK_ABL_ROUTER ((void)0)
#endif
#if LK_ABLATE_SMALL == 2 || LK_ABLATE_SMALL == 5
#define LK_ABL_SWIGLU return
#else
#define LK_ABL_SWIGLU ((void)0)
#endif
#if LK_ABLATE_SMALL == 3 || LK_ABLATE_SMALL == 5
#define LK_ABL_SHORTCONV return
#else
#define LK_ABL_SHORTCONV ((void)0)
#endif
#if LK_ABLATE_SMALL == 4 || LK_ABLATE_SMALL == 5
#define LK_ABL_REST return
#else
#define LK_ABL_REST ((void)0)
#endif

// T4r: rebuild the top-4 in the consumer's prologue instead of in its own
// <<<1,64>>> node.  MEASURED AND REJECTED -- 0.17% SLOWER.  The selection work
// is barrier-free in the Phase 4y sense, but hosting it in k_gemv_q4_0_exp2
// puts a __syncthreads in front of the weight stream of the single hottest
// kernel in the engine (1536 blocks), and that stall costs more than the
// 1.77 us/node the <<<1,64>>> node was measured at in isolation.
//
// This is the SECOND router fusion to lose.  Both lost for the same reason:
// the router's own node is tiny, so any fusion that perturbs a large kernel to
// absorb it pays more than it saves.  Kept so the negative result is
// reproducible.
#ifndef LK_ROUTER_PRO
#define LK_ROUTER_PRO 0
#endif

// LK_FUSE_ROUTER (Phase 4x Step 3) is RETIRED, not merely off.  It folded the
// top-4 into the last block of k_router_dot behind a threadfence+atomicAdd
// barrier, measured 0.09-0.39% slower, and has since gone stale: it never
// learned about Phase 4y's inv_rms multiply, so enabling it today would route
// on unscaled logits.  T4r supersedes it with no barrier at all.
#if LK_FUSE_ROUTER
#error "LK_FUSE_ROUTER is retired: it is stale (no Phase 4y inv) and was measured slower. See DIAGNOSIS.md."
#endif

// attn_norm(L) is produced by layer L-1's channel mixer.  Which kernel that is
// depends on whether layer L-1 is dense or MoE, and the two producers sit in
// different halves of the fusion:
//   L = 0        embedding        never fused
//   L = 1, 2     dense_down(L-1)  GEMV half  (layers 0 and 1 are dense)
//   L >= 3       moe_accum4(L-1)  ACCUM half (layers 2..39 are MoE)
//   L = 40       moe_accum4(39)   ACCUM half, this is norm_final
// The two sets are disjoint, and each producer site asks for its own half, so
// the split cannot silently disagree with itself.
#if LK_NORM_PROD_ACCUM
#define LK_FUSE_ATTN_NORM_A(il) ((il) >= 3)
#else
#define LK_FUSE_ATTN_NORM_A(il) 0
#endif
#if LK_NORM_PROD_GEMV
#define LK_FUSE_ATTN_NORM_G(il) ((il) == 1 || (il) == 2)
#else
#define LK_FUSE_ATTN_NORM_G(il) 0
#endif
// What a CONSUMER of attn_norm(il) must believe.
#define LK_FUSE_ATTN_NORM(il) (LK_FUSE_ATTN_NORM_A(il) || LK_FUSE_ATTN_NORM_G(il))

// ffn_norm(L) is produced by this layer's own mixer output projection, for
// every layer.
#if LK_NORM_PROD_GEMV
#define LK_FUSE_FFN_NORM(l) 1
#else
#define LK_FUSE_FFN_NORM(l) 0
#endif

// Generic Q4_0 x Q8_1 GEMV: LK_GEMV_ROWS output rows per warp.
//
// Phase 4x: an optional RMSNorm epilogue.  Every FFN norm in a token is
// preceded by one of three GEMVs (shortconv_out, attn_out, dense_down) that
// produces exactly the 2048 residual values the norm then reduces.  Running
// the norm as its own kernel costs 2.66 us of which 0.61 is dispatch, and
// bench_launch.cu measures the marginal cost of such a node behind a busy
// stream at 3.81 us -- 80 times a token.  So the norm moves into the producer:
// every block contributes its rows' sum of squares, and the LAST block to
// finish (a threadfence + atomicAdd, NOT a grid sync) reduces them in block
// order, applies the norm and emits Q8_1 and optionally f32.
//
// The reduction is deterministic -- partials are summed in strict block index
// order -- but it is NOT bit-identical to k_rmsnorm_q8_1, whose 2048 elements
// are partitioned across threads differently.  C2 adjudicates that; C1 cannot.
#define LK_NORM_PARTIALS 256      // == LK_GEMV_BLOCKS(LK_HIDDEN); see static_assert
// Phase 4x Step 1: run the FFN norm inside the projection that produces its
// input, for the 40 shortconv_out/attn_out sites.
//
// MEASURED AND REJECTED -- default 0.  Interleaved A/B, 3 rounds each:
//
//   ctx      fuse0    fuse1
//    272     417.02   410.39   -1.6%
//   2128     406.71   399.25   -1.8%
//   8128     367.26   361.68   -1.5%
//
// The 40 norm nodes did disappear (norm_ffn 0.185 -> 0.054 ms) but each
// producer absorbed +3.2 us (shortconv_out 0.186 -> 0.282, attn_out 0.063 ->
// 0.094), which is the 3.3 us node it replaced.  The norm's cost is NOT launch
// overhead: bench_launch.cu measures a 1-block node at 0.605 us empty and
// 2.661 us with a real 2048-element reduction, and 3.814 us when it sits
// behind a busy stream.  The extra 1.15 us is the stall waiting for the
// preceding kernel's last wave to drain, and that stall is a property of the
// DEPENDENCY, not of the kernel -- folding the work into the producer inherits
// the identical stall and adds a threadfence plus a 256-way atomic to it.
//
// Kept as a switch so the negative result stays reproducible, and because
// tests/test_normfuse.cu guards the epilogue if it is ever revisited.
#ifndef LK_FUSE_NORM
#define LK_FUSE_NORM 0
#endif
// Phase 4x Step 3: fold k_router_top4 into k_router_dot.
//
// MEASURED AND REJECTED -- default 0.  This was the FAIR test of the
// last-block mechanism: the selection is ~30 shuffle ops against a ~1.9 us
// single-block node, so if folding pays anywhere it pays here.  It does not.
// Interleaved A/B, REPS=3:
//
//   ctx      fuse0    fuse1
//    272     416.16   415.80   -0.09%  (within noise)
//   2128     406.73   405.19   -0.38%
//   8128     367.11   365.69   -0.39%
//
// The output is BIT-IDENTICAL across all 7 suite cases, so this is purely a
// cost result.  The 38 router_top4 nodes do vanish, worth 38 x 0.605 = 0.023 ms
// of dispatch, but k_router_dot's real time goes 0.120 -> 0.155 ms because
// every one of its 64 blocks now pays a __threadfence plus a contending
// atomicAdd and the last block's tail is exposed with 63 SMs idle.
//
// THE GENERAL RESULT, which is the point of both experiments: a cross-block
// last-block synchronisation costs ~1.5-2 us on this part, and a graph node
// costs 0.605 us.  Folding a node into its producer therefore CANNOT pay,
// whatever the size of the work being folded -- the norm (heavy, 2048-element
// reduction) and the top-4 (light, ~30 shuffle ops) both lose by about the
// same amount.  Steps 1, 2, 3 and 5 of the Phase 4x plan are all this
// mechanism and are all dead.  Only fusions that are purely local within one
// block can pay, and none of the remaining small kernels is shaped that way.
#ifndef LK_FUSE_ROUTER
#define LK_FUSE_ROUTER 0
#endif
struct LKNormEpilogue {
    const float * w;              // [n] norm weights, or null to skip the epilogue
    float         eps;
    float *       y32;            // optional f32 output (the router's input)
    block_q8_1 *  yq8;            // Q8_1 output
    float *       partial;        // [LK_NORM_PARTIALS] scratch
    unsigned int * ctr;           // one counter, reset by the last block
};

// ---------------------------------------------------------------------------
// Phase 4y: the scale-invariant Q8_1 norm split.
//
// An RMSNorm multiplies all 2048 activations by one positive scalar, inv, and
// the next kernel quantises the result to Q8_1.  ggml's block_q8_1 stores
//     ds.x = d   = amax/127
//     ds.y = SUM = the RAW FP32 SUM of the 32 activations    (see quant.cuh)
// and its dot product is d4*(d8*sumi - 8*SUM) with sumi an integer dp4a.
//
// Write u = x*w for the raw product and y = inv*u for the true activation.
// The int8 quants are invariant to inv --
//     q_i = round(u_i*127/amax_u) = round(y_i*127/amax_y)
// -- and BOTH ds fields are linear in it, so
//     dot(y) = inv * dot(u)
// up to where the fp32 multiply lands.  The norm therefore splits across the
// kernel boundary with NO synchronisation at all:
//     producer: quantise the RAW x*w         (never needs inv)
//     consumer: ONE multiply per output row  (never needs a per-block scale)
// the only shared state being 64 sum-of-squares partials.
//
// The obvious alternative -- producer emits amax/127, consumer rescales the 64
// ds half2 values -- additionally rounds d through fp16 TWICE, once unscaled and
// once after scaling, for ~5e-4 relative error on every block of every layer.
// This form rounds once.
//
// Residual difference from running the norm as its own kernel: (x*inv)*w versus
// inv*(x*w) reassociate at ~1e-7, which flips an int8 rint in roughly one call
// in fifty.  C2 adjudicates; C1 will move and that is expected, not a failure.
//
// Rejected alternatives, with reasons:
//   * reduce the partials in shared memory and broadcast -- introduces the only
//     __syncthreads() into kernels that have none, and DEADLOCKS wherever a warp
//     can leave early via `if (row >= out_rows) return`.
//   * atomicAdd the partials -- non-deterministic, so replay is not reproducible.
//   * warp 0 reduces while the others stream -- warp 0 also owns an output row,
//     so it lands on the block's critical path anyway.
#define LK_NORM_BLOCKS (LK_HIDDEN / 32)     // 64 Q8_1 blocks in 2048 elements
static_assert(LK_NORM_BLOCKS == 64, "lk_norm_inv reads ssq[lane] and ssq[lane+32]");

// Producer-side descriptor.  `w == nullptr` disables the epilogue entirely, so
// the same kernel serves both the converted and the unconverted layers.
struct LKNormProd {
    const float * __restrict__ w;      // norm weights to apply, or null
    float * __restrict__ h32;          // raw x*w in f32 for the router, or null
    block_q8_1 * __restrict__ yq8;     // quantised RAW activation
    float * __restrict__ ssq;          // [LK_NORM_BLOCKS] sum-of-squares partials
};

// One warp owns exactly one Q8_1 block: 32 consecutive channels, one per lane.
__device__ __forceinline__ void lk_norm_prod_warp(float xv, float uv, int lane,
                                                  block_q8_1 * __restrict__ yq8,
                                                  float * __restrict__ ssq, int b) {
    quantize_q8_1_warp(uv, yq8 + b, lane);
    // Sum of squares is of the RESIDUAL, not of x*w: ggml_rms_norm reduces
    // sum(x^2)/n and applies w afterwards.
    float sq = xv * xv;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if (lane == 0) ssq[b] = sq;
}

// Consumer side: recover inv from the 64 partials.
//
// Every consumer WARP reads the partials itself -- two floats per lane -- rather
// than staging them through shared memory.  That is 256 B of L2 per warp against
// the 2304 B of xq the same warp already reads, and it needs no barrier.
//
// The summation order differs from k_rmsnorm_q8_1's tree, so the result differs
// at ~1e-7.  It is deterministic, which is what the graph replay needs.
__device__ __forceinline__ float lk_norm_inv(const float * __restrict__ ssq,
                                             int lane, float eps) {
    float s = __ldg(ssq + lane) + __ldg(ssq + lane + 32);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    return rsqrtf(s * (1.0f / (float) LK_HIDDEN) + eps);
}

__global__ void k_gemv_q4_0(const uint4 * __restrict__ qs, const __half * __restrict__ d,
                            const block_q8_1 * __restrict__ xq,
                            int out_rows, int bpr, float * __restrict__ y, int add_residual,
                            const float * __restrict__ ssq, float eps);
__global__ void k_gemv_q4_0_norm(const uint4 * __restrict__ qs, const __half * __restrict__ d,
                                 const block_q8_1 * __restrict__ xq,
                                 int out_rows, int bpr, float * __restrict__ y, int add_residual,
                                 LKNormEpilogue ne);
// Phase 4y producer: writes the residual AND quantises it.  Requires the
// LK_GEMV_BIGBLOCK shape (1024 threads), which makes block b own rows
// 32b..32b+31 -- exactly Q8_1 block b, one row per warp.  The 32 values of a
// Q8_1 block therefore live one-per-warp and cannot be shuffled; they meet in
// shared memory exactly once.
__global__ void k_gemv_q4_0_prod(const uint4 * __restrict__ qs, const __half * __restrict__ d,
                                 const block_q8_1 * __restrict__ xq,
                                 int out_rows, int bpr, float * __restrict__ y, int add_residual,
                                 LKNormProd np);

// Generic Q6_K x Q8_1 GEMV: one warp per output row; each super-block consumes
// 8 consecutive Q8_1 blocks.
__global__ void k_gemv_q6_K(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                            const int8_t * __restrict__ sc, const __half * __restrict__ d,
                            const block_q8_1 * __restrict__ xq,
                            int out_rows, int sbpr, float * __restrict__ y,
                            const float * __restrict__ ssq, float eps);
__global__ void k_gemv_q6_K_dp4a(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                                 const int8_t * __restrict__ sc, const __half * __restrict__ d,
                                 const block_q8_1 * __restrict__ xq,
                                 int out_rows, int sbpr, float * __restrict__ y,
                                 const float * __restrict__ ssq, float eps);

// SwiGLU: out[i] = silu(gate[i]) * up[i]
__global__ void k_swiglu(const float * __restrict__ gate, const float * __restrict__ up,
                         int n, float * __restrict__ out, block_q8_1 * __restrict__ yq8,
                         const uint4 * __restrict__ pf_qs = nullptr,
                         const int * __restrict__ pf_idx = nullptr,
                         int pf_rows_per_expert = 0, int pf_bpr = 0,
                         long pf_bytes = 0);

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
//
// T4r.  The selection half, callable from any kernel with one warp to spare.
// This is byte-for-byte the body of the original k_router_top4 -- same four
// rounds of butterfly argmax, same tie-break `ov > mv || (ov == mv && oi < mi)`
// so ties still go to the lower expert index, same unbiased weights
// renormalised under the same guard -- so extracting it cannot change a result.
//
// `sel` is (sigmoid + bias), which is what the RANKING uses; `sig` is the
// unbiased sigmoid, which is what the WEIGHTS use.  Both are 64 floats (512 B
// together) so they stay in L1 across every block that re-reads them.
//
// The expf is deliberately NOT here.  k_router_dot already has each logit in a
// register and computes the sigmoid once per expert (64 expf total); moving it
// into a consumer that runs in 1536 blocks would cost 1536*64 = 98k expf, about
// 3.3 us per layer, which is more than the node it removes.
__device__ __forceinline__ void lk_router_select(const float * __restrict__ sel,
                                                 const float * __restrict__ sig,
                                                 int lane, int * ord, float * wv) {
    static_assert(LK_TOP_K == 4, "four selection rounds assume top-4");
    float v0 = sel[lane], v1 = sel[lane + 32];
    const int i0 = lane,  i1 = lane + 32;
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
        if (i0 == mi) v0 = -1e30f;
        if (i1 == mi) v1 = -1e30f;
    }
    float wsum = 0.0f;
#pragma unroll
    for (int k = 0; k < LK_TOP_K; ++k) { wv[k] = sig[ord[k]]; wsum += wv[k]; }
    if (wsum < LK_TOPK_GUARD) wsum = LK_TOPK_GUARD;
#pragma unroll
    for (int k = 0; k < LK_TOP_K; ++k) wv[k] = wv[k] / wsum;
}

__global__ void __launch_bounds__(32) k_router_dot(
        const float * __restrict__ h, const float * __restrict__ gate_inp,
        float * __restrict__ logits, const float * __restrict__ bias,
        float * __restrict__ sel, float * __restrict__ sig,
        const float * __restrict__ ssq, float eps);
__global__ void k_router_top4(const float * __restrict__ logits,
                              const float * __restrict__ bias,
                              int * __restrict__ idx_out, float * __restrict__ w_out);
// Phase 4x Step 3: the same selection, folded into the last block of the
// router dot.  MEASURED AND REJECTED -- see the note at LK_FUSE_ROUTER.
__global__ void k_router_dot_top4(const float * __restrict__ h,
                                  const float * __restrict__ gate_inp,
                                  float * __restrict__ logits,
                                  const float * __restrict__ bias,
                                  int * __restrict__ idx_out, float * __restrict__ w_out,
                                  unsigned int * __restrict__ ctr);

// Single-query decode attention over an fp16 KV cache.
// The block width was tried at 1024 (with launch bounds) and measured NEUTRAL:
// attention at 8k context was 466 us per layer against 476 at 256 threads, and
// short context got slightly worse.  Widening the block does not help because
// the two hot loops want opposite things -- the score loop is coalesced at
// token-major and the V loop needs a token split -- so the remaining lever is
// split-K, not more warps.  Kept at 256 so the reduction order is unchanged.
#define LK_ATTN_BLOCK   256
// ---------------------------------------------------------------------------
// Attention context split (Phase 4w).
//
// The grid is (8 kv_heads x slices).  How many slices is worth having depends
// on the context length, and the dependence is real and measured:
//
//   seq      best slices   tok/s at that setting   tok/s at 8 slices
//    272          16              419.1                 418.5
//   2128          32              408.5                 389.4
//   4128          32              394.7                 372.6
//   8128         128              373.3                 352.6
//
// so a compile-time constant is wrong at one end or the other.  The rule that
// reproduces every measured optimum is "aim for LK_ATTN_TARGET_N tokens per
// slice", floored and capped:
//
//     slices = clamp(ceil(seq / TARGET_N), SPLIT_MIN, SPLIT)
//
// LK_ATTN_SPLIT is now the CAP (it sizes the grid and the partial buffer), not
// the operating point.  Every block still launches -- CUDA graphs need fixed
// launch dimensions -- and blocks past the live slice count exit immediately.
//
// The cap makes LK_ATTN_CHUNK a constant: with TARGET_N = 64 and SPLIT_MIN = 8,
// chunk = ceil(seq/slices) is at most 64 for every seq (below 512 the floor of
// 8 dominates and gives chunk <= 64; above it, slices >= seq/64 forces
// chunk <= 64).  LK_ATTN_TARGET_N is therefore both the tuning knob and the
// exact shared-memory requirement.
//
// An earlier attempt set SPLIT=224 as a single global constant on the strength
// of a sweep that turned out to be measuring a kernel producing NaN.  The
// numbers above are all post-fix.
#ifndef LK_ATTN_SPLIT
#define LK_ATTN_SPLIT      128
#endif
#ifndef LK_ATTN_TARGET_N
#define LK_ATTN_TARGET_N   64
#endif
#ifndef LK_ATTN_SPLIT_MIN
#define LK_ATTN_SPLIT_MIN  8
#endif
#define LK_ATTN_CHUNK      LK_ATTN_TARGET_N
// The number of slices that actually hold tokens, as a function of position.
// Both attention kernels derive it the same way from d_pos, so they always
// agree; k_attn_combine uses it to avoid reading partials that were never
// written.
__device__ __forceinline__ int lk_attn_slices(int seq_len) {
    int s = (seq_len + LK_ATTN_TARGET_N - 1) / LK_ATTN_TARGET_N;
    if (s < LK_ATTN_SPLIT_MIN) s = LK_ATTN_SPLIT_MIN;
    if (s > LK_ATTN_SPLIT)     s = LK_ATTN_SPLIT;
    return s;
}
// Ablation switch for the attention kernel, ONLY for finding which of its two
// loops owns the time.  Any non-zero value produces WRONG output.
//   0 = off (the committed configuration)
//   1 = skip the V scan     2 = skip the score dot     3 = skip both
#ifndef LK_ATTN_ABLATE
#define LK_ATTN_ABLATE 0
#endif
#define LK_ATTN_PARTW  (2 + LK_HEAD_DIM)
__global__ void k_attn_split(const float * __restrict__ q,
                             const __half * __restrict__ kc, const __half * __restrict__ vc,
                             const int * __restrict__ d_pos, float * __restrict__ part);
__global__ void k_attn_combine(const float * __restrict__ part,
                               const int * __restrict__ d_pos, float * __restrict__ out,
                               block_q8_1 * __restrict__ yq8);
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
__global__ void k_gemv_q4_0_3(const uint4 * __restrict__ qs0, const __half * __restrict__ d0,
                              int rows0, float * __restrict__ y0,
                              const uint4 * __restrict__ qs1, const __half * __restrict__ d1,
                              int rows1, float * __restrict__ y1,
                              const uint4 * __restrict__ qs2, const __half * __restrict__ d2,
                              int rows2, float * __restrict__ y2,
                              const block_q8_1 * __restrict__ xq, int bpr,
                              const float * __restrict__ ssq, float eps);
__global__ void k_gemv_q4_0_exp2(const uint4 * __restrict__ qs_a, const __half * __restrict__ d_a,
                                 const uint4 * __restrict__ qs_b, const __half * __restrict__ d_b,
                                 const block_q8_1 * __restrict__ xq,
                                 int out_rows, int bpr, int rows_per_expert,
                                 int xq_stride,
                                 const int * __restrict__ d_idx,
                                 const float * __restrict__ sel, const float * __restrict__ sig,
                                 int * __restrict__ idx_out, float * __restrict__ w_out,
                                 float * __restrict__ y_a, float * __restrict__ y_b,
                                 const float * __restrict__ ssq, float eps);
__global__ void k_gemv_q4_0_exp(const uint4 * __restrict__ qs, const __half * __restrict__ d,
                                const block_q8_1 * __restrict__ xq,
                                int out_rows, int bpr, int rows_per_expert,
                                int xq_stride,   // 0 = all experts share the same
                                                 // activation; bpr = per-expert
                                const int * __restrict__ d_idx,
                                float * __restrict__ y);

// dst[i] += sum_k w[k] * down4[k*n + i]  -- keeps the router weights on device,
// which removes the per-MoE-layer cudaStreamSynchronize host round-trip.
//
// Phase 4y: this is also the producer for the NEXT layer's attn_norm (and for
// norm_final at layer 39).  Thread i owns element i and warp w owns elements
// 32w..32w+31, which is exactly Q8_1 block w -- so the epilogue needs no
// cross-warp communication, no barrier, and no counter.  This is the 38-site
// half of the fusion that requires no launch-shape change at all.
__global__ void k_moe_accum4(float * __restrict__ dst, const float * __restrict__ down4,
                             const float * __restrict__ d_w, int n, LKNormProd np);

// T1: k_moe_accum4 folded into the expert down projection.
//
// k_moe_accum4 used to be its own graph node: it read the four expert outputs
// back from global memory, weighted them, added the residual and ran the Phase
// 4y norm epilogue -- but moe_down had written those four outputs one node
// earlier, so the round trip was 32 KB of pure waste per layer.
//
// The barrier this removes is not a cross-block one, it is a whole kernel
// launch, and Phase 4x established that a node costs more than the work it
// carries.  Grid is LK_HIDDEN/32 x 1024, so one thread block owns 32 output
// rows == exactly one Q8_1 block of the next activation, and the epilogue is
// the same shared-memory pattern k_gemv_q4_0_prod uses.
__global__ void k_moe_down_accum(const uint4 * __restrict__ qs, const __half * __restrict__ d,
                                 const block_q8_1 * __restrict__ xq,
                                 int out_rows, int bpr, int rows_per_expert, int xq_stride,
                                 const int * __restrict__ d_idx, const float * __restrict__ d_w,
                                 float * __restrict__ dst, LKNormProd np);
