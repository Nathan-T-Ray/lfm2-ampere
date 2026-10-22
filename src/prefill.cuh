// prefill.cuh - the batched prefill path.
//
// Decode runs one token per CUDA graph replay, which is right for generation and
// wrong for a prompt: every prompt token used to cost one full decode step, so a
// 131072-token prompt was 131072 sequential steps (526 s against llama.cpp's
// 34.5 s).  This path ingests a CHUNK of prompt tokens per kernel launch.
//
// It computes the SAME per-token formulas as the decode kernels -- the same Q8_1
// activation quantiser, the same Phase 4y norm split (quantise x*w, multiply the
// consumer's output by inv), the same Q4_0 per-block term d4*(d8*sumi - 8*sum),
// the same router selection, the same shortconv taps, the same RoPE -- so the
// only numerical differences from the sequential path are:
//   * the ORDER in which a GEMM's 32-element blocks are summed (the GEMV sums
//     lane-strided partials through a butterfly; the tensor-core GEMM sums them
//     in block order), which is reassociation at ~1e-7 relative;
//   * attention, which runs on fp16 tensor cores (fp16 Q and P, fp32
//     accumulation) instead of fp32 CUDA cores.
// Neither is bit-identical, so the gate is accuracy against the sequential path
// and against the float64 reference, not memcmp (see tests/test_prefill.cu).
//
// The LAST prompt token always goes through the ordinary decode graph, so the
// logits, d_token and d_pos that generation starts from are produced by exactly
// the code that produces every later token.
#pragma once
#include "kernels.cuh"

// Planar Q8_1 activations.  block_q8_1 is 36 bytes, which no 16-byte copy can
// tile, so the batched path keeps the quants and the scales in separate planes:
//   qs : int8   [rows][kb * 32]
//   sc : float2 [rows][kb]   .x = d8 = float(fp16(amax/127))
//                            .y = 8 * float(fp16(sum))   (exact: *8 is a power of 2)
// The values are exactly what block_q8_1 would hold, widened to fp32.
struct PfAct {
    int8_t * qs;
    float2 * sc;
};

// One tensor-core Q4_0 x Q8_1 GEMM:  out[orow][n] (+)= scale[arow] * sum_b f(b)
// with f(b) = d4[n][b] * (d8[arow][b] * sumi - 8*sum[arow][b]).
//
// Rows (output features) can come from up to three weight tensors stacked in
// row order (q|k|v, gate|up), and for MoE every tensor is offset by
// expert * exp_stride blocks.  Tokens are either a dense range [0, ntok) or a
// list of slots grouped by expert (perm + tiles).
struct PfGemm {
    const uint4 *  qs[3];
    const __half * d[3];
    int            seg_end[3];   // cumulative row ends; boundaries are multiples of 128
    int            N;            // total output rows, multiple of 128
    int            bpr;          // 32-element blocks per row (K/32), multiple of 4
    long long      exp_stride;   // blocks per expert (rows_per_expert * bpr), 0 if dense
    const int8_t * aqs;          // activations: qs plane, row pitch bpr*32
    const float2 * asc;          //              sc plane, row pitch bpr
    int            ntok;         // dense: number of token rows
    const int *    perm;         // MoE: slot ids grouped by expert, or null (dense)
    const int4 *   tiles;        // MoE: {expert, start in perm, count, 0}
    const int *    ntiles;       // MoE: device-side tile count
    int            a_shift;      // activation row = slot >> a_shift
    const float *  scale;        // per activation row, or null (= 1.0f)
    float *        out;
    int            out_ld;
    int            add;          // 1: out += result (the residual add)
    int8_t *       oqs;          // SwiGLU epilogue only: Q8_1 output planes,
    float2 *       osc;          //   okb 32-element blocks per row
    int            okb;
};

// Host launchers (prefill.cu).
void pf_gemm_dense(const PfGemm & g, cudaStream_t st);
void pf_gemm_moe(const PfGemm & g, int max_tiles, cudaStream_t st);
void pf_gemm_swiglu(const PfGemm & g, int max_tiles, cudaStream_t st);   // max_tiles ignored if dense
#define PF_MOE_BN 64              // token tile of the MoE GEMMs (also the plan's tile size)

// ---- kernels ---------------------------------------------------------------
__global__ void k_pf_embed(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                           const int8_t * __restrict__ sc, const __half * __restrict__ d,
                           const int * __restrict__ toks, float * __restrict__ x);
// plain = 1 reproduces k_rmsnorm_q8_1 ((x*inv)*w quantised, consumers scale by 1);
// plain = 0 is the Phase 4y split (x*w quantised, inv written for the consumer).
__global__ void k_pf_norm(const float * __restrict__ X, const float * __restrict__ w, int plain,
                          PfAct a, float * __restrict__ inv_out, float * __restrict__ h32);
__global__ void k_pf_conv(const float * __restrict__ Y, const float * __restrict__ cw,
                          const float * __restrict__ hist, PfAct a);
__global__ void k_pf_conv_hist(const float * __restrict__ Y, int m, float * __restrict__ hist);
__global__ void k_pf_qknr(const float * __restrict__ Y, const float * __restrict__ qn,
                          const float * __restrict__ kn, int P,
                          __half * __restrict__ qh, __half * __restrict__ kc, __half * __restrict__ vc);
__global__ void k_pf_attn(const __half * __restrict__ qh, int m, int P,
                          const __half * __restrict__ kc, const __half * __restrict__ vc,
                          float * __restrict__ o);
// INT8-score attention (prefill.cu): quantise Q and K per 64-row, then score
// with the int8 mma.  ksc is head-major [8][kstride]; cm is the per-head
// smoothing vector [8][64] (k_pf_kmean) or null.  kstride must be a multiple
// of 4 (the scale tiles are copied 16 bytes at a time).
__global__ void k_pf_q8(const __half * __restrict__ qh, int m, const float * __restrict__ cm,
                        int8_t * __restrict__ q8, float * __restrict__ qsc, float * __restrict__ qcs);
__global__ void k_pf_k8(const __half * __restrict__ kc, int p0, int p1, const float * __restrict__ cm,
                        int kstride, int8_t * __restrict__ k8, float * __restrict__ ksc);
__global__ void k_pf_kmean(const __half * __restrict__ kc, int n, float * __restrict__ cm);
// Key tiles within `near` positions of a block's first query (and tile 0 with
// `sink`) are scored exactly as k_pf_attn does, from qh and the fp16 cache kc.
__global__ void k_pf_attn8(const int8_t * __restrict__ q8, const float * __restrict__ qsc,
                           const float * __restrict__ qcs, const __half * __restrict__ qh, int m, int P,
                           const int8_t * __restrict__ k8, const float * __restrict__ ksc, int kstride,
                           const __half * __restrict__ kc, const __half * __restrict__ vc,
                           int near, int sink, float * __restrict__ o);
__global__ void k_pf_quant_rows(const float * __restrict__ X, int ld, int kb, PfAct a);
__global__ void k_pf_swiglu(const float * __restrict__ Y, int ld, int up_off, int kb, PfAct a);
__global__ void k_pf_router(const float * __restrict__ H, int m, const float * __restrict__ Wr,
                            const float * __restrict__ bias, const float * __restrict__ inv,
                            float * __restrict__ sel, float * __restrict__ sig);
__global__ void k_pf_router_small(const float * __restrict__ H, const float * __restrict__ Wr,
                                  const float * __restrict__ bias, const float * __restrict__ inv,
                                  float * __restrict__ sel, float * __restrict__ sig);
__global__ void k_pf_select(const float * __restrict__ sel, const float * __restrict__ sig, int m,
                            int * __restrict__ idx, float * __restrict__ w);
__global__ void k_pf_moe_count(const int * __restrict__ idx, int n, int * __restrict__ cnt);
__global__ void k_pf_moe_plan(const int * __restrict__ cnt, int * __restrict__ off,
                              int4 * __restrict__ tiles, int * __restrict__ ntiles);
__global__ void k_pf_moe_scatter(const int * __restrict__ idx, int n, const int * __restrict__ off,
                                 int * __restrict__ perm);
__global__ void k_pf_moe_combine(float * __restrict__ X, const float * __restrict__ D,
                                 const float * __restrict__ w);
__global__ void k_pf_moe_combine_norm(float * __restrict__ X, const float * __restrict__ D,
                                      const float * __restrict__ w, const float * __restrict__ nw,
                                      PfAct a, float * __restrict__ inv_out);
__global__ void k_pf_pack_q8_1(PfAct a, int t, block_q8_1 * __restrict__ y);
