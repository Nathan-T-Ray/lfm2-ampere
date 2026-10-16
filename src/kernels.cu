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
    LK_ABL_REST;   // LK_ABLATE_SMALL measurement hook
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
// Phase 4x: the Q4_0 GEMV with an optional RMSNorm epilogue folded in.
//
// See kernels.cuh for why.  The epilogue is the tail of k_rmsnorm_q8_1's
// n == 2048 path, moved into the block that happens to finish last.  The
// thread-to-element mapping is kept identical to that kernel (thread `tid`
// owns elements tid + 256j, which is exactly Q8_1 block (tid>>5) + 8j), so the
// quantisation itself is unchanged; only the reduction tree that produces
// `inv` differs, and it is deterministic.
//
// The done-counter is a plain device counter shared by every call site.  That
// is safe because all of them are stream-ordered and the last block resets it
// before exiting, but it means the reset is load-bearing: if the last block
// fails to run, the epilogue is skipped silently on the next replay.  A stale
// counter produces plausible output, not a crash, which is why
// tests/test_normfuse.cu drives two consecutive replays rather than one.
template <bool NORM_EPILOGUE>
__device__ __forceinline__ void k_gemv_q4_0_t(const uint4 * __restrict__ qs,
                              const __half * __restrict__ d,
                              const block_q8_1 * __restrict__ xq,
                              int out_rows, int bpr, float * __restrict__ y, int add_residual,
                              LKNormEpilogue ne,
                              const float * __restrict__ ssq, float eps) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = warp * LK_GEMV_ROWS;
    if (row0 >= out_rows) return;

    // Phase 4y: when the preceding RMSNorm was fused into its producer, `xq`
    // holds the RAW x*w and the true activation is inv*(x*w).  The whole dot is
    // linear in inv (see kernels.cuh), so ONE multiply per output row restores
    // it -- no per-block scale, no barrier.  Issued here rather than after the
    // loop so the two loads are in flight alongside the first weight loads.
    // `ssq` is uniform across the grid, so the branch costs nothing.
    float ninv = 1.0f;
    if (ssq) ninv = lk_norm_inv(ssq, lane, eps);

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
    float v[LK_GEMV_ROWS];
#pragma unroll
    for (int r = 0; r < LK_GEMV_ROWS; ++r) {
        const int row = row0 + r;
        v[r] = 0.0f;
        if (row < out_rows) {
            const float a = warp_sum(acc[r]) * ninv;
            if (NORM_EPILOGUE) {
                // warp_sum returns the total to every lane, so all 32 lanes
                // compute the same v and the sum of squares can be formed
                // without another shuffle.
                v[r] = add_residual ? (y[row] + a) : a;
                if (lane == 0) y[row] = v[r];
            } else {
                if (lane == 0) y[row] = add_residual ? (y[row] + a) : a;
            }
        }
    }

    if (!NORM_EPILOGUE) return;

    static_assert(LK_GEMV_ROWS == 1,
                  "the fused norm epilogue assumes one output row per warp");
    static_assert(LK_GEMV_BLOCKS(LK_HIDDEN) == LK_NORM_PARTIALS,
                  "LK_NORM_PARTIALS must equal the block count of a 2048-row GEMV");

    __shared__ float s_warp[8];
    __shared__ int   s_last;
    __shared__ float s_ss;
    __shared__ float s_red[LK_NORM_PARTIALS];

    if (lane == 0) {
        float sq = 0.0f;
#pragma unroll
        for (int r = 0; r < LK_GEMV_ROWS; ++r) sq += v[r] * v[r];
        s_warp[threadIdx.x >> 5] = sq;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) s += s_warp[i];
        ne.partial[blockIdx.x] = s;
        // Required: without it the last block may read a partial written by a
        // block that has incremented the counter but whose store has not
        // reached L2 yet.  This is the standard threadfence-reduction pattern.
        __threadfence();
        s_last = (atomicAdd(ne.ctr, 1u) == (unsigned) gridDim.x - 1u);
    }
    __syncthreads();
    if (!s_last) return;

    // ---- last block out ---------------------------------------------------
    s_red[threadIdx.x] = ne.partial[threadIdx.x];
    __syncthreads();
#pragma unroll
    for (int o = LK_NORM_PARTIALS / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) s_red[threadIdx.x] += s_red[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) { s_ss = s_red[0]; *ne.ctr = 0u; }
    __syncthreads();

    const float inv = rsqrtf(s_ss / (float) out_rows + ne.eps);
    const int wp = threadIdx.x >> 5, ln = threadIdx.x & 31;
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
        const int i0 = threadIdx.x + j * 256, i1 = threadIdx.x + (j + 1) * 256;
        const float t0 = y[i0] * inv * ne.w[i0];
        const float t1 = y[i1] * inv * ne.w[i1];
        if (ne.y32) { ne.y32[i0] = t0; ne.y32[i1] = t1; }
        if (ne.yq8) quantize_q8_1_warp2(t0, t1, ne.yq8 + wp + 8 * j, ne.yq8 + wp + 8 * j + 8, ln);
    }
}

__global__ void k_gemv_q4_0(const uint4 * __restrict__ qs, const __half * __restrict__ d,
                            const block_q8_1 * __restrict__ xq,
                            int out_rows, int bpr, float * __restrict__ y, int add_residual,
                            const float * __restrict__ ssq, float eps) {
    k_gemv_q4_0_t<false>(qs, d, xq, out_rows, bpr, y, add_residual, LKNormEpilogue{},
                         ssq, eps);
}

// Phase 4y producer.  Requires the LK_GEMV_BIGBLOCK launch shape: 1024 threads,
// so block b owns rows 32b..32b+31 -- exactly Q8_1 block b, one row per warp.
//
// The 32 values of a Q8_1 block therefore live one-per-warp and cannot be
// shuffled across; they meet in shared memory exactly once.  That is the whole
// reason this needs a shape change while k_moe_accum4 does not.
//
// The barrier is unconditional and outside every early-exit: with 2048 rows and
// 64 blocks of 32 warps the grid is exact, but a divergent __syncthreads() would
// hang rather than merely be wrong, so it is not placed behind a guard.
__global__ void k_gemv_q4_0_prod(const uint4 * __restrict__ qs, const __half * __restrict__ d,
                                 const block_q8_1 * __restrict__ xq,
                                 int out_rows, int bpr, float * __restrict__ y, int add_residual,
                                 LKNormProd np) {
    __shared__ float s_x[32];
    __shared__ float s_u[32];
    const int tw   = threadIdx.x >> 5;                 // warp within the block
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * 32 + tw;             // one output row per warp

    if (row < out_rows) {
        float acc = 0.0f;
        for (int k = lane; k < bpr; k += 32)
            acc += dot_q4_0_q8_1(qs + (size_t)row * bpr + k, d + (size_t)row * bpr + k, 0, xq + k);
        acc = warp_sum(acc);
        if (lane == 0) {
            const float xv = add_residual ? (y[row] + acc) : acc;
            y[row] = xv;
            const float uv = np.w ? xv * np.w[row] : 0.0f;
            if (np.h32) np.h32[row] = uv;
            s_x[tw] = xv;
            s_u[tw] = uv;
        }
    }
    __syncthreads();
    // One Q8_1 block per kernel block, owned entirely by warp 0.
    // NOTE: lk_norm_prod_warp indexes yq8 and ssq by `b` itself -- pass the BASE
    // pointer, not a pre-offset one.  (Offsetting here as well writes every
    // odd-numbered activation block and leaves the rest stale.)
    if (np.w && tw == 0)
        lk_norm_prod_warp(s_x[lane], s_u[lane], lane, np.yq8, np.ssq, blockIdx.x);
}

__global__ void k_gemv_q4_0_norm(const uint4 * __restrict__ qs, const __half * __restrict__ d,
                                 const block_q8_1 * __restrict__ xq,
                                 int out_rows, int bpr, float * __restrict__ y, int add_residual,
                                 LKNormEpilogue ne) {
    k_gemv_q4_0_t<true>(qs, d, xq, out_rows, bpr, y, add_residual, ne, nullptr, 0.0f);
}

// ---------------------------------------------------------------------------
__global__ void k_gemv_q6_K(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                            const int8_t * __restrict__ sc, const __half * __restrict__ d,
                            const block_q8_1 * __restrict__ xq,
                            int out_rows, int sbpr, float * __restrict__ y,
                            const float * __restrict__ ssq, float eps) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= out_rows) return;
    // Phase 4y: norm_final is produced by k_moe_accum4, so `xq` here is the raw
    // x*w and the true activation is inv*(x*w).  One multiply per output row.
    float inv = 1.0f;
    if (ssq) inv = lk_norm_inv(ssq, lane, eps);
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
    if (lane == 0) y[warp] = acc * inv;
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
                                 int out_rows, int sbpr, float * __restrict__ y,
                                 const float * __restrict__ ssq, float eps) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= out_rows) return;
    // Phase 4y: see k_gemv_q6_K.
    float ninv = 1.0f;
    if (ssq) ninv = lk_norm_inv(ssq, lane, eps);

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
    if (lane == 0) y[warp] = acc * ninv;
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
                         int n, float * __restrict__ out, block_q8_1 * __restrict__ yq8,
                         const uint4 * __restrict__ pf_qs,
                         const int * __restrict__ pf_idx,
                         int pf_rows_per_expert, int pf_bpr,
                         long pf_bytes) {
    // Speculative L2 prefetch of the NEXT kernel's weights.  The host is this
    // kernel because it has 12288 threads, an idle memory system, and d_idx is
    // already known (router_top4 ran before moe_gate_up).  The first wave of
    // moe_down reads all four selected experts concurrently, so the HEADS of
    // all four are what is worth pulling in, not one expert in full.
    if (pf_qs && pf_bytes > 0) {
        const long expert_bytes = (long)pf_rows_per_expert * pf_bpr * 16;
        const long per   = pf_bytes / LK_TOP_K;              // bytes per expert
        const long lines = ((per < expert_bytes ? per : expert_bytes) >> 7);
        const long tid   = (long)blockIdx.x * blockDim.x + threadIdx.x;
        const long nthr  = (long)gridDim.x * blockDim.x;
        for (int k = 0; k < LK_TOP_K; ++k) {
            const char * base = (const char *)(pf_qs +
                                 (size_t)pf_idx[k] * pf_rows_per_expert * pf_bpr);
            for (long i = tid; i < lines; i += nthr) lk_prefetch_l2(base + (i << 7));
        }
    }

    LK_ABL_SWIGLU;   // LK_ABLATE_SMALL measurement hook
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
    LK_ABL_SHORTCONV;   // LK_ABLATE_SMALL measurement hook
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
// LK_ROUTER_UNROLL controls how many loads each router warp keeps in flight.
//
// It is a SWEEPABLE CONSTANT AND IT IS BIT-IDENTICAL AT EVERY VALUE: the
// accumulation runs i = lane, lane+32, lane+64, ... in strictly increasing
// index order, and groups are contiguous ([g*32*U, (g+1)*32*U)), so changing U
// only re-partitions that same sequence -- it never reorders a single add.
//
// It matters because the router is the most under-parallelised kernel in the
// engine: 64 warps for a 524 KB read, i.e. 0.76 warps per SM.  With U = 16 each
// warp exposes HIDDEN/(32*16) = 4 separate memory round trips; the kernel is
// latency-bound, not bandwidth-bound, and this is the lever.
#ifndef LK_ROUTER_UNROLL
#define LK_ROUTER_UNROLL 16
#endif
__global__ void __launch_bounds__(32) k_router_dot(
        const float * __restrict__ h, const float * __restrict__ gate_inp,
        float * __restrict__ logits, const float * __restrict__ bias,
        float * __restrict__ sel, float * __restrict__ sig,
        const float * __restrict__ ssq, float eps) {
    LK_ABL_ROUTER;   // LK_ABLATE_SMALL measurement hook
    const int e = blockIdx.x;
    const int lane = threadIdx.x;
    // Phase 4y: with ffn_norm folded into the mixer output projection, `h` holds
    // the RAW x*w and the true activation is inv*(x*w).  The router's dot is
    // linear in that scalar for the same reason the Q4_0 dot is, so one multiply
    // on the logit restores it.  This is the only consumer that needs the f32
    // form rather than the quantised one.
    float ninv = 1.0f;
    if (ssq) ninv = lk_norm_inv(ssq, lane, eps);
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
    if (lane == 0) {
        const float l = acc * ninv;
        logits[e] = l;
        // T4r.  The sigmoid happens HERE, where the logit is already in a
        // register and there is exactly one thread per expert -- 64 expf for
        // the whole layer.  Doing it in the consumer would run it once per
        // block of the 1536-block gate/up GEMV: 98k expf, ~3.3 us per layer,
        // which is more than the node this transformation deletes.
        if (sel) {
            const float g = 1.0f / (1.0f + expf(-l));
            sig[e] = g;
            sel[e] = g + bias[e];
        }
    }
}

// The selection half: sigmoid + bias ranking, top-4, renormalised.  Byte for
// byte the original body, except that the 64 logits now arrive from global
// memory instead of a block-local array.
__global__ void k_router_top4(const float * __restrict__ logits,
                              const float * __restrict__ bias,
                              int * __restrict__ idx_out, float * __restrict__ w_out) {
#if LK_ABLATE_SMALL == 1 || LK_ABLATE_SMALL == 5
    // CONTAMINATION GUARD.  A bare early return here leaves d_idx at whatever
    // the buffer holds, and if that is all zeros then moe_gate_up reads ONE
    // expert four times -- 3.5 MB, which fits in the 6 MB L2, so three quarters
    // of its DRAM traffic silently disappears and the router looks far more
    // expensive than it is.  Writing four DISTINCT experts keeps the weight
    // traffic identical to the real thing, which is the whole point.
    if (threadIdx.x < LK_TOP_K) {
        idx_out[threadIdx.x] = threadIdx.x;
        w_out[threadIdx.x] = 0.25f;
    }
    return;
#endif
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
    // The body now lives in lk_router_select so that this kernel and the T4r
    // consumer prologue cannot drift apart.  Shared memory is no longer needed
    // for the selection -- lk_router_select reads the two 64-float arrays
    // straight from global, where they stay L1-resident.
    const int lane = t;
    int   ord[LK_TOP_K];
    float wv[LK_TOP_K];
    lk_router_select(s_sel, s_sig, lane, ord, wv);
    if (lane != 0) return;
#pragma unroll
    for (int k = 0; k < LK_TOP_K; ++k) { idx_out[k] = ord[k]; w_out[k] = wv[k]; }
}

// ---------------------------------------------------------------------------
// Phase 4x Step 3: k_router_top4 folded into k_router_dot.
//
// This is the case the fusion arithmetic actually favours.  The top-4 costs
// about thirty shuffle-and-compare operations, far below the ~1.9 us a
// single-block node costs (0.605 us of dispatch plus the rest as kernel
// start/stop latency), whereas the FFN norm Step 1 tried to fold carries a
// genuine 2048-element reduction and absorbed almost exactly what it replaced.
// If folding pays anywhere, it pays here.
//
// Same threadfence-reduction pattern as the norm epilogue, over 64 blocks
// instead of 256, and the counter is reset by the last block so graph replay
// is safe.  The selection is the SAME code as k_router_top4 -- same sigmoid,
// same four rounds of butterfly argmax, same tie-break `ov > mv || (ov == mv
// && oi < mi)`, same unbiased weights renormalised with the same guard -- run
// on 32 lanes instead of 64, each carrying two candidates.  Any divergence
// from the standalone kernel is a bug, not a rounding difference.
__global__ void __launch_bounds__(32) k_router_dot_top4(
        const float * __restrict__ h, const float * __restrict__ gate_inp,
        float * __restrict__ logits, const float * __restrict__ bias,
        int * __restrict__ idx_out, float * __restrict__ w_out,
        unsigned int * __restrict__ ctr) {
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

    __shared__ int s_last;
    if (lane == 0) {
        logits[e] = acc;
        __threadfence();                       // logits[e] visible before the count
        const unsigned prev = atomicAdd(ctr, 1u);
        s_last = (prev == (unsigned) gridDim.x - 1u);
        if (s_last) *ctr = 0u;                 // reset for the next graph replay
    }
    __syncthreads();
    if (!s_last) return;

    // ---- top-4, on this block's 32 lanes, two candidates each -------------
    __shared__ float s_sel[LK_N_EXPERTS];
    __shared__ float s_sig[LK_N_EXPERTS];
    for (int t = lane; t < LK_N_EXPERTS; t += 32) {
        const float g = 1.0f / (1.0f + expf(-logits[t]));
        s_sig[t] = g;
        s_sel[t] = g + bias[t];
    }
    __syncthreads();
    float v0 = s_sel[lane], v1 = s_sel[lane + 32];
    const int i0 = lane,     i1 = lane + 32;
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
        ord[r] = mi;
        if (i0 == mi) { v0 = -1e30f; }
        if (i1 == mi) { v1 = -1e30f; }
    }
    if (lane != 0) return;
    float wsum = 0.0f;
    float wv[LK_TOP_K];
#pragma unroll
    for (int k = 0; k < LK_TOP_K; ++k) { wv[k] = s_sig[ord[k]]; wsum += wv[k]; }
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
    LK_ABL_REST;   // LK_ABLATE_SMALL measurement hook
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
    LK_ABL_REST;   // LK_ABLATE_SMALL measurement hook
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
    LK_ABL_REST;   // LK_ABLATE_SMALL measurement hook
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
    LK_ABL_REST;   // LK_ABLATE_SMALL measurement hook
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
                              const block_q8_1 * __restrict__ xq, int bpr,
                              const float * __restrict__ ssq, float eps) {
    const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    // Phase 4y.  The grid is exact here (3072 warps for 2048+512+512 rows), so
    // there is no early exit and the shuffles below are always full-mask.
    float inv = 1.0f;
    if (ssq) inv = lk_norm_inv(ssq, lane, eps);
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
    if (lane == 0) y[row] = acc * inv;
}

__global__ void k_gemv_q4_0_exp2(const uint4 * __restrict__ qs_a, const __half * __restrict__ d_a,
                                 const uint4 * __restrict__ qs_b, const __half * __restrict__ d_b,
                                 const block_q8_1 * __restrict__ xq,
                                 int out_rows, int bpr, int rows_per_expert,
                                 int xq_stride,
                                 const int * __restrict__ d_idx,
                                 const float * __restrict__ sel, const float * __restrict__ sig,
                                 int * __restrict__ idx_out, float * __restrict__ w_out,
                                 float * __restrict__ y_a, float * __restrict__ y_b,
                                 const float * __restrict__ ssq, float eps) {
    const int half = blockIdx.y >> 2;               // 0 = gate, 1 = up
    const int k    = blockIdx.y & (LK_TOP_K - 1);   // which of the 4 experts
    const uint4  * qs = half ? qs_b : qs_a;
    const __half * d  = half ? d_b  : d_a;
    float        * y  = half ? y_b  : y_a;
    const int tw   = threadIdx.x >> 5;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

#if LK_ROUTER_PRO
    // ---- T4r: the top-4, re-derived here instead of by its own node --------
    //
    // k_router_top4 was a <<<1,64>>> node costing 1.769 us per layer in a
    // graph -- far more than the ~70 instructions of work it carries.  The
    // selection is barrier-free in the Phase 4y sense: k_router_dot has already
    // reduced the logits to two 64-float arrays (512 B, L1-resident), so every
    // block can rebuild the answer from them with one warp and no cross-block
    // communication at all.
    //
    // Only warp 0 runs it, and the other 7 warps wait on the barrier below.
    // Re-running it in every warp instead would avoid the wait but costs
    // 1536 blocks * 8 warps * 70 instructions, measured at ~1.45 us per layer,
    // which is most of what this saves.
    __shared__ int s_idx[LK_TOP_K];
    if (tw == 0) {
        int   ord[LK_TOP_K];
        float wv[LK_TOP_K];
        lk_router_select(sel, sig, lane, ord, wv);
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < LK_TOP_K; ++j) s_idx[j] = ord[j];
            // Only the expert INDICES are needed inside this grid.  The weights
            // are consumed from global by moe_down, which is a later kernel
            // launch, so they are written once by block (0,0) and never staged
            // through shared memory.
            if (blockIdx.x == 0 && blockIdx.y == 0) {
#pragma unroll
                for (int j = 0; j < LK_TOP_K; ++j) { idx_out[j] = ord[j]; w_out[j] = wv[j]; }
            }
        }
    }
    // Unconditional, and ahead of the early exit below: a divergent
    // __syncthreads hangs rather than errors.
    __syncthreads();
    const int expert = s_idx[k];
    (void) d_idx;
#else
    (void) tw; (void) sel; (void) sig; (void) idx_out; (void) w_out;
    const int expert = d_idx[k];
#endif

    if (warp >= out_rows) return;
    // Phase 4y.  The grid is exact here (blockIdx.x*8 warps covers out_rows, and
    // rows_per_expert == out_rows for this model), so no warp exits early and
    // the full-mask shuffles below are well defined.
    float ninv = 1.0f;
    if (ssq) ninv = lk_norm_inv(ssq, lane, eps);
    const size_t base = (size_t) expert * rows_per_expert * bpr;
    const uint4  * qrow = qs + base + (size_t) warp * bpr;
    const __half * drow = d  + base + (size_t) warp * bpr;
    float acc = 0.0f;
    const block_q8_1 * xq_k = xq + (size_t) k * xq_stride;
    for (int j = lane; j < bpr; j += 32) acc += dot_q4_0_q8_1(qrow, drow, j, xq_k + j);
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) k * out_rows + warp] = acc * ninv;
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
                             const float * __restrict__ d_w, int n, LKNormProd np) {
    const int lane = threadIdx.x & 31;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float a = 0.0f;
#pragma unroll
        for (int k = 0; k < LK_TOP_K; ++k) a += d_w[k] * down4[(size_t)k * n + i];
        const float xv = dst[i] + a;
        dst[i] = xv;
        // Phase 4y.  Thread i owns element i, so warp w owns elements
        // 32w..32w+31 -- exactly Q8_1 block w.  The quantisation and the
        // sum-of-squares are both pure warp ops: no shared memory, no barrier,
        // no counter, and nothing to reset between replays.
        //
        // The launch is <<<8,256>>> for n == 2048, so every lane is active on
        // the single grid-stride iteration and the full-mask shuffles inside
        // lk_norm_prod_warp are well defined.
        if (np.w) {
            const float uv = xv * np.w[i];
            if (np.h32) np.h32[i] = uv;
            lk_norm_prod_warp(xv, uv, lane, np.yq8, np.ssq, i >> 5);
        }
    }
}

// ---------------------------------------------------------------------------
// T1: k_moe_accum4 folded into k_moe_down.
//
// k_moe_accum4 used to be its own graph node: it read the four expert outputs
// back from global, weighted them, added the residual and ran the Phase 4y norm
// epilogue -- but k_gemv_q4_0_exp had written those four outputs one node
// earlier, so the round trip was 4*2048*4 = 32 KB of pure waste per layer, plus
// a whole node.
//
// The barrier this removes is not a cross-block one, it is a kernel launch, and
// Phase 4x established that a node costs more than the work it carries (a
// cross-block last-block sync measured 1.5-2.0 us against 0.605 us of dispatch).
//
// Grid is LK_HIDDEN/32 x 1024: one thread block owns 32 output rows, which is
// exactly one Q8_1 block of the next activation, so the 32 values meet in
// shared memory exactly once and warp 0 quantises them -- the same epilogue
// shape k_gemv_q4_0_prod uses.
//
// Bit-identical to the two kernels it replaces.  The per-expert dot keeps the
// `j = lane; j < bpr; j += 32` order and the same warp_sum, and the weighted
// sum runs k = 0,1,2,3 in the order k_moe_accum4 used, so both the dot and the
// combination round at exactly the same points.
__global__ void k_moe_down_accum(const uint4 * __restrict__ qs, const __half * __restrict__ d,
                                 const block_q8_1 * __restrict__ xq,
                                 int out_rows, int bpr, int rows_per_expert, int xq_stride,
                                 const int * __restrict__ d_idx, const float * __restrict__ d_w,
                                 float * __restrict__ dst, LKNormProd np) {
    __shared__ float s_x[32];
    __shared__ float s_u[32];
    const int tw   = threadIdx.x >> 5;                 // warp within the block
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * 32 + tw;             // one output row per warp

    if (row < out_rows) {
        // The four experts are walked one after another rather than issued
        // together.  Hoisting all four dots above the reductions so their loads
        // overlap is the obvious-looking move and it measured WORSE (12.77 vs
        // 12.32 us per layer): four live accumulators plus four sets of row
        // pointers cost enough registers to lose more latency hiding than the
        // extra memory-level parallelism returns.
        float a = 0.0f;
#pragma unroll
        for (int k = 0; k < LK_TOP_K; ++k) {
            const size_t base = (size_t) d_idx[k] * rows_per_expert * bpr;
            const uint4  * qrow = qs + base + (size_t) row * bpr;
            const __half * drow = d  + base + (size_t) row * bpr;
            const block_q8_1 * xk = xq + (size_t) k * xq_stride;
            float acc = 0.0f;
            for (int j = lane; j < bpr; j += 32) acc += dot_q4_0_q8_1(qrow, drow, j, xk + j);
            a += d_w[k] * warp_sum(acc);
        }
        const float xv = dst[row] + a;
        dst[row] = xv;
        const float uv = np.w ? xv * np.w[row] : 0.0f;
        if (np.h32) np.h32[row] = uv;
        if (lane == 0) { s_x[tw] = xv; s_u[tw] = uv; }
    }
    // Unconditional, and outside every early exit: a divergent __syncthreads
    // hangs rather than errors.
    __syncthreads();
    // One Q8_1 block per kernel block, owned entirely by warp 0.  Pass the BASE
    // yq8 -- lk_norm_prod_warp indexes by `b` itself.
    if (np.w && tw == 0)
        lk_norm_prod_warp(s_x[lane], s_u[lane], lane, np.yq8, np.ssq, blockIdx.x);
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
//
// That grid was still wrong, and Phase 4v below explains why.
// LK_ATTN_CHUNK now lives in kernels.cuh, derived from LK_ATTN_TARGET_N.
#define LK_ATTN_GQA   (LK_N_HEADS / LK_N_KV_HEADS)      // query heads per kv head
// The attention split derives its slice count at runtime, so the shared score
// array must cover the LARGEST chunk that rule can ever ask for.  With
// slices = clamp(ceil(seq/TARGET_N), SPLIT_MIN, SPLIT), chunk = ceil(seq/slices)
// is at most TARGET_N provided the cap can cover the whole sequence:
static_assert(LK_ATTN_SPLIT * LK_ATTN_TARGET_N >= LK_ATTN_MAX_SEQ,
              "attention split cap too small: chunk can exceed LK_ATTN_CHUNK");
static_assert(LK_ATTN_CHUNK == LK_ATTN_TARGET_N,
              "LK_ATTN_CHUNK must equal LK_ATTN_TARGET_N; see the argument in kernels.cuh");

// ---------------------------------------------------------------------------
// Phase 4v: GQA-aware split -- the fix for the context cliff.
//
// The Phase 4u grid was (32 query heads x SPLIT), and each block fetched its
// own kv_head's K and V for its slice.  Four query heads share every kv_head,
// so the context was streamed from DRAM FOUR TIMES OVER.  That was invisible
// while this was "just an optimisation": the kernel measured 870 GB/s at 8k
// context against a 956 GB/s ceiling, i.e. it already looked like it was at the
// wall, and 870 GB/s of *useful* data is really 4 x 870 = 3480 GB/s of pages,
// which is impossible.  The traffic was the bug.
//
// The grid is now (8 kv_heads x SPLIT).  One block loads a kv_head's slice
// ONCE and computes all four query heads that share it, so DRAM sees each K
// and V byte exactly once.  Context traffic drops by exactly 4x:
//
//     before   32 heads x SPLIT x (seq/SPLIT) x 128 B x 2 (K,V) = seq * 4096 B
//     after     8 kvh  x SPLIT x (seq/SPLIT) x 128 B x 2 (K,V) = seq * 1024 B
//
// and 8k context costs 164.5 MB instead of 658 MB per layer-token.
//
// Bit-exactness: for each head, dims are still accumulated d = 0..63 in order
// and tokens still accumulate t = 0..n-1 in order, with the same 256-lane
// halving tree for both reductions.  The max is exact under fmaxf regardless of
// order.  So with LK_ATTN_SPLIT unchanged this kernel must produce BIT-IDENTICAL
// partials to the Phase 4u kernel -- which is a testable claim, not a hope.
//
// The partial buffer layout is unchanged: part[head][sp][2 + 64], so
// k_attn_combine is untouched.
__global__ void __launch_bounds__(256)
k_attn_split(const float * __restrict__ q,
             const __half * __restrict__ kc, const __half * __restrict__ vc,
             const int * __restrict__ d_pos, float * __restrict__ part) {
    const int seq_len = *d_pos + 1;
    const int kv_head = blockIdx.x;
    const int sp      = blockIdx.y;
    const int head0   = kv_head * LK_ATTN_GQA;
    // DIM-major q: the four heads that share this kv_head are contiguous for a
    // given dim, so one float4 shared load fetches all four q values.  Head-major
    // (the obvious layout) needs four separate shared loads per dim, i.e. 256
    // floating-point shared transactions per token against 64 -- and it was that
    // shared traffic plus the 4-way dot product's live registers that pushed this
    // kernel to 255 registers with spills, which in turn stopped the V loop
    // unrolling and cost it the memory-level parallelism it needs.  Registers
    // and shared traffic here are the same problem.
    __shared__ __align__(16) float s_q[LK_HEAD_DIM * LK_ATTN_GQA];
    __shared__ float s_score[LK_ATTN_GQA][LK_ATTN_CHUNK];
    __shared__ float s_m[LK_ATTN_GQA][256];
    {
        const int d = threadIdx.x & (LK_HEAD_DIM - 1);
        const int g = threadIdx.x >> 6;                 // 0..3
        s_q[d * LK_ATTN_GQA + g] = q[(head0 + g) * LK_HEAD_DIM + d];
    }
    __syncthreads();

    const int nslices = lk_attn_slices(seq_len);
    const int chunk = (seq_len + nslices - 1) / nslices;
    const int t0 = sp * chunk;
    const int t1 = (t0 + chunk < seq_len) ? (t0 + chunk) : seq_len;
    // Blocks past the live slice count own no tokens and write NOTHING: at short
    // context most of the grid is in this state, and having them store 4 x 66
    // sentinel floats each cost a megabyte of pointless global traffic per
    // layer.  k_attn_combine derives the same boundary and skips them.
    if (t0 >= t1) return;
    const int n = t1 - t0;
    float m0 = -1e30f, m1 = -1e30f, m2 = -1e30f, m3 = -1e30f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const __half2 * k2 = (const __half2 *)
            (kc + ((size_t)(t0 + j) * LK_N_KV_HEADS + kv_head) * LK_HEAD_DIM);
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#if LK_ATTN_ABLATE == 2 || LK_ATTN_ABLATE == 3
        if (k2) { a0 = (float) j; }        // ABLATION: no K read, no dot
#else
        // 32 half2 loads = the token's 64 dims = one 128-byte cache line.  The
        // uint4 form used before also read one line, but a uint4 shared load for
        // q cannot be expressed against it; half2 is the granularity at which
        // both sides can be 128-bit.  Unroll 4 keeps 4 lines in flight, which is
        // what the register budget allows; the 8-way unroll this replaced was
        // the single biggest consumer of registers in the kernel.
#pragma unroll 4
        for (int i = 0; i < LK_HEAD_DIM / 2; ++i) {
            const __half2 w = k2[i];
            const float v0 = __half2float(__low2half(w));
            const float v1 = __half2float(__high2half(w));
            const float4 qa = ((const float4 *) s_q)[2 * i];
            const float4 qb = ((const float4 *) s_q)[2 * i + 1];
            a0 += qa.x * v0; a0 += qb.x * v1;
            a1 += qa.y * v0; a1 += qb.y * v1;
            a2 += qa.z * v0; a2 += qb.z * v1;
            a3 += qa.w * v0; a3 += qb.w * v1;
        }
#endif
        a0 *= LK_ATTN_SCALE; a1 *= LK_ATTN_SCALE;
        a2 *= LK_ATTN_SCALE; a3 *= LK_ATTN_SCALE;
        s_score[0][j] = a0; s_score[1][j] = a1;
        s_score[2][j] = a2; s_score[3][j] = a3;
        m0 = fmaxf(m0, a0); m1 = fmaxf(m1, a1);
        m2 = fmaxf(m2, a2); m3 = fmaxf(m3, a3);
    }
    // fmaxf is exact and associative, so this tree's shape cannot change the
    // result; only the sum tree below is order-sensitive.
    s_m[0][threadIdx.x] = m0; s_m[1][threadIdx.x] = m1;
    s_m[2][threadIdx.x] = m2; s_m[3][threadIdx.x] = m3;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) {
            s_m[0][threadIdx.x] = fmaxf(s_m[0][threadIdx.x], s_m[0][threadIdx.x + o]);
            s_m[1][threadIdx.x] = fmaxf(s_m[1][threadIdx.x], s_m[1][threadIdx.x + o]);
            s_m[2][threadIdx.x] = fmaxf(s_m[2][threadIdx.x], s_m[2][threadIdx.x + o]);
            s_m[3][threadIdx.x] = fmaxf(s_m[3][threadIdx.x], s_m[3][threadIdx.x + o]);
        }
        __syncthreads();
    }
    const float M0 = s_m[0][0], M1 = s_m[1][0], M2 = s_m[2][0], M3 = s_m[3][0];
    __syncthreads();

    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float p0 = __expf(s_score[0][j] - M0); s_score[0][j] = p0; l0 += p0;
        const float p1 = __expf(s_score[1][j] - M1); s_score[1][j] = p1; l1 += p1;
        const float p2 = __expf(s_score[2][j] - M2); s_score[2][j] = p2; l2 += p2;
        const float p3 = __expf(s_score[3][j] - M3); s_score[3][j] = p3; l3 += p3;
    }
    s_m[0][threadIdx.x] = l0; s_m[1][threadIdx.x] = l1;
    s_m[2][threadIdx.x] = l2; s_m[3][threadIdx.x] = l3;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) {
            s_m[0][threadIdx.x] += s_m[0][threadIdx.x + o];
            s_m[1][threadIdx.x] += s_m[1][threadIdx.x + o];
            s_m[2][threadIdx.x] += s_m[2][threadIdx.x + o];
            s_m[3][threadIdx.x] += s_m[3][threadIdx.x + o];
        }
        __syncthreads();
    }
    if (threadIdx.x < LK_ATTN_GQA) {
        float * P = part + ((size_t)(head0 + threadIdx.x) * LK_ATTN_SPLIT + sp) * LK_ATTN_PARTW;
        P[0] = (threadIdx.x == 0) ? M0 : (threadIdx.x == 1) ? M1
             : (threadIdx.x == 2) ? M2 : M3;
        P[1] = s_m[threadIdx.x][0];
    }

    // V scan.
    //
    // The pre-4v kernel had 64 live threads and 192 idle ones, because there was
    // only one head to compute.  With 4 heads there are 4 x 64 = 256 outputs and
    // a naive (head, dim) map uses every thread -- but it issues a 2-byte load
    // per FMA and a warp only covers 64 bytes, half a cache line.  This is the
    // shape that measured best of the ones tried:
    //
    //   tid -> (half, head, pair)  =  (tid>>7, (tid>>5)&3, tid&31)
    //
    // so a warp covers all 32 half2 pairs of ONE head = 128 contiguous bytes =
    // exactly one cache line, and each thread carries TWO dims for the price of
    // one 4-byte load and one shared broadcast.  The slice is split in two over
    // t, which halves the dependent FMA chain and halves the number of iterations
    // -- the loads per byte drop from 1 per 2 bytes to 1 per 4, and the bytes in
    // flight at a given unroll double.  The two half-sums are added once at the
    // end, which is a different rounding from the single chain (measured against
    // the float64 reference by make test-attn, not assumed).
    {
        const int half = threadIdx.x >> 7;
        const int g    = (threadIdx.x >> 5) & (LK_ATTN_GQA - 1);
        const int i    = threadIdx.x & 31;           // half2 index: dims 2i, 2i+1
        const int tlo  = half ? (n >> 1) : 0;
        const int thi  = half ? n : (n >> 1);
        const __half2 * v2 = (const __half2 *)
            (vc + ((size_t) t0 * LK_N_KV_HEADS + kv_head) * LK_HEAD_DIM) + i;
        const float * sc = s_score[g];
        float acc0 = 0.0f, acc1 = 0.0f;
#if LK_ATTN_ABLATE == 1 || LK_ATTN_ABLATE == 3
        if (v2 && tlo < thi) { acc0 = sc[tlo]; }
#else
#pragma unroll 8
        for (int t = tlo; t < thi; ++t) {
            const __half2 w = v2[(size_t) t * LK_N_KV_HEADS * (LK_HEAD_DIM / 2)];
            const float w0 = __half2float(__low2half(w));
            const float w1 = __half2float(__high2half(w));
            const float p = sc[t];
            acc0 += p * w0;
            acc1 += p * w1;
        }
#endif
        __shared__ float2 s_v[2][LK_ATTN_GQA][32];
        s_v[half][g][i] = make_float2(acc0, acc1);
        __syncthreads();
        if (half == 0) {
            const float2 o = make_float2(s_v[0][g][i].x + s_v[1][g][i].x,
                                         s_v[0][g][i].y + s_v[1][g][i].y);
            float * P = part + ((size_t)(head0 + g) * LK_ATTN_SPLIT + sp) * LK_ATTN_PARTW + 2;
            P[2 * i]     = o.x;
            P[2 * i + 1] = o.y;
        }
    }
}

__global__ void k_attn_combine(const float * __restrict__ part, const int * __restrict__ d_pos,
                               float * __restrict__ out,
                               block_q8_1 * __restrict__ yq8) {
    const int head = blockIdx.x;
    const int d = threadIdx.x;            // 64 threads, one per head dim
    const int seq_len = *d_pos + 1;
    const int nslices = lk_attn_slices(seq_len);
    const int chunk = (seq_len + nslices - 1) / nslices;
    const float * P = part + (size_t) head * LK_ATTN_SPLIT * LK_ATTN_PARTW;
    __shared__ float s_m[LK_ATTN_SPLIT];
    __shared__ float s_l[LK_ATTN_SPLIT];
    // Stride the staging loop by the block.  The previous version used
    // `if (d < LK_ATTN_SPLIT)` with d = threadIdx.x and a 64-thread block, which
    // silently only worked while LK_ATTN_SPLIT <= 64: above that it left
    // s_m[64..SPLIT-1] uninitialised and the whole attention output became
    // garbage (parity 0/825, every logit NaN).  Found by make test-attn, which
    // exists precisely because this class of thing is invisible in a benchmark.
    //
    // Slices that hold no tokens are given the same (-1e30, 0) sentinel the old
    // split kernel used to write, so exp(s_m - mf) is exactly 0 for them and
    // they contribute nothing to either the denominator or the numerator.
    // k_attn_split no longer writes them, so this is the only place that knows.
    for (int s = threadIdx.x; s < nslices; s += blockDim.x) {
        const bool empty = (s * chunk >= seq_len);
        s_m[s] = empty ? -1e30f : P[s * LK_ATTN_PARTW + 0];
        s_l[s] = empty ? 0.0f   : P[s * LK_ATTN_PARTW + 1];
    }
    __syncthreads();
    float mf = -1e30f;
    for (int s = 0; s < nslices; ++s) mf = fmaxf(mf, s_m[s]);
    float lf = 0.0f, acc = 0.0f;
    for (int s = 0; s < nslices; ++s) {
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
