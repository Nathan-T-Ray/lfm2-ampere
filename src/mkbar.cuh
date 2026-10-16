// mkbar.cuh - the barrier zoo (Phase 5, campaign 2).
//
// WHY THIS FILE EXISTS.  tests/test_megakernel.cu priced the first grid barrier
// at 2.17-2.56 us against a 0.78-0.83 us kernel boundary, and on that basis the
// megakernel was declared refuted.  That verdict is only as strong as the
// barrier it was measured with, and the barrier it was measured with has THREE
// properties that were never isolated:
//
//   1. FALSE SHARING.  d_bar was cudaMalloc'd at 2*sizeof(int) = 8 bytes, so the
//      arrival counter (bar[0]), the release flag (bar[1]), and every poller's
//      volatile load live on ONE 128-byte L2 line.  Every stage, 504 blocks
//      issue atomicAdd RMWs to that line while up to 504 blocks poll it.  The
//      arrival RMW must take the line exclusive, which invalidates every poller.
//   2. A HIDDEN LAUNCH.  The monotonic scheme needs the counters reset between
//      replays, and the harness did it with cudaMemsetAsync INSIDE the timed
//      region -- for the barrier mode only, never for the separate-launch mode.
//      So part of the "barrier cost" is a memset kernel launch, i.e. exactly the
//      toll the megakernel is supposed to remove.
//   3. BACKOFF AND FENCES, neither swept.  __nanosleep(64) was chosen once.  And
//      __threadfence() compiles to a full membar; sm_70+ has acquire/release at
//      gpu scope, which is the ordering the handoff actually needs.
//
// This file gives the barrier a fair trial, one factor at a time:
//
//   V0  legacy published    unpadded, 2 fences, ns=64, no post-spin acquire
//   V1  unpadded + acquire  -> isolates the POST-SPIN ACQUIRE      vs V0
//   V2  padded,   + acquire -> isolates FALSE SHARING              vs V1
//   V3  padded, ns=0     |
//   V4  padded, ns=32    |  isolates BACKOFF                       vs V2
//   V5  padded, ns=128   |
//   V6  padded, ns=256   |
//   V7  padded, PTX acq/rel, ns=64   -> isolates FENCES            vs V2
//   V8  padded, PTX, ns=0
//   V9  padded, PTX, ns=32
//
// V0..V9 were all measured and all landed within 4% of each other, so none of
// those three hypotheses is the bottleneck.  What is left is TRUE sharing: every
// block increments the SAME L2 address, and 504 serialized atomicAdds at ~2-4 ns
// is 1.0-2.0 us -- i.e. the whole barrier.  So:
//
//   V10 two-level arrival tree, G=4    \
//   V11 two-level arrival tree, G=8     |  attacks ARRIVAL SERIALIZATION
//   V12 two-level arrival tree, G=16    |
//   V13 two-level arrival tree, G=84   /
//
// Every variant still has to pass the memcmp bit-identity gate.  A fast barrier
// that moves a bit is not a barrier.
#ifndef LK_MKBAR_CUH
#define LK_MKBAR_CUH

#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// The padded barrier.  `arrived` and `released` are 128 bytes apart, so the
// arrival RMW traffic and the release/poll traffic cannot share a line.
// ---------------------------------------------------------------------------
struct alignas(128) lk_bp_t {
    unsigned arrived;
    unsigned _pad0[31];
    unsigned released;
    unsigned _pad1[31];
};
static_assert(sizeof(lk_bp_t) == 256, "lk_bp_t must be two 128B lines");

// ---------------------------------------------------------------------------
// PTX acquire/release at gpu scope (sm_70+).  These are the operations the
// hardware actually has for a handoff: the release folds the store and the
// ordering into one instruction, and the acquire does the same for the load.
// __threadfence() is a full membar, which is strictly stronger.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned lk_atom_add_rel(unsigned * p, unsigned v) {
    unsigned old;
    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                 : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}
__device__ __forceinline__ void lk_st_rel(unsigned * p, unsigned v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned lk_ld_acq(const unsigned * p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// ---------------------------------------------------------------------------
// V0: the published barrier, unchanged, with its unpadded int[2] layout.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void lk_bar_v0(int * __restrict__ bar, unsigned nb, unsigned gen) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(&bar[0], 1) + 1 == (int) (gen * nb)) {
            __threadfence();
            *(volatile int *) &bar[1] = (int) gen;
        }
        while (*(volatile int *) &bar[1] < (int) gen) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(64);
#endif
        }
    }
    __syncthreads();
}

// V1: same layout, same fences, same backoff -- but with the post-spin acquire
// the published one is missing.  A waiter that observes the release must not let
// its subsequent loads of the producers' data float above the observation.
__device__ __forceinline__ void lk_bar_v1(int * __restrict__ bar, unsigned nb, unsigned gen) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(&bar[0], 1) + 1 == (int) (gen * nb)) {
            __threadfence();
            *(volatile int *) &bar[1] = (int) gen;
        }
        while (*(volatile int *) &bar[1] < (int) gen) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(64);
#endif
        }
        __threadfence();
    }
    __syncthreads();
}

// V2..V6: padded, fence + volatile, post-spin acquire, backoff NS.
template<int NS>
__device__ __forceinline__ void lk_bar_f(lk_bp_t * __restrict__ bp, unsigned nb, unsigned gen) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned tgt = gen * nb;
        __threadfence();
        if (atomicAdd(&bp->arrived, 1u) + 1u == tgt) {
            __threadfence();
            *(volatile unsigned *) &bp->released = gen;
        }
        while (*(volatile unsigned *) &bp->released < gen) {
#if __CUDA_ARCH__ >= 700
            if (NS) __nanosleep(NS);
#endif
        }
        __threadfence();
    }
    __syncthreads();
}

// V7..V9: padded, PTX acquire/release, backoff NS.
template<int NS>
__device__ __forceinline__ void lk_bar_p(lk_bp_t * __restrict__ bp, unsigned nb, unsigned gen) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned tgt = gen * nb;
        if (lk_atom_add_rel(&bp->arrived, 1u) + 1u == tgt)
            lk_st_rel(&bp->released, gen);
        while (lk_ld_acq(&bp->released) < gen) {
#if __CUDA_ARCH__ >= 700
            if (NS) __nanosleep(NS);
#endif
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// V10..V13: THE TWO-LEVEL ARRIVAL TREE.
//
// Why this and not the others.  Padding the counters (different addresses that
// happened to share a line) changed nothing; folding __threadfence() into PTX
// acquire/release changed nothing; sweeping the backoff changed nothing outside
// noise.  What remains is that every one of the 504 blocks performs an
// atomicAdd on ONE address, and L2 serializes same-address RMWs -- at 2-4 ns per
// operation that is 1.0-2.0 us, which is essentially the entire measured 2.2 us.
//
// The tree splits the grid into G groups by blockIdx % G.  Block b increments
// its group's leaf; only the LAST arriver in each group touches the root.  The
// G leaves are independent so they drain in parallel, and the deepest serial
// chain becomes nb/G + G instead of nb -- at nb=504, G=8 gives 71 instead of 504.
//
// It also deletes the release broadcast.  The old barrier needed a SECOND hop:
// producers -> arrived, then last producer -> released, then consumers poll
// released.  Here the consumers poll the ROOT arrival count directly, so the
// rendezvous is complete the instant the last root increment lands.  One hop.
//
// `leaf` must hold G*32 unsigned, so that no two leaf counters share a line.
// Requires G <= nb, or a group with zero blocks never fires and the root never
// reaches gen*G: a deadlock.  The host asserts this.
// ---------------------------------------------------------------------------
template<int G, int NS>
__device__ __forceinline__ void lk_bar_tree(unsigned * __restrict__ leaf,
                                            unsigned * __restrict__ root,
                                            unsigned nb, unsigned gen) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned g   = blockIdx.x % (unsigned) G;
        const unsigned cnt = (nb - g + (unsigned) G - 1u) / (unsigned) G;
        __threadfence();
        if (atomicAdd(&leaf[(size_t) g * 32u], 1u) + 1u == gen * cnt) {
            __threadfence();
            atomicAdd(root, 1u);              // the only root traffic: G per gen
        }
        const unsigned rtgt = gen * (unsigned) G;
        while (*(volatile unsigned *) root < rtgt) {
#if __CUDA_ARCH__ >= 700
            if (NS) __nanosleep(NS);
#endif
        }
        __threadfence();
    }
    __syncthreads();
}

// Number of variants, for the harness's dispatch and its result table.
#define LK_NBAR_VARIANTS 14
// Leaves for the widest tree (G=84), padded to 32 unsigned each.
#define LK_LEAF_U32 (84 * 32)
// The one the harness treats as "best known"; kept here so the label and the
// dispatch cannot drift apart.
#define LK_BAR_BEST 2

// Host-side only: it is a label table for the harness, never used in a kernel.
static inline const char * lk_bar_name(int v) {
    switch (v) {
    case 0: return "legacy unpadded 2fence ns64";
    case 1: return "unpadded +acquire ns64";
    case 2: return "PADDED 2fence ns64";
    case 3: return "PADDED 2fence ns0";
    case 4: return "PADDED 2fence ns32";
    case 5: return "PADDED 2fence ns128";
    case 6: return "PADDED 2fence ns256";
    case 7: return "PADDED PTX ns64";
    case 8: return "PADDED PTX ns0";
    case 9: return "PADDED PTX ns32";
    case 10: return "TREE G=4";
    case 11: return "TREE G=8";
    case 12: return "TREE G=16";
    case 13: return "TREE G=84";
    default: return "?";
    }
}

template<int V>
__device__ __forceinline__ void lk_bar_any(int * bar, lk_bp_t * bp, unsigned * leaf,
                                           unsigned nb, unsigned gen) {
    if      (V == 0) lk_bar_v0(bar, nb, gen);
    else if (V == 1) lk_bar_v1(bar, nb, gen);
    else if (V == 2) lk_bar_f<64>(bp, nb, gen);
    else if (V == 3) lk_bar_f<0>(bp, nb, gen);
    else if (V == 4) lk_bar_f<32>(bp, nb, gen);
    else if (V == 5) lk_bar_f<128>(bp, nb, gen);
    else if (V == 6) lk_bar_f<256>(bp, nb, gen);
    else if (V == 7) lk_bar_p<64>(bp, nb, gen);
    else if (V == 8) lk_bar_p<0>(bp, nb, gen);
    else if (V == 9) lk_bar_p<32>(bp, nb, gen);
    else if (V == 10) lk_bar_tree<4,  64>(leaf, &bp->arrived, nb, gen);
    else if (V == 11) lk_bar_tree<8,  64>(leaf, &bp->arrived, nb, gen);
    else if (V == 12) lk_bar_tree<16, 64>(leaf, &bp->arrived, nb, gen);
    else              lk_bar_tree<84, 64>(leaf, &bp->arrived, nb, gen);
}

#endif  // LK_MKBAR_CUH
