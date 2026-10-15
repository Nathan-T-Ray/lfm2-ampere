# liquid-kernel

A hand-written sm_86 CUDA inference engine for **LFM2.5-24B-A2B-QAD-Q4_0**, targeting
ludicrous single-stream decode speed on an RTX 3090 Ti — with **no quality loss**.

## Why this exists

`llama.cpp` is a general-purpose engine. On this model it reaches **227 tok/s**
(333 GB/s effective, **33% of the 1008 GB/s memory roofline**). A specialised
batch-1 engine can fuse aggressively, keep every intermediate on-device, capture
the whole 40-layer forward into one CUDA graph, and stream weights with
evict-first loads. The physical ceiling is **~600 tok/s**; the target is **≥550**.

## Facts established during recon (these correct earlier assumptions)

1. **The GGUF is 100% standard GGUF v3.** 428 tensors, `general.alignment=32`,
   data section at byte 8,250,912, tensor offsets perfectly contiguous, zero
   trailing bytes. Types: Q4_0 ×220, F32 ×207, and **Q6_K ×1** (`token_embd.weight`).
   There is no custom "type-14" format — ggml type 14 *is* Q6_K.
2. **QAD = Quantization-Aware Distillation**, a *retrained* checkpoint
   (`general.finetune="QAD"`, authored by Liquid AI). Its weights do **not** match
   the public `LiquidAI/LFM2.5-24B-A2B` BF16 checkpoint — measured token-embedding
   correlation is **-0.036**. The public checkpoint is therefore *not* a valid
   reference, and neither is any `hf_ref/` tensor dump.
3. **The only valid oracle is `llama.cpp` running this exact GGUF.**

## Roofline (batch-1, top-4 of 64 experts)

| Component | Bytes/token | Share |
|---|---|---|
| MoE experts (4-of-64) | 806.9 MB | 55.0% |
| shortconv projections ×30 | 264.2 MB | 18.0% |
| lm_head (tied Q6_K, full vocab) | 215.0 MB | 14.7% |
| dense FFN L0–1 | 100.3 MB | 6.8% |
| attention projections ×10 | 59.0 MB | 4.0% |
| MoE routers (F32) | 19.9 MB | 1.4% |
| norms + conv taps (F32) | 1.4 MB | 0.1% |
| **Total** | **1398.8 MiB** | |

Measured device limits (see `make devinfo`): L2 = 6.00 MiB,
`persistingL2CacheMaxSize` = 4.12 MiB, peak BW = 1008.1 GB/s.

## Architecture ground truth

Extracted from llama.cpp `src/models/lfm2.cpp` (which `lfm2moe.cpp` aliases):

- **RoPE: NEOX half-split**, `n_rot=64`, `theta=1e7`, `freq_scale=1`, no YaRN; applied to Q and K per head.
- **ShortConv ×30**: `in_proj → [B|C|x]` → `bx = B*x` → causal *unflipped* 3-tap conv → `y = C*conv_out` → `out_proj` → residual. **No non-linearity inside the block.** State = last two `B*x`, F32, `[tap, channel]`.
- **Router is SIGMOID** (`expert_gating_func=2`): top-4 by `sigmoid + exp_bias` (bias for **ranking only**), weights from the **unbiased** sigmoid, renormalised (`clamp(sum, 6.1e-5, ∞)`), scale 1.0.
- Q/K per-head RMSNorm(64) **before** RoPE; V unnormalised; the router consumes the `ffn_norm` output.
- SwiGLU `down(silu(gate(x))*up(x))`; attention iff `layer % 4 == 2`; dense FFN only at layers 0–1; no SWA; scale 0.125.

## Layout

```
setenv.sh            portable env (host + container); resolves the CUDA 12.6 tree
Makefile             nvcc-driven build (.RECIPEPREFIX set, no tab hazards)
scripts/oracle.sh    start|stop|status the llama.cpp oracle (pinned to GPU 1)
tools/devinfo.cu     GPU/L2 capability probe
tests/make_quant_fixture.py  builds a fixture from REAL pack blocks + references
tests/test_quant.cu  verifies the dot products (Gate G2)
scripts/setup-toolchain.sh   installs CCCL headers into the assembled CUDA tree
tools/oracle.py      oracle client (pure-greedy sampler config)
tools/make_suite.py  freezes the C1 parity suite as raw token IDs
tools/capture_ref.py captures oracle ground truth + baseline timings
```

The engine is pinned to **GPU 0**; the oracle owns **GPU 1**.

## Usage

```bash
source ./setenv.sh
make devinfo
scripts/oracle.sh start
python3 tools/capture_ref.py
```

## Status

- [x] **G0** repo, toolchain, oracle on GPU 1, baseline `227.07 tok/s`, L2 limits, parity suite
- [ ] **G1** repack, all 428 tensors bit-exact
- [x] **G1** repack, all 428 tensors bit-exact
- [x] **G2 (core)** Q4_0/Q6_K x Q8_1 dp4a dots verified vs exact integer reference
- [ ] **G2** remaining kernels (RMSNorm, RoPE, conv, router)
- [ ] **G3** C1/C2 — greedy parity + perplexity within 0.1%
- [ ] **G4** C3/C4 — ≥550 tok/s, ≥90% of roofline
- [ ] **G4b** routing-skew measurement → Phase 5 decision
- [ ] **G5** speculative decoding (conditional)
