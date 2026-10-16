# liquid-kernel - hand-written sm_86 inference engine for LFM2.5-24B-A2B-QAD
# GNU make; .RECIPEPREFIX is set to '>' so recipes never depend on literal tabs.
.RECIPEPREFIX = >
.DEFAULT_GOAL := help

CUDA_HOME ?= $(shell for c in $$HOME/local-agent-harnesses/model-scratch/cuda-12.6/usr/local/cuda-12.6 /model-scratch/cuda-12.6/usr/local/cuda-12.6 /usr/local/cuda; do [ -x $$c/bin/nvcc ] && echo $$c && break; done)
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := -gencode arch=compute_86,code=sm_86
# Sweepable tuning constants.
#
# LK_ATTN_SPLIT is the attention context split CAP (the operating point is
# derived at runtime from the sequence length; see kernels.cuh).  It MUST be swept through this
# variable rather than by editing kernels.cuh, because a sweep that edits a
# source constant and then fails to restore it silently invalidates every
# measurement taken afterwards -- that happened once already with LK_GEMV_ROWS
# and cost hours.  `make SPLIT=n engine` either rebuilds with n or fails loudly.
SPLIT     ?= 128
TARGET_N  ?= 64
SPLIT_MIN ?= 8
# Phase 4x fusion switches.  FUSE_NORM=1 was measured 1.6% SLOWER on
# interleaved A/B and is off; see the note in kernels.cuh.  The switch is kept
# so the negative result stays reproducible with one flag.
FUSE_NORM ?= 0
FUSE_ROUTER ?= 0
# T4r: rebuild the router top-4 in the consumer's prologue instead of its own
# node.  MEASURED AND REJECTED at -0.17%; default off, kept so the negative
# result stays reproducible.
ROUTER_PRO ?= 0
# Phase 4y: fold RMSNorm into the kernel that produces its input, via the
# scale-invariance of Q8_1 (see kernels.cuh).  Two halves, measured separately:
#   ACCUM  attn_norm(L>=3) + norm_final into k_moe_accum4 -- no shape change
#   GEMV   the other 42 sites                              -- requires BIGBLOCK
#
# These are ON because they WIN: +5.80/+5.33/+5.05% on interleaved A/B at ctx
# 272/2128/8128, C2 0.0140% against a 0.1% gate, and an end-to-end nll that
# moves by less than the corpus can resolve.  (Phase 4x's FUSE_NORM and
# FUSE_ROUTER stay off because they lost; a losing switch keeps its flag so the
# negative result stays reproducible, a winning one becomes the default.)
#
# `make BIGBLOCK=0 NORM_ACCUM=0 NORM_GEMV=0` reproduces the pre-Phase-4y build,
# and was verified bit-identical to it on all 7 trajectory cases.
BIGBLOCK   ?= 1
NORM_ACCUM ?= 1
NORM_GEMV  ?= 1
# Diagnostic ablations.  ABLATE=0 is the only correct build; see kernels.cuh.
ABLATE    ?= 0
NVFLAGS   := -O3 -std=c++17 -lineinfo $(ARCH) -Xcompiler -Wall -Xcompiler -Wextra \
             -I$(CURDIR)/src -DLK_ATTN_SPLIT=$(SPLIT) -DLK_ATTN_ABLATE=$(ABLATE) \
             -DLK_ATTN_TARGET_N=$(TARGET_N) -DLK_ATTN_SPLIT_MIN=$(SPLIT_MIN) \
             -DLK_FUSE_NORM=$(FUSE_NORM) -DLK_FUSE_ROUTER=$(FUSE_ROUTER) \
             -DLK_ROUTER_PRO=$(ROUTER_PRO) \
             -DLK_GEMV_BIGBLOCK=$(BIGBLOCK) \
             -DLK_NORM_PROD_ACCUM=$(NORM_ACCUM) -DLK_NORM_PROD_GEMV=$(NORM_GEMV)
LDFLAGS   := -L$(CUDA_HOME)/lib64 -lcudart -lm
BUILD     := build

.PHONY: help all check-cuda devinfo test-all clean

help:
> @echo "liquid-kernel targets:"
> @echo "  make devinfo   - build+run tools/devinfo (GPU/L2 properties)  [Gate G0]"
> @echo "  make all       - build everything"
> @echo "  make clean"

check-cuda:
> @test -x "$(NVCC)" || { echo "ERROR: nvcc not found at $(NVCC)"; exit 1; }
> @echo "nvcc: $(NVCC)"

$(BUILD):
> @mkdir -p $(BUILD)

# ---------------------------------------------------------------------------
# Flag-change detection.
#
# GNU make does not rebuild when a -D value changes, because nothing on disk
# changed.  That is not a theoretical hazard here: an ablation sweep that
# appeared to show "attention costs nothing at 8k context" was in fact running
# the SAME binary four times, and an earlier sweep left LK_GEMV_ROWS at 4 for
# hours of measurements.  This stamp makes a flag change a real dependency, so
# `make SPLIT=16` either rebuilds or fails; it can never silently reuse objects
# built with different flags.
FORCE:
$(BUILD)/flags.stamp: FORCE | $(BUILD)
> @printf '%s\n' '$(NVFLAGS)' '$(LDFLAGS)' > $@.new
> @cmp -s $@.new $@ || { mv $@.new $@; echo "build flags changed -> rebuilding objects"; }
> @rm -f $@.new

$(BUILD)/devinfo: tools/devinfo.cu $(BUILD)/flags.stamp | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) $< -o $@ $(LDFLAGS)

devinfo: $(BUILD)/devinfo
> @echo "--- device properties (CUDA_VISIBLE_DEVICES=$(LK_GPU)) ---"
> @CUDA_VISIBLE_DEVICES=$(LK_GPU) ./$(BUILD)/devinfo

$(BUILD)/test_quant: tests/test_quant.cu src/quant.cuh $(BUILD)/flags.stamp | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) tests/test_quant.cu -o $@ $(LDFLAGS)

test-quant: $(BUILD)/test_quant
> @CUDA_VISIBLE_DEVICES=$(LK_GPU) ./$(BUILD)/test_quant $(LK_REF)/quant_fixture.bin

$(BUILD)/kernels.o: src/kernels.cu src/kernels.cuh src/quant.cuh $(BUILD)/flags.stamp | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) -c src/kernels.cu -o $@

$(BUILD)/test_kernels: tests/test_kernels.cu $(BUILD)/kernels.o $(BUILD)/flags.stamp | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) tests/test_kernels.cu $(BUILD)/kernels.o -o $@ $(LDFLAGS)

test-kernels: $(BUILD)/test_kernels
> @CUDA_VISIBLE_DEVICES=$(LK_GPU) ./$(BUILD)/test_kernels $(LK_REF)/kernel_fixture.bin

$(BUILD)/test_attn: tests/test_attn.cu $(BUILD)/kernels.o $(BUILD)/flags.stamp | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) tests/test_attn.cu $(BUILD)/kernels.o -o $@ $(LDFLAGS)

test-attn: $(BUILD)/test_attn
> @CUDA_VISIBLE_DEVICES=$(LK_GPU) ./$(BUILD)/test_attn

$(BUILD)/test_normfuse: tests/test_normfuse.cu $(BUILD)/kernels.o $(BUILD)/flags.stamp | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) tests/test_normfuse.cu $(BUILD)/kernels.o -o $@ $(LDFLAGS)

test-normfuse: $(BUILD)/test_normfuse
> @CUDA_VISIBLE_DEVICES=$(LK_GPU) ./$(BUILD)/test_normfuse

$(BUILD)/test_normscale: tests/test_normscale.cu $(BUILD)/kernels.o $(BUILD)/flags.stamp | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) tests/test_normscale.cu $(BUILD)/kernels.o -o $@ $(LDFLAGS)

test-normscale: $(BUILD)/test_normscale
> @CUDA_VISIBLE_DEVICES=$(LK_GPU) ./$(BUILD)/test_normscale

$(BUILD)/engine: src/engine.cu $(BUILD)/kernels.o src/tensor_table.h $(BUILD)/flags.stamp | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) src/engine.cu $(BUILD)/kernels.o -o $@ $(LDFLAGS)

engine: $(BUILD)/engine

# --- oracle tools -----------------------------------------------------------
# These link libllama directly.  The HTTP API only exposes top-k logprobs, so
# full-vocabulary logits and a whole-corpus teacher-forced pass both require
# calling into the library.  LLAMA_DIR only has to point at a llama.cpp build
# that produced versioned .so files.
LLAMA_DIR ?= $(HOME)/.unsloth/llama.cpp
LLAMA_BIN := $(LLAMA_DIR)/build/bin

$(BUILD)/oracle_logits: tools/oracle_logits.cpp | $(BUILD)
> g++ -O2 -std=c++17 $< -o $@ -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include \
>   $(LLAMA_BIN)/libllama.so.0 $(LLAMA_BIN)/libggml.so.0 $(LLAMA_BIN)/libggml-base.so.0 \
>   -Wl,-rpath,$(LLAMA_BIN)

$(BUILD)/oracle_ppl: tools/oracle_ppl.cpp | $(BUILD)
> g++ -O2 -std=c++17 $< -o $@ -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include \
>   $(LLAMA_BIN)/libllama.so.0 $(LLAMA_BIN)/libggml.so.0 $(LLAMA_BIN)/libggml-base.so.0 \
>   -Wl,-rpath,$(LLAMA_BIN) -lm

oracle-tools: $(BUILD)/oracle_logits $(BUILD)/oracle_ppl

$(BUILD)/bench_launch: tests/bench_launch.cu $(BUILD)/flags.stamp | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) $< -o $@ $(LDFLAGS)

$(BUILD)/bench_norm: tests/bench_norm.cu $(BUILD)/kernels.o $(BUILD)/flags.stamp | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) -I src tests/bench_norm.cu $(BUILD)/kernels.o -o $@ $(LDFLAGS)

$(BUILD)/bench_small: tests/bench_small.cu $(BUILD)/kernels.o $(BUILD)/flags.stamp | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) -I src tests/bench_small.cu $(BUILD)/kernels.o -o $@ $(LDFLAGS)

# True marginal cost of the small kernels, and a phase ablation of the norm.
# The per-kernel profiler adds ~2.1 us of event instrument to every reading,
# which is larger than several of these kernels; these measure back to back
# under one event pair instead.
bench-norm: $(BUILD)/bench_norm
> @CUDA_VISIBLE_DEVICES=$(LK_GPU) ./$(BUILD)/bench_norm

bench-small: $(BUILD)/bench_small
> @CUDA_VISIBLE_DEVICES=$(LK_GPU) ./$(BUILD)/bench_small

$(BUILD)/bench_ovh: tests/bench_ovh.cu $(BUILD)/flags.stamp | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) $< -o $@ $(LDFLAGS)

# The per-launch fixed cost, and whether a small grid still saturates DRAM.
# The engine's profile table shows GB/s falling monotonically with kernel size
# (925 at 215 MB/call, 728 at 14 MB, 620 at 7 MB, 431 at 3.5 MB, 110 at 0.5 MB).
# This decides whether that is a real size-dependent efficiency or one constant
# added per launch -- and therefore whether there is any headroom to recover.
# It also gates the 64-block x 1024-thread shapes the planned fusions require.
bench-ovh: $(BUILD)/bench_ovh
> @CUDA_VISIBLE_DEVICES=$(LK_GPU) ./$(BUILD)/bench_ovh

# Gate G3: launch cost, event-bracket cost and graph-replay cost.  Run this
# before believing any per-kernel profile.
bench-launch: $(BUILD)/bench_launch
> @CUDA_VISIBLE_DEVICES=$(LK_GPU) ./$(BUILD)/bench_launch

# --- gates ------------------------------------------------------------------
# C1  greedy parity vs llama.cpp
# C1a step-1 top-1 agreement + full-vocabulary correlation
# C2  teacher-forced perplexity vs exact float64 mathematics
# G4b routing skew, which gates Phase 5 speculative decoding
gate-c1: engine
> @python3 tools/parity.py

gate-c1a: engine oracle-tools
> @python3 tools/c1a.py

gate-c2: engine oracle-tools
> @python3 tools/perplexity.py ppl_natural.bin 200

all: devinfo test_quant test_kernels test_attn test_normfuse test_normscale engine oracle-tools

# Gate G2: every kernel primitive verified against an independent reference.
test-all: test-quant test-kernels
> @echo ""
> @echo "GATE G2: all kernel tests passed"

clean:
> rm -rf $(BUILD)
