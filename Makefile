# liquid-kernel - hand-written sm_86 inference engine for LFM2.5-24B-A2B-QAD
# GNU make; .RECIPEPREFIX is set to '>' so recipes never depend on literal tabs.
.RECIPEPREFIX = >
.DEFAULT_GOAL := help

CUDA_HOME ?= $(shell for c in $$HOME/local-agent-harnesses/model-scratch/cuda-12.6/usr/local/cuda-12.6 /model-scratch/cuda-12.6/usr/local/cuda-12.6 /usr/local/cuda; do [ -x $$c/bin/nvcc ] && echo $$c && break; done)
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := -gencode arch=compute_86,code=sm_86
NVFLAGS   := -O3 -std=c++17 -lineinfo $(ARCH) -Xcompiler -Wall -Xcompiler -Wextra -I$(CURDIR)/src
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

$(BUILD)/devinfo: tools/devinfo.cu | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) $< -o $@ $(LDFLAGS)

devinfo: $(BUILD)/devinfo
> @echo "--- device properties (CUDA_VISIBLE_DEVICES=$(LK_GPU)) ---"
> @CUDA_VISIBLE_DEVICES=$(LK_GPU) ./$(BUILD)/devinfo

$(BUILD)/test_quant: tests/test_quant.cu src/quant.cuh | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) tests/test_quant.cu -o $@ $(LDFLAGS)

test-quant: $(BUILD)/test_quant
> @CUDA_VISIBLE_DEVICES=$(LK_GPU) ./$(BUILD)/test_quant $(LK_REF)/quant_fixture.bin

$(BUILD)/kernels.o: src/kernels.cu src/kernels.cuh src/quant.cuh | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) -c src/kernels.cu -o $@

$(BUILD)/test_kernels: tests/test_kernels.cu $(BUILD)/kernels.o | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) tests/test_kernels.cu $(BUILD)/kernels.o -o $@ $(LDFLAGS)

test-kernels: $(BUILD)/test_kernels
> @CUDA_VISIBLE_DEVICES=$(LK_GPU) ./$(BUILD)/test_kernels $(LK_REF)/kernel_fixture.bin

$(BUILD)/engine: src/engine.cu $(BUILD)/kernels.o src/tensor_table.h | $(BUILD) check-cuda
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

$(BUILD)/bench_launch: tests/bench_launch.cu | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) $< -o $@ $(LDFLAGS)

$(BUILD)/bench_norm: tests/bench_norm.cu $(BUILD)/kernels.o | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) -I src tests/bench_norm.cu $(BUILD)/kernels.o -o $@ $(LDFLAGS)

$(BUILD)/bench_small: tests/bench_small.cu $(BUILD)/kernels.o | $(BUILD) check-cuda
> $(NVCC) $(NVFLAGS) -I src tests/bench_small.cu $(BUILD)/kernels.o -o $@ $(LDFLAGS)

# True marginal cost of the small kernels, and a phase ablation of the norm.
# The per-kernel profiler adds ~2.1 us of event instrument to every reading,
# which is larger than several of these kernels; these measure back to back
# under one event pair instead.
bench-norm: $(BUILD)/bench_norm
> @CUDA_VISIBLE_DEVICES=$(LK_GPU) ./$(BUILD)/bench_norm

bench-small: $(BUILD)/bench_small
> @CUDA_VISIBLE_DEVICES=$(LK_GPU) ./$(BUILD)/bench_small

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

all: devinfo test_quant test_kernels engine oracle-tools

# Gate G2: every kernel primitive verified against an independent reference.
test-all: test-quant test-kernels
> @echo ""
> @echo "GATE G2: all kernel tests passed"

clean:
> rm -rf $(BUILD)
