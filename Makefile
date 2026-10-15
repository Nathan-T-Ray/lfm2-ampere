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

all: devinfo test_quant test_kernels engine

# Gate G2: every kernel primitive verified against an independent reference.
test-all: test-quant test-kernels
> @echo ""
> @echo "GATE G2: all kernel tests passed"

clean:
> rm -rf $(BUILD)
