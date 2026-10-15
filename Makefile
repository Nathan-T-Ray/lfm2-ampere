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

.PHONY: help all check-cuda devinfo clean

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

all: devinfo test_quant

clean:
> rm -rf $(BUILD)
