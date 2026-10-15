#!/usr/bin/env bash
# Complete the hand-assembled CUDA 12.6 tree with the CCCL headers.
#
# cuda_fp16.h includes <nv/target>, which lives in CCCL.  The assembled tree
# (nvcc + cudart + cuda-nvvm + cuda-crt debs) does not ship CCCL, so without
# this every translation unit that includes cuda_fp16.h fails with
#   fatal error: nv/target: No such file or directory
#
# We copy the version-matched CCCL 12.6.77 (same minor as cudart 12.6.77) out of
# the conda package cache into the CUDA include dir, so the toolchain becomes
# self-contained and also works through the container's /model-scratch mount.
set -euo pipefail
CUDA_INC="${1:-$HOME/local-agent-harnesses/model-scratch/cuda-12.6/usr/local/cuda-12.6/include}"
CCCL="$HOME/miniconda3/pkgs/cuda-cccl_linux-64-12.6.77-ha770c72_0/targets/x86_64-linux/include"

if [ -d "$CUDA_INC/nv" ] && [ -d "$CUDA_INC/cuda" ]; then
  echo "CCCL already present in $CUDA_INC"; exit 0
fi
[ -d "$CCCL" ] || { echo "ERROR: CCCL source not found at $CCCL" >&2; exit 1; }
[ -d "$CUDA_INC" ] || { echo "ERROR: CUDA include dir not found at $CUDA_INC" >&2; exit 1; }
for d in nv cuda; do
  cp -r "$CCCL/$d" "$CUDA_INC/"
  echo "installed $d/ -> $CUDA_INC/$d"
done
echo "toolchain complete"
