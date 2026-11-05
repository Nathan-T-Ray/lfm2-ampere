#!/usr/bin/env bash
# Complete the hand-assembled CUDA 12.6 tree with the CCCL headers.
#
# cuda_fp16.h includes <nv/target>, which lives in CCCL.  The assembled tree
# (nvcc + cudart + cuda-nvvm + cuda-crt debs) does not ship CCCL, so without
# this every translation unit that includes cuda_fp16.h fails with
#   fatal error: nv/target: No such file or directory
#
# A normal CUDA install already has CCCL; this is only for a toolkit assembled
# from individual packages.  It copies a version-matched CCCL include tree (for
# example from a conda cuda-cccl package) into the CUDA include dir.
#
#   usage: scripts/setup-toolchain.sh CCCL_INCLUDE_DIR [CUDA_INCLUDE_DIR]
set -euo pipefail
CCCL="${1:?usage: setup-toolchain.sh CCCL_INCLUDE_DIR [CUDA_INCLUDE_DIR]}"
CUDA_INC="${2:-${CUDA_HOME:?set CUDA_HOME or pass CUDA_INCLUDE_DIR}/include}"

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
