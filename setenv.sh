#!/usr/bin/env bash
# liquid-kernel environment.  Usage:  source ./setenv.sh
# Resolves paths so the same script works on the host (think-tank) and inside
# the DSH lab container, where the CUDA tree is bind-mounted at /model-scratch.

export LK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- locate the hand-assembled CUDA 12.6 toolkit ------------------------
_lk_cands=(
  "$HOME/local-agent-harnesses/model-scratch/cuda-12.6/usr/local/cuda-12.6"
  "/model-scratch/cuda-12.6/usr/local/cuda-12.6"
  "/usr/local/cuda"
)
export CUDA_HOME=""
for _c in "${_lk_cands[@]}"; do
  if [ -x "$_c/bin/nvcc" ]; then CUDA_HOME="$_c"; break; fi
done
unset _lk_cands _c
if [ -z "$CUDA_HOME" ]; then
  echo "setenv.sh: ERROR: no nvcc found" >&2
  return 1 2>/dev/null || exit 1
fi
case ":$PATH:" in *":$CUDA_HOME/bin:"*) ;; *) PATH="$CUDA_HOME/bin:$PATH" ;; esac
export PATH
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---- scratch + python ---------------------------------------------------
export LK_SCRATCH="${LK_SCRATCH:-$HOME/local-agent-harnesses/model-scratch/lfm25}"
if [ -d /model-scratch ]; then export LK_SCRATCH_CTR="/model-scratch/lfm25"; fi
_lk_venv="${LK_VENV:-$HOME/local-agent-harnesses/model-scratch/venv}"
if [ -x "$_lk_venv/bin/python" ]; then
  export LK_PYTHON="$_lk_venv/bin/python"
  case ":$PATH:" in *":$_lk_venv/bin:"*) ;; *) PATH="$_lk_venv/bin:$PATH" ;; esac
  export PATH
else
  export LK_PYTHON="$(command -v python3)"
fi
unset _lk_venv

# ---- project paths ------------------------------------------------------
export LK_MODEL="${LK_MODEL:-$HOME/qwen-serving/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf}"
export LK_PACK="${LK_PACK:-$LK_SCRATCH/lfm25.pack}"
export LK_MANIFEST="${LK_MANIFEST:-$LK_SCRATCH/lfm25.manifest.json}"
export LK_REF="${LK_REF:-$LK_ROOT/ref}"

# ---- oracle (llama.cpp on this exact QAD GGUF) --------------------------
export LK_ORACLE="${LK_ORACLE:-$HOME/.unsloth/llama.cpp/build/bin/llama-server}"
export LK_OLLAMA_CUDA="${LK_OLLAMA_CUDA:-/usr/local/lib/ollama/cuda_v13}"
export LK_ORACLE_GPU=1
export LK_ORACLE_PORT="${LK_ORACLE_PORT:-18099}"

# ---- engine GPU (0 = free; oracle owns 1) -------------------------------
export LK_GPU=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "liquid-kernel env:"
echo "  LK_ROOT    = $LK_ROOT"
echo "  CUDA_HOME  = $CUDA_HOME"
echo "  LK_SCRATCH = $LK_SCRATCH"
echo "  LK_PYTHON  = $LK_PYTHON"
echo "  LK_MODEL   = $LK_MODEL"
echo "  LK_ORACLE  = $LK_ORACLE (GPU $LK_ORACLE_GPU :$LK_ORACLE_PORT)"
"$CUDA_HOME/bin/nvcc" --version | tail -2
