#!/usr/bin/env bash
# liquid-kernel environment.  Usage:  source ./setenv.sh
#
# Every path below can be overridden by exporting it first, or by putting the
# exports in setenv.local.sh next to this file (git ignores it).  The defaults
# assume a stock CUDA install and keep downloads and generated files inside the
# repo (models/ and work/, both ignored by git).

export LK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Machine-specific overrides, if any.
if [ -f "$LK_ROOT/setenv.local.sh" ]; then source "$LK_ROOT/setenv.local.sh"; fi

# ---- locate the CUDA toolkit (12.x, needs nvcc) --------------------------
if [ -z "${CUDA_HOME:-}" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  CUDA_HOME=""
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  elif [ -x /usr/local/cuda/bin/nvcc ]; then
    CUDA_HOME=/usr/local/cuda
  fi
fi
export CUDA_HOME
if [ -z "$CUDA_HOME" ]; then
  echo "setenv.sh: ERROR: no nvcc found; install the CUDA toolkit or set CUDA_HOME" >&2
  return 1 2>/dev/null || exit 1
fi
case ":$PATH:" in *":$CUDA_HOME/bin:"*) ;; *) PATH="$CUDA_HOME/bin:$PATH" ;; esac
export PATH
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---- scratch + python ---------------------------------------------------
# LK_SCRATCH holds the converted weight pack (~13 GB) and oracle logs.
export LK_SCRATCH="${LK_SCRATCH:-$LK_ROOT/work}"
mkdir -p "$LK_SCRATCH"
if [ -n "${LK_VENV:-}" ] && [ -x "$LK_VENV/bin/python" ]; then
  export LK_PYTHON="$LK_VENV/bin/python"
  case ":$PATH:" in *":$LK_VENV/bin:"*) ;; *) PATH="$LK_VENV/bin:$PATH" ;; esac
  export PATH
else
  export LK_PYTHON="${LK_PYTHON:-$(command -v python3)}"
fi

# ---- project paths ------------------------------------------------------
export LK_MODEL="${LK_MODEL:-$LK_ROOT/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf}"
export LK_PACK="${LK_PACK:-$LK_SCRATCH/lfm25.pack}"
export LK_MANIFEST="${LK_MANIFEST:-$LK_SCRATCH/lfm25.manifest.json}"
export LK_REF="${LK_REF:-$LK_ROOT/ref}"

# ---- oracle (llama.cpp on the same GGUF; only needed for comparisons) ---
export LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
if [ -z "${LK_ORACLE:-}" ]; then
  if [ -x "$LLAMA_DIR/build/bin/llama-server" ]; then
    LK_ORACLE="$LLAMA_DIR/build/bin/llama-server"
  else
    LK_ORACLE="$(command -v llama-server || echo llama-server)"
  fi
fi
export LK_ORACLE
export LK_BACKEND_DIR="${LK_BACKEND_DIR:-$LLAMA_DIR/build/bin}"
export LK_ORACLE_LIBS="${LK_ORACLE_LIBS:-}"   # extra LD_LIBRARY_PATH for llama-server
export LK_ORACLE_PORT="${LK_ORACLE_PORT:-18099}"

# ---- GPUs ---------------------------------------------------------------
# The engine runs on LK_GPU.  The oracle defaults to the same card; on a
# two-GPU box set LK_ORACLE_GPU=1 so both can stay resident.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LK_GPU="${LK_GPU:-0}"
export LK_ORACLE_GPU="${LK_ORACLE_GPU:-$LK_GPU}"

echo "liquid-kernel env:"
echo "  LK_ROOT    = $LK_ROOT"
echo "  CUDA_HOME  = $CUDA_HOME"
echo "  LK_SCRATCH = $LK_SCRATCH"
echo "  LK_PYTHON  = $LK_PYTHON"
echo "  LK_MODEL   = $LK_MODEL"
echo "  LK_ORACLE  = $LK_ORACLE (GPU $LK_ORACLE_GPU :$LK_ORACLE_PORT)"
"$CUDA_HOME/bin/nvcc" --version | tail -2
