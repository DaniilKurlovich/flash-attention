#!/usr/bin/env bash
# Profile the CUDA attention kernel with Nsight Compute (ncu).
#
# Usage:
#   ./bench/profile_cuda.sh [ncu-extra-args]
#
# Prerequisites:
#   - Build with CUDA: cmake -S . -B build -DFLASH_ATTENTION_ENABLE_CUDA=ON && cmake --build build
#   - ncu must be in PATH (comes with CUDA toolkit, e.g. /usr/local/cuda/bin/ncu)
#   - If you get ERR_NVGPUCTRPERM, run once with sudo or fix permissions:
#       sudo chmod 666 /dev/nvidia-caps/nvidia-cap1 /dev/nvidia-caps/nvidia-cap2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_EXE="${REPO_ROOT}/.venv/bin/python"
PYTHON_VER="$(${PYTHON_EXE} -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"

# Prefer build/ if it contains a working CUDA .so for the venv python,
# otherwise fall back to build-cuda/.
BUILD_DIR="${REPO_ROOT}/build-cuda"

NCU_BIN="/usr/local/cuda-12/bin/ncu"

PYTHONPATH="${BUILD_DIR}/python:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH

# KERNEL_NAME="attention_tiled_online_softmax_kernel_stub_v2"
KERNEL_NAME="attention_tiled_online_softmax_kernel_stub_v3"

# Base ncu flags:
#   --kernel-name      only profile our kernel (ignore PyTorch internals)
#   --full             collect all default Nsight Compute sections
#   --import-source    embed source code into the report so the UI can show it
#   -o                 write report file for Nsight Compute UI
NCU_FLAGS=(
    "--kernel-name" "regex:${KERNEL_NAME}"
    "--import-source" "1"
    "--launch-skip" "10"
    "--launch-count" "1"
    "--set" "full"
    "--source-folders" "${REPO_ROOT}/src/csrc"
    "-o" "${REPO_ROOT}/ncu_report"
    "-f"
)

# Allow user to pass extra ncu args (e.g. --device 0)
EXTRA_ARGS=("${@}")

echo "==> PYTHONPATH=${PYTHONPATH}"
echo "==> Build dir: ${BUILD_DIR}"
echo "==> Profiling kernel: ${KERNEL_NAME}"
echo "==> Launching ncu..."

"${NCU_BIN}" "${NCU_FLAGS[@]}" "${EXTRA_ARGS[@]}" \
    "${PYTHON_EXE}" - <<'PYEOF'
import sys, torch
from src.flash_attention import attention_tiled_online_softmax_cpp

torch.cuda.synchronize()
q = torch.randn(4, 8, 1024, 64, dtype=torch.bfloat16, device="cuda")
k = torch.randn(4, 8, 1024, 64, dtype=torch.bfloat16, device="cuda")
v = torch.randn(4, 8, 1024, 64, dtype=torch.bfloat16, device="cuda")

# Warmup
for _ in range(10):
    out = attention_tiled_online_softmax_cpp(q, k, v)
torch.cuda.synchronize()

# Profiled iteration
out = attention_tiled_online_softmax_cpp(q, k, v)
torch.cuda.synchronize()
print("done", out.shape)
PYEOF

echo "==> Report written to ${REPO_ROOT}/ncu_report.ncu-rep"
echo "==> Open with: ncu-ui ${REPO_ROOT}/ncu_report.ncu-rep"
