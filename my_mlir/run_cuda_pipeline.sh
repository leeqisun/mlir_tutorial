#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
NS_OPT="${BUILD_DIR}/my_mlir/src/NS-opt/NS-opt"
MLIR_OPT="${ROOT_DIR}/install/bin/mlir-opt"

INPUT_MLIR="${1:-${ROOT_DIR}/my_mlir/test/Conversion/device_kernels_to_gpu_loops.mlir}"
GPU_IR="${2:-/tmp/northstar_gpu.mlir}"
NVVM_IR="${3:-/tmp/northstar_nvvm.mlir}"
PIPELINE_LOG="${4:-/tmp/northstar_gpu_pipeline.log}"
FULL_PIPELINE_IR="${5:-/tmp/northstar_full_pipeline.mlir}"
PTX_OUT="${6:-/tmp/northstar_kernel.ptx}"
PTX_EXTRACT="${ROOT_DIR}/my_mlir/extract_cuda_ptx.py"

if [[ ! -x "${NS_OPT}" ]]; then
  echo "missing NS-opt: ${NS_OPT}" >&2
  echo "build first: cmake --build ${BUILD_DIR} --target NS-opt" >&2
  exit 1
fi

if [[ ! -x "${MLIR_OPT}" ]]; then
  echo "missing mlir-opt: ${MLIR_OPT}" >&2
  exit 1
fi

if [[ ! -f "${INPUT_MLIR}" ]]; then
  echo "missing input mlir: ${INPUT_MLIR}" >&2
  exit 1
fi

if [[ ! -x "${PTX_EXTRACT}" ]]; then
  echo "missing PTX extractor: ${PTX_EXTRACT}" >&2
  exit 1
fi

find_libdevice() {
  local candidates=(
    "${CUDA_HOME:-}/nvvm/libdevice/libdevice.10.bc"
    "${CUDA_PATH:-}/nvvm/libdevice/libdevice.10.bc"
    "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc"
    "/usr/lib/cuda/nvvm/libdevice/libdevice.10.bc"
    "/usr/nvvm/libdevice/libdevice.10.bc"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" && -f "${candidate}" ]] && {
      echo "${candidate}"
      return 0
    }
  done
  return 1
}

toolkit_root_from_libdevice() {
  local libdevice_path="$1"
  local toolkit_root
  toolkit_root="$(dirname "$(dirname "$(dirname "${libdevice_path}")")")"
  if [[ -d "${toolkit_root}" ]]; then
    echo "${toolkit_root}"
    return 0
  fi
  return 1
}

echo "[1/3] Lower north_star device kernels to gpu"
"${NS_OPT}" "${INPUT_MLIR}" \
  --convert-north-satr-to-linalg \
  --outline-north-star-device-kernels \
  --lower-north-star-device-kernels-to-loops \
  --lower-north-star-device-kernels-to-gpu \
  | sed '/^initializing north_star$/d;/^register north_star /d;/^destroying north_star$/d' \
  > "${GPU_IR}"
echo "wrote ${GPU_IR}"

echo "[2/3] Lower gpu kernels to NVVM/LLVM"
"${MLIR_OPT}" "${GPU_IR}" \
  --allow-unregistered-dialect \
  '--pass-pipeline=builtin.module(nvvm-attach-target{chip=sm_120 O=3 verify-target-arch=false},gpu.module(strip-debuginfo,convert-gpu-to-nvvm,convert-index-to-llvm{index-bitwidth=32},canonicalize,cse))' \
  > "${NVVM_IR}"
echo "wrote ${NVVM_IR}"

echo "[3/3] Try full gpu-lower-to-nvvm-pipeline"
if LIBDEVICE_PATH="$(find_libdevice)"; then
  echo "found libdevice: ${LIBDEVICE_PATH}"
  if ! TOOLKIT_ROOT="$(toolkit_root_from_libdevice "${LIBDEVICE_PATH}")"; then
    echo "failed to infer CUDA toolkit root from ${LIBDEVICE_PATH}" >&2
    exit 2
  fi
  echo "using CUDA toolkit root: ${TOOLKIT_ROOT}"
  if CUDA_HOME="${TOOLKIT_ROOT}" CUDA_PATH="${TOOLKIT_ROOT}" \
      "${MLIR_OPT}" "${GPU_IR}" \
      --allow-unregistered-dialect \
      '--gpu-lower-to-nvvm-pipeline=cubin-format=isa cubin-chip=sm_120' \
      > "${FULL_PIPELINE_IR}" 2> "${PIPELINE_LOG}"; then
    if rg -n "error:" "${FULL_PIPELINE_IR}" "${PIPELINE_LOG}" >/dev/null 2>&1; then
      echo "full pipeline emitted diagnostics; see ${PIPELINE_LOG} and ${FULL_PIPELINE_IR}" >&2
      exit 2
    fi
    echo "full pipeline succeeded"
    echo "wrote ${FULL_PIPELINE_IR}"
    echo "log: ${PIPELINE_LOG}"
    "${PTX_EXTRACT}" "${FULL_PIPELINE_IR}" "${PTX_OUT}"
    echo "wrote ${PTX_OUT}"
  else
    echo "full pipeline failed; see ${PIPELINE_LOG}" >&2
    cat "${PIPELINE_LOG}" >&2
    exit 2
  fi
else
  echo "full pipeline skipped: libdevice.10.bc not found" >&2
  echo "checked CUDA_HOME/CUDA_PATH and common system CUDA locations" >&2
  echo "manual NVVM IR is still available at ${NVVM_IR}" >&2
fi
