#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
PIPELINE_SCRIPT="${ROOT_DIR}/my_mlir/run_cuda_pipeline.sh"
SMOKE_BIN="${BUILD_DIR}/cuda-generated-ptx-smoke"
INPUT_MLIR="${1:-${ROOT_DIR}/my_mlir/test/Conversion/device_kernels_to_gpu_loops.mlir}"
PTX_OUT="${2:-/tmp/northstar_kernel.ptx}"

if [[ ! -x "${PIPELINE_SCRIPT}" ]]; then
  echo "missing pipeline script: ${PIPELINE_SCRIPT}" >&2
  exit 1
fi

if [[ ! -x "${SMOKE_BIN}" ]]; then
  echo "missing smoke binary: ${SMOKE_BIN}" >&2
  echo "build first: ninja -C ${BUILD_DIR} cuda-generated-ptx-smoke" >&2
  exit 1
fi

"${PIPELINE_SCRIPT}" "${INPUT_MLIR}" /tmp/northstar_gpu.mlir /tmp/northstar_nvvm.mlir \
  /tmp/northstar_gpu_pipeline.log /tmp/northstar_full_pipeline.mlir "${PTX_OUT}"

echo "[4/4] Launch generated PTX with CUDA driver backend"
"${SMOKE_BIN}" "${PTX_OUT}"
