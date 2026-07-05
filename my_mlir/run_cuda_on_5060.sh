#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
NS_OPT="${BUILD_DIR}/my_mlir/src/NS-opt/NS-opt"
RUNNER_BIN="${BUILD_DIR}/northstar-cuda-runner"
PTX_SCRIPT="${ROOT_DIR}/my_mlir/run_cuda_pipeline.sh"
JSON_EXTRACT="${ROOT_DIR}/my_mlir/extract_runtime_json.py"

INPUT_MLIR="${1:-${ROOT_DIR}/my_mlir/test/Conversion/device_kernels_to_gpu_loops.mlir}"
DESCRIPTOR_ID="${2:-0}"
DIM0="${3:-2}"
DIM1="${4:-2}"
DIM2="${5:-2}"
shift $(( $# >= 5 ? 5 : $# ))

DEFAULT_VALUES=(1 2 0 1 3 0.5 1 -1)
VALUES=("$@")
if [[ ${#VALUES[@]} -eq 0 ]]; then
  VALUES=("${DEFAULT_VALUES[@]}")
fi

RUNTIME_JSON_MLIR="/tmp/northstar_runtime_bundle.mlir"
RUNTIME_JSON="/tmp/northstar_runtime.json"
PTX_OUT="/tmp/northstar_kernel.ptx"

if [[ ! -x "${NS_OPT}" ]]; then
  echo "missing NS-opt: ${NS_OPT}" >&2
  exit 1
fi

if [[ ! -x "${RUNNER_BIN}" ]]; then
  echo "missing runner binary: ${RUNNER_BIN}" >&2
  echo "build first: ninja -C ${BUILD_DIR} northstar-cuda-runner" >&2
  exit 1
fi

if [[ ! -x "${PTX_SCRIPT}" || ! -x "${JSON_EXTRACT}" ]]; then
  echo "missing helper scripts" >&2
  exit 1
fi

echo "[1/4] Generate PTX from GPU pipeline"
"${PTX_SCRIPT}" "${INPUT_MLIR}" /tmp/northstar_gpu.mlir /tmp/northstar_nvvm.mlir \
  /tmp/northstar_gpu_pipeline.log /tmp/northstar_full_pipeline.mlir "${PTX_OUT}"

echo "[2/4] Generate runtime bridge JSON"
"${NS_OPT}" "${INPUT_MLIR}" \
  --convert-north-satr-to-linalg \
  --outline-north-star-device-kernels \
  --lower-north-star-device-kernels-to-loops \
  --lower-north-star-host-to-npu-calls \
  --lower-north-star-npu-calls-to-runtime \
  --lower-north-star-runtime-stubs-to-cuda-calls \
  --generate-north-star-runtime-json \
  | sed '/^initializing north_star$/d;/^register north_star /d;/^destroying north_star$/d' \
  > "${RUNTIME_JSON_MLIR}"

echo "[3/4] Extract runtime JSON payload"
"${JSON_EXTRACT}" "${RUNTIME_JSON_MLIR}" "${RUNTIME_JSON}"
echo "wrote ${RUNTIME_JSON}"

echo "[4/4] Run descriptor ${DESCRIPTOR_ID} on RTX 5060"
"${RUNNER_BIN}" "${RUNTIME_JSON}" "${PTX_OUT}" "${DESCRIPTOR_ID}" \
  "${DIM0}" "${DIM1}" "${DIM2}" "${VALUES[@]}" --device-override 0
