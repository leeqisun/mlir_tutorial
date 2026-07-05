#include <cstdint>
#include <cstdio>

#include "Utils/CUDARuntimeAPI.h"

using mlir::north_star::CUDAKernelDescriptor;
using mlir::north_star::CUDALaunchCallback;
using mlir::north_star::CUDARuntimeStatus;
using mlir::north_star::CUDATensorBindingF32;

namespace {

CUDARuntimeStatus mockLaunch(const CUDAKernelDescriptor &descriptor,
                             const CUDATensorBindingF32 *inputs,
                             int64_t numInputs,
                             CUDATensorBindingF32 *outputs,
                             int64_t numOutputs) {
  if (descriptor.descriptorId != 11 || descriptor.deviceId != 0 || numInputs != 1 ||
      numOutputs != 1)
    return CUDARuntimeStatus::kInvalidArgument;
  if (!inputs || !outputs || inputs[0].rank != 1 || outputs[0].rank != 1)
    return CUDARuntimeStatus::kInvalidArgument;

  auto *input = static_cast<float *>(inputs[0].data);
  auto *output = static_cast<float *>(outputs[0].data);
  output[0] = input[0] * 2.0f;
  output[1] = input[1] * 2.0f;
  return CUDARuntimeStatus::kSuccess;
}

}  // namespace

int main() {
  using namespace mlir::north_star;

  setCUDALaunchCallback(static_cast<CUDALaunchCallback>(mockLaunch));
  int32_t registerStatus = north_star_cuda_register_descriptor(
      11, "softmax_cuda_device_0", 0, "nvidia_rtx_5060_cuda", 1, 1);
  if (registerStatus != static_cast<int32_t>(CUDARuntimeStatus::kSuccess)) {
    std::fprintf(stderr, "register failed: %d\n", registerStatus);
    return 1;
  }

  float inputData[2] = {3.0f, 4.0f};
  float outputData[2] = {0.0f, 0.0f};
  int64_t sizes[1] = {2};
  int64_t strides[1] = {1};
  CUDATensorBindingF32 input{inputData, 1, sizes, strides};
  CUDATensorBindingF32 output{outputData, 1, sizes, strides};

  int32_t launchStatus =
      north_star_cuda_launch_descriptor_f32(11, &input, 1, &output, 1);
  if (launchStatus != static_cast<int32_t>(CUDARuntimeStatus::kSuccess)) {
    std::fprintf(stderr, "launch failed: %d\n", launchStatus);
    return 1;
  }

  std::printf("descriptor=%d output=%0.1f,%0.1f\n", 11, outputData[0], outputData[1]);
  return 0;
}
