#include <cstdint>
#include <cstdio>

#include "Utils/H350RuntimeAPI.h"

using mlir::north_star::H350KernelDescriptor;
using mlir::north_star::H350LaunchCallback;
using mlir::north_star::H350RuntimeStatus;
using mlir::north_star::H350TensorBindingF32;

namespace {

H350RuntimeStatus mockLaunch(const H350KernelDescriptor &descriptor,
                             const H350TensorBindingF32 *inputs,
                             int64_t numInputs,
                             H350TensorBindingF32 *outputs,
                             int64_t numOutputs) {
  if (descriptor.descriptorId != 7 || descriptor.deviceId != 1 || numInputs != 1 ||
      numOutputs != 1)
    return H350RuntimeStatus::kInvalidArgument;
  if (!inputs || !outputs || inputs[0].rank != 1 || outputs[0].rank != 1)
    return H350RuntimeStatus::kInvalidArgument;

  auto *input = static_cast<float *>(inputs[0].data);
  auto *output = static_cast<float *>(outputs[0].data);
  output[0] = input[0] + 1.0f;
  output[1] = input[1] + 1.0f;
  return H350RuntimeStatus::kSuccess;
}

}  // namespace

int main() {
  using namespace mlir::north_star;

  setH350LaunchCallback(static_cast<H350LaunchCallback>(mockLaunch));
  int32_t registerStatus = north_star_h350_register_descriptor(
      7, "softmax_device_1", 1, "amd_h350_npu", 1, 1);
  if (registerStatus != static_cast<int32_t>(H350RuntimeStatus::kSuccess)) {
    std::fprintf(stderr, "register failed: %d\n", registerStatus);
    return 1;
  }

  float inputData[2] = {3.0f, 4.0f};
  float outputData[2] = {0.0f, 0.0f};
  int64_t sizes[1] = {2};
  int64_t strides[1] = {1};
  H350TensorBindingF32 input{inputData, 1, sizes, strides};
  H350TensorBindingF32 output{outputData, 1, sizes, strides};

  int32_t launchStatus =
      north_star_h350_launch_descriptor_f32(7, &input, 1, &output, 1);
  if (launchStatus != static_cast<int32_t>(H350RuntimeStatus::kSuccess)) {
    std::fprintf(stderr, "launch failed: %d\n", launchStatus);
    return 1;
  }

  std::printf("descriptor=%d output=%0.1f,%0.1f\n", 7, outputData[0], outputData[1]);
  return 0;
}
