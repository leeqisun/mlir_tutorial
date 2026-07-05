#ifndef UTILS_MLIR_UTILS_H350RUNTIMEAPI_H
#define UTILS_MLIR_UTILS_H350RUNTIMEAPI_H

#include <cstdint>

namespace mlir::north_star {

struct H350TensorBindingF32 {
  void *data;
  int64_t rank;
  const int64_t *sizes;
  const int64_t *strides;
};

struct H350KernelDescriptor {
  int64_t descriptorId;
  const char *kernelName;
  int64_t deviceId;
  const char *target;
  int64_t numInputs;
  int64_t numOutputs;
};

enum class H350RuntimeStatus : int32_t {
  kSuccess = 0,
  kDescriptorExists = 1,
  kDescriptorNotFound = 2,
  kBackendMissing = 3,
  kInvalidArgument = 4,
};

using H350LaunchCallback = H350RuntimeStatus (*)(
    const H350KernelDescriptor &descriptor, const H350TensorBindingF32 *inputs,
    int64_t numInputs, H350TensorBindingF32 *outputs, int64_t numOutputs);

H350RuntimeStatus registerH350KernelDescriptor(H350KernelDescriptor descriptor);
const H350KernelDescriptor *lookupH350KernelDescriptor(int64_t descriptorId);
void setH350LaunchCallback(H350LaunchCallback callback);
H350LaunchCallback getH350LaunchCallback();

extern "C" int32_t north_star_h350_register_descriptor(
    int64_t descriptorId, const char *kernelName, int64_t deviceId,
    const char *target, int64_t numInputs, int64_t numOutputs);

extern "C" int32_t north_star_h350_launch_descriptor_f32(
    int64_t descriptorId, const H350TensorBindingF32 *inputs, int64_t numInputs,
    H350TensorBindingF32 *outputs, int64_t numOutputs);

}  // namespace mlir::north_star

#endif  // UTILS_MLIR_UTILS_H350RUNTIMEAPI_H
