#ifndef UTILS_MLIR_UTILS_CUDARUNTIMEAPI_H
#define UTILS_MLIR_UTILS_CUDARUNTIMEAPI_H

#include <cstdint>

namespace mlir::north_star {

struct CUDATensorBindingF32 {
  void *data;
  int64_t rank;
  const int64_t *sizes;
  const int64_t *strides;
};

struct CUDAKernelDescriptor {
  int64_t descriptorId;
  const char *kernelName;
  int64_t deviceId;
  const char *target;
  int64_t numInputs;
  int64_t numOutputs;
};

enum class CUDARuntimeStatus : int32_t {
  kSuccess = 0,
  kDescriptorExists = 1,
  kDescriptorNotFound = 2,
  kBackendMissing = 3,
  kInvalidArgument = 4,
  kModuleExists = 5,
  kModuleLoadFailed = 6,
  kLaunchFailed = 7,
  kUnsupported = 8,
  kKernelImageMissing = 9,
};

using CUDALaunchCallback = CUDARuntimeStatus (*)(
    const CUDAKernelDescriptor &descriptor, const CUDATensorBindingF32 *inputs,
    int64_t numInputs, CUDATensorBindingF32 *outputs, int64_t numOutputs);

CUDARuntimeStatus registerCUDAKernelDescriptor(CUDAKernelDescriptor descriptor);
const CUDAKernelDescriptor *lookupCUDAKernelDescriptor(int64_t descriptorId);
void clearCUDAKernelDescriptors();
void setCUDALaunchCallback(CUDALaunchCallback callback);
CUDALaunchCallback getCUDALaunchCallback();

extern "C" int32_t north_star_cuda_register_descriptor(
    int64_t descriptorId, const char *kernelName, int64_t deviceId,
    const char *target, int64_t numInputs, int64_t numOutputs);

extern "C" int32_t north_star_cuda_launch_descriptor_f32(
    int64_t descriptorId, const CUDATensorBindingF32 *inputs, int64_t numInputs,
    CUDATensorBindingF32 *outputs, int64_t numOutputs);

}  // namespace mlir::north_star

#endif  // UTILS_MLIR_UTILS_CUDARUNTIMEAPI_H
