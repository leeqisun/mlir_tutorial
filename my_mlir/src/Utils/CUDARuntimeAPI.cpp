#include "Utils/CUDARuntimeAPI.h"

#include <mutex>
#include <string>
#include <unordered_map>

namespace mlir::north_star {
namespace {

struct RuntimeState {
  std::mutex mutex;
  std::unordered_map<int64_t, CUDAKernelDescriptor> descriptors;
  std::unordered_map<int64_t, std::string> kernelNames;
  std::unordered_map<int64_t, std::string> targets;
  CUDALaunchCallback callback = nullptr;
};

RuntimeState &getRuntimeState() {
  static RuntimeState state;
  return state;
}

}  // namespace

CUDARuntimeStatus registerCUDAKernelDescriptor(CUDAKernelDescriptor descriptor) {
  if (!descriptor.kernelName || !descriptor.target || descriptor.numInputs < 0 ||
      descriptor.numOutputs < 0)
    return CUDARuntimeStatus::kInvalidArgument;

  auto &state = getRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (auto it = state.descriptors.find(descriptor.descriptorId);
      it != state.descriptors.end()) {
    const CUDAKernelDescriptor &existing = it->second;
    if (existing.deviceId == descriptor.deviceId &&
        existing.numInputs == descriptor.numInputs &&
        existing.numOutputs == descriptor.numOutputs &&
        std::string(existing.kernelName) == descriptor.kernelName &&
        std::string(existing.target) == descriptor.target)
      return CUDARuntimeStatus::kSuccess;
    return CUDARuntimeStatus::kDescriptorExists;
  }

  auto &kernelName = state.kernelNames[descriptor.descriptorId];
  kernelName = descriptor.kernelName;
  auto &target = state.targets[descriptor.descriptorId];
  target = descriptor.target;

  CUDAKernelDescriptor stored = descriptor;
  stored.kernelName = kernelName.c_str();
  stored.target = target.c_str();
  state.descriptors.emplace(descriptor.descriptorId, stored);
  return CUDARuntimeStatus::kSuccess;
}

const CUDAKernelDescriptor *lookupCUDAKernelDescriptor(int64_t descriptorId) {
  auto &state = getRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto it = state.descriptors.find(descriptorId);
  if (it == state.descriptors.end())
    return nullptr;
  return &it->second;
}

void clearCUDAKernelDescriptors() {
  auto &state = getRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.descriptors.clear();
  state.kernelNames.clear();
  state.targets.clear();
  state.callback = nullptr;
}

void setCUDALaunchCallback(CUDALaunchCallback callback) {
  auto &state = getRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.callback = callback;
}

CUDALaunchCallback getCUDALaunchCallback() {
  auto &state = getRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.callback;
}

extern "C" int32_t north_star_cuda_register_descriptor(
    int64_t descriptorId, const char *kernelName, int64_t deviceId,
    const char *target, int64_t numInputs, int64_t numOutputs) {
  return static_cast<int32_t>(registerCUDAKernelDescriptor(
      CUDAKernelDescriptor{descriptorId, kernelName, deviceId, target, numInputs,
                           numOutputs}));
}

extern "C" int32_t north_star_cuda_launch_descriptor_f32(
    int64_t descriptorId, const CUDATensorBindingF32 *inputs, int64_t numInputs,
    CUDATensorBindingF32 *outputs, int64_t numOutputs) {
  if ((numInputs > 0 && !inputs) || (numOutputs > 0 && !outputs))
    return static_cast<int32_t>(CUDARuntimeStatus::kInvalidArgument);

  const CUDAKernelDescriptor *descriptor =
      lookupCUDAKernelDescriptor(descriptorId);
  if (!descriptor)
    return static_cast<int32_t>(CUDARuntimeStatus::kDescriptorNotFound);

  CUDALaunchCallback callback = getCUDALaunchCallback();
  if (!callback)
    return static_cast<int32_t>(CUDARuntimeStatus::kBackendMissing);

  return static_cast<int32_t>(
      callback(*descriptor, inputs, numInputs, outputs, numOutputs));
}

}  // namespace mlir::north_star
