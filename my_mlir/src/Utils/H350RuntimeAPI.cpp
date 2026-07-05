#include "Utils/H350RuntimeAPI.h"

#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace mlir::north_star {
namespace {

struct RuntimeState {
  std::mutex mutex;
  std::unordered_map<int64_t, H350KernelDescriptor> descriptors;
  std::unordered_map<int64_t, std::string> kernelNames;
  std::unordered_map<int64_t, std::string> targets;
  H350LaunchCallback callback = nullptr;
};

RuntimeState &getRuntimeState() {
  static RuntimeState state;
  return state;
}

}  // namespace

H350RuntimeStatus registerH350KernelDescriptor(H350KernelDescriptor descriptor) {
  if (!descriptor.kernelName || !descriptor.target || descriptor.numInputs < 0 ||
      descriptor.numOutputs < 0)
    return H350RuntimeStatus::kInvalidArgument;

  auto &state = getRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.descriptors.count(descriptor.descriptorId) != 0)
    return H350RuntimeStatus::kDescriptorExists;

  auto &kernelName = state.kernelNames[descriptor.descriptorId];
  kernelName = descriptor.kernelName;
  auto &target = state.targets[descriptor.descriptorId];
  target = descriptor.target;

  H350KernelDescriptor stored = descriptor;
  stored.kernelName = kernelName.c_str();
  stored.target = target.c_str();
  state.descriptors.emplace(descriptor.descriptorId, stored);
  return H350RuntimeStatus::kSuccess;
}

const H350KernelDescriptor *lookupH350KernelDescriptor(int64_t descriptorId) {
  auto &state = getRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto it = state.descriptors.find(descriptorId);
  if (it == state.descriptors.end())
    return nullptr;
  return &it->second;
}

void setH350LaunchCallback(H350LaunchCallback callback) {
  auto &state = getRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.callback = callback;
}

H350LaunchCallback getH350LaunchCallback() {
  auto &state = getRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.callback;
}

extern "C" int32_t north_star_h350_register_descriptor(
    int64_t descriptorId, const char *kernelName, int64_t deviceId,
    const char *target, int64_t numInputs, int64_t numOutputs) {
  return static_cast<int32_t>(registerH350KernelDescriptor(
      H350KernelDescriptor{descriptorId, kernelName, deviceId, target, numInputs,
                           numOutputs}));
}

extern "C" int32_t north_star_h350_launch_descriptor_f32(
    int64_t descriptorId, const H350TensorBindingF32 *inputs, int64_t numInputs,
    H350TensorBindingF32 *outputs, int64_t numOutputs) {
  if ((numInputs > 0 && !inputs) || (numOutputs > 0 && !outputs))
    return static_cast<int32_t>(H350RuntimeStatus::kInvalidArgument);

  const H350KernelDescriptor *descriptor = lookupH350KernelDescriptor(descriptorId);
  if (!descriptor)
    return static_cast<int32_t>(H350RuntimeStatus::kDescriptorNotFound);

  H350LaunchCallback callback = getH350LaunchCallback();
  if (!callback)
    return static_cast<int32_t>(H350RuntimeStatus::kBackendMissing);

  return static_cast<int32_t>(
      callback(*descriptor, inputs, numInputs, outputs, numOutputs));
}

}  // namespace mlir::north_star
