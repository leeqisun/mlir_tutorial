#include "Utils/CUDARuntimeArtifacts.h"

#include <string>

#include "Utils/CUDADriverBackend.h"
#include "llvm/Support/JSON.h"

namespace mlir::north_star {
namespace {

std::optional<int64_t> getInt64(llvm::json::Object &obj, llvm::StringRef key) {
  if (auto value = obj.getInteger(key))
    return *value;
  return std::nullopt;
}

std::optional<std::string> getString(llvm::json::Object &obj,
                                     llvm::StringRef key) {
  if (auto value = obj.getString(key))
    return value->str();
  return std::nullopt;
}

}  // namespace

void resetCUDARuntimeArtifacts() {
  clearCUDAPTXKernels();
  clearCUDAKernelDescriptors();
}

CUDARuntimeStatus loadCUDARuntimeArtifacts(const char *runtimeJson,
                                           const char *ptx,
                                           int64_t deviceOverride) {
  if (!runtimeJson || !ptx)
    return CUDARuntimeStatus::kInvalidArgument;

  resetCUDARuntimeArtifacts();

  auto parsed = llvm::json::parse(runtimeJson);
  if (!parsed)
    return CUDARuntimeStatus::kInvalidArgument;

  auto *root = parsed->getAsObject();
  if (!root)
    return CUDARuntimeStatus::kInvalidArgument;

  auto *descriptors = root->getArray("descriptors");
  if (!descriptors)
    return CUDARuntimeStatus::kInvalidArgument;

  for (llvm::json::Value &descriptorValue : *descriptors) {
    auto *descriptorObj = descriptorValue.getAsObject();
    if (!descriptorObj)
      return CUDARuntimeStatus::kInvalidArgument;

    auto descriptorId = getInt64(*descriptorObj, "descriptor_id");
    auto kernelName = getString(*descriptorObj, "kernel_name");
    auto target = getString(*descriptorObj, "target");
    auto deviceId = getInt64(*descriptorObj, "device_id");
    auto numInputs = getInt64(*descriptorObj, "num_inputs");
    auto numOutputs = getInt64(*descriptorObj, "num_outputs");
    if (!descriptorId || !kernelName || !target || !deviceId || !numInputs ||
        !numOutputs)
      return CUDARuntimeStatus::kInvalidArgument;

    int64_t effectiveDeviceId = deviceOverride >= 0 ? deviceOverride : *deviceId;
    auto registerStatus = registerCUDAKernelDescriptor(
        CUDAKernelDescriptor{*descriptorId, kernelName->c_str(), effectiveDeviceId,
                             target->c_str(), *numInputs, *numOutputs});
    if (registerStatus != CUDARuntimeStatus::kSuccess)
      return registerStatus;

    std::string kernelEntry = *kernelName + "__kernel";
    auto ptxStatus =
        registerCUDAPTXKernel(*descriptorId, ptx, kernelEntry.c_str());
    if (ptxStatus != CUDARuntimeStatus::kSuccess)
      return ptxStatus;
  }

  enableCUDADriverBackend();
  return CUDARuntimeStatus::kSuccess;
}

extern "C" int32_t north_star_cuda_load_runtime_artifacts(
    const char *runtimeJson, const char *ptx, int64_t deviceOverride) {
  return static_cast<int32_t>(
      loadCUDARuntimeArtifacts(runtimeJson, ptx, deviceOverride));
}

extern "C" void north_star_cuda_reset_runtime_artifacts() {
  resetCUDARuntimeArtifacts();
}

}  // namespace mlir::north_star
