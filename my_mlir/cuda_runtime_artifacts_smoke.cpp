#include <cstdint>
#include <cstdio>

#include "Utils/CUDARuntimeAPI.h"
#include "Utils/CUDARuntimeArtifacts.h"

namespace {

constexpr const char *kRuntimeJson = R"json(
{
  "target": "nvidia_rtx_5060_cuda",
  "c_api": "north_star_cuda_launch_f32",
  "descriptors": [
    {
      "descriptor_id": 0,
      "kernel_name": "softmax_2_d_d_softmax_2_d_d__device_1",
      "target": "nvidia_rtx_5060_cuda",
      "device_id": 1,
      "num_inputs": 1,
      "num_outputs": 1
    },
    {
      "descriptor_id": 1,
      "kernel_name": "softmax_2_d_d_softmax_2_d_d__device_2",
      "target": "nvidia_rtx_5060_cuda",
      "device_id": 2,
      "num_inputs": 1,
      "num_outputs": 1
    }
  ]
}
)json";

constexpr const char *kDummyPTX = ".version 8.0\n.target sm_50\n.address_size 64\n";

}  // namespace

int main() {
  using namespace mlir::north_star;

  north_star_cuda_reset_runtime_artifacts();
  int32_t firstLoad =
      north_star_cuda_load_runtime_artifacts(kRuntimeJson, kDummyPTX, 0);
  if (firstLoad != static_cast<int32_t>(CUDARuntimeStatus::kSuccess)) {
    std::fprintf(stderr, "first load failed: %d\n", firstLoad);
    return 1;
  }

  const CUDAKernelDescriptor *descriptor0 = lookupCUDAKernelDescriptor(0);
  const CUDAKernelDescriptor *descriptor1 = lookupCUDAKernelDescriptor(1);
  if (!descriptor0 || !descriptor1) {
    std::fprintf(stderr, "lookup failed after first load\n");
    return 1;
  }

  int32_t secondLoad =
      north_star_cuda_load_runtime_artifacts(kRuntimeJson, kDummyPTX, 0);
  if (secondLoad != static_cast<int32_t>(CUDARuntimeStatus::kSuccess)) {
    std::fprintf(stderr, "second load failed: %d\n", secondLoad);
    return 1;
  }

  descriptor0 = lookupCUDAKernelDescriptor(0);
  descriptor1 = lookupCUDAKernelDescriptor(1);
  if (!descriptor0 || !descriptor1) {
    std::fprintf(stderr, "lookup failed after second load\n");
    return 1;
  }

  std::printf("artifacts descriptors=%d:%s@%lld,%d:%s@%lld\n", 0,
              descriptor0->kernelName, static_cast<long long>(descriptor0->deviceId),
              1, descriptor1->kernelName,
              static_cast<long long>(descriptor1->deviceId));
  return 0;
}
