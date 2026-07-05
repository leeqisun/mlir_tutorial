#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#include "Utils/CUDADriverBackend.h"
#include "Utils/CUDARuntimeAPI.h"

namespace {

std::string readFile(const char *path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return {};
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

}  // namespace

int main(int argc, char **argv) {
  using namespace mlir::north_star;

  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <kernel.ptx>\n", argv[0]);
    return 1;
  }

  if (!isCUDADriverBackendAvailable()) {
    std::fprintf(stderr, "cuda driver backend unavailable\n");
    return 2;
  }

  std::string ptx = readFile(argv[1]);
  if (ptx.empty()) {
    std::fprintf(stderr, "failed to read PTX: %s\n", argv[1]);
    return 1;
  }

  constexpr int64_t kDescriptorId = 31;
  constexpr const char *kKernelName =
      "softmax_2_d_d_softmax_2_d_d__device_1";
  constexpr const char *kKernelEntry =
      "softmax_2_d_d_softmax_2_d_d__device_1__kernel";

  int32_t registerStatus = north_star_cuda_register_descriptor(
      kDescriptorId, kKernelName, 0, "nvidia_rtx_5060_cuda", 1, 1);
  if (registerStatus != static_cast<int32_t>(CUDARuntimeStatus::kSuccess)) {
    std::fprintf(stderr, "descriptor register failed: %d\n", registerStatus);
    return 1;
  }

  int32_t ptxStatus =
      north_star_cuda_register_ptx(kDescriptorId, ptx.c_str(), kKernelEntry);
  if (ptxStatus != static_cast<int32_t>(CUDARuntimeStatus::kSuccess)) {
    std::fprintf(stderr, "ptx register failed: %d\n", ptxStatus);
    return 1;
  }

  float inputData[8] = {1.0f, 2.0f, 0.0f, 1.0f, 3.0f, 0.5f, 1.0f, -1.0f};
  float outputData[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  int64_t sizes[3] = {2, 2, 2};
  int64_t strides[3] = {4, 2, 1};
  CUDATensorBindingF32 input{inputData, 3, sizes, strides};
  CUDATensorBindingF32 output{outputData, 3, sizes, strides};

  int32_t launchStatus =
      north_star_cuda_launch_descriptor_f32(kDescriptorId, &input, 1, &output, 1);
  if (launchStatus != static_cast<int32_t>(CUDARuntimeStatus::kSuccess)) {
    std::fprintf(stderr, "launch failed: %d\n", launchStatus);
    return 1;
  }

  std::printf("generated-ptx output=%0.6f,%0.6f,%0.6f,%0.6f\n", outputData[0],
              outputData[1], outputData[2], outputData[3]);
  return 0;
}
