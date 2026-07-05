#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#include "Utils/CUDARuntimeAPI.h"
#include "Utils/CUDARuntimeArtifacts.h"

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

  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <runtime.json> <kernel.ptx>\n", argv[0]);
    return 1;
  }

  std::string runtimeJson = readFile(argv[1]);
  std::string ptx = readFile(argv[2]);
  if (runtimeJson.empty() || ptx.empty()) {
    std::fprintf(stderr, "failed to read runtime bundle inputs\n");
    return 1;
  }

  int32_t loadStatus =
      north_star_cuda_load_runtime_artifacts(runtimeJson.c_str(), ptx.c_str(), 0);
  if (loadStatus != static_cast<int32_t>(CUDARuntimeStatus::kSuccess)) {
    std::fprintf(stderr, "bundle load failed: %d\n", loadStatus);
    return 1;
  }

  float inputData[8] = {1.0f, 2.0f, 0.0f, 1.0f, 3.0f, 0.5f, 1.0f, -1.0f};
  float outputData[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  int64_t sizes[3] = {2, 2, 2};
  int64_t strides[3] = {4, 2, 1};
  CUDATensorBindingF32 input{inputData, 3, sizes, strides};
  CUDATensorBindingF32 output{outputData, 3, sizes, strides};

  int32_t launchStatus = north_star_cuda_launch_descriptor_f32(0, &input, 1,
                                                               &output, 1);
  if (launchStatus != static_cast<int32_t>(CUDARuntimeStatus::kSuccess)) {
    std::fprintf(stderr, "launch failed: %d\n", launchStatus);
    return 1;
  }

  std::printf("bundle output=%0.6f,%0.6f,%0.6f,%0.6f\n", outputData[0],
              outputData[1], outputData[2], outputData[3]);
  return 0;
}
