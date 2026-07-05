#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

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

bool parseInt64(const char *text, int64_t &value) {
  char *end = nullptr;
  long long parsed = std::strtoll(text, &end, 10);
  if (!end || *end != '\0')
    return false;
  value = static_cast<int64_t>(parsed);
  return true;
}

bool parseFloat(const char *text, float &value) {
  char *end = nullptr;
  value = std::strtof(text, &end);
  return end && *end == '\0';
}

int64_t getElementCount(const std::vector<int64_t> &shape) {
  int64_t elements = 1;
  for (int64_t dim : shape)
    elements *= dim;
  return elements;
}

std::vector<int64_t> buildRowMajorStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
}

void printUsage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s <runtime.json> <kernel.ptx> <descriptor-id> "
               "<dim0> <dim1> <dim2> <v0> ... <vN-1> [--device-override N]\n",
               argv0);
}

}  // namespace

int main(int argc, char **argv) {
  using namespace mlir::north_star;

  if (argc < 10) {
    printUsage(argv[0]);
    return 1;
  }

  int64_t descriptorId = 0;
  std::vector<int64_t> shape(3, 0);
  if (!parseInt64(argv[3], descriptorId) || !parseInt64(argv[4], shape[0]) ||
      !parseInt64(argv[5], shape[1]) || !parseInt64(argv[6], shape[2])) {
    std::fprintf(stderr, "failed to parse descriptor id or shape\n");
    return 1;
  }

  int64_t deviceOverride = 0;
  int valueArgEnd = argc;
  if (argc >= 12 && std::string(argv[argc - 2]) == "--device-override") {
    if (!parseInt64(argv[argc - 1], deviceOverride)) {
      std::fprintf(stderr, "failed to parse device override\n");
      return 1;
    }
    valueArgEnd -= 2;
  }

  const int64_t elements = getElementCount(shape);
  if (elements <= 0) {
    std::fprintf(stderr, "invalid shape\n");
    return 1;
  }

  if (valueArgEnd - 7 != elements) {
    std::fprintf(stderr, "expected %lld values, got %d\n",
                 static_cast<long long>(elements), valueArgEnd - 7);
    return 1;
  }

  std::string runtimeJson = readFile(argv[1]);
  std::string ptx = readFile(argv[2]);
  if (runtimeJson.empty() || ptx.empty()) {
    std::fprintf(stderr, "failed to read runtime bundle inputs\n");
    return 1;
  }

  int32_t loadStatus = north_star_cuda_load_runtime_artifacts(
      runtimeJson.c_str(), ptx.c_str(), deviceOverride);
  if (loadStatus != static_cast<int32_t>(CUDARuntimeStatus::kSuccess)) {
    std::fprintf(stderr, "bundle load failed: %d\n", loadStatus);
    return 1;
  }

  std::vector<float> inputData(elements, 0.0f);
  for (int i = 0; i < elements; ++i) {
    if (!parseFloat(argv[7 + i], inputData[i])) {
      std::fprintf(stderr, "failed to parse input value %d\n", i);
      return 1;
    }
  }
  std::vector<float> outputData(elements, 0.0f);
  std::vector<int64_t> strides = buildRowMajorStrides(shape);

  CUDATensorBindingF32 input{inputData.data(), static_cast<int64_t>(shape.size()),
                             shape.data(), strides.data()};
  CUDATensorBindingF32 output{outputData.data(),
                              static_cast<int64_t>(shape.size()), shape.data(),
                              strides.data()};

  int32_t launchStatus = north_star_cuda_launch_descriptor_f32(
      descriptorId, &input, 1, &output, 1);
  if (launchStatus != static_cast<int32_t>(CUDARuntimeStatus::kSuccess)) {
    std::fprintf(stderr, "launch failed: %d\n", launchStatus);
    return 1;
  }

  std::printf("descriptor=%lld shape=%lldx%lldx%lld output=",
              static_cast<long long>(descriptorId),
              static_cast<long long>(shape[0]),
              static_cast<long long>(shape[1]),
              static_cast<long long>(shape[2]));
  for (int64_t i = 0; i < elements; ++i) {
    std::printf("%s%0.6f", i == 0 ? "" : ",", outputData[i]);
  }
  std::printf("\n");
  return 0;
}
