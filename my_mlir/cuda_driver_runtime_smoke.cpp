#include <cstdint>
#include <cstdio>

#include "Utils/CUDADriverBackend.h"
#include "Utils/CUDARuntimeAPI.h"

namespace {

constexpr const char *kPTX = R"ptx(
.version 8.0
.target sm_50
.address_size 64

.visible .entry softmax_cuda_device_0__kernel(
    .param .u64 softmax_cuda_device_0__kernel_param_0,
    .param .u64 softmax_cuda_device_0__kernel_param_1,
    .param .u64 softmax_cuda_device_0__kernel_param_2,
    .param .u64 softmax_cuda_device_0__kernel_param_3,
    .param .u64 softmax_cuda_device_0__kernel_param_4,
    .param .u64 softmax_cuda_device_0__kernel_param_5,
    .param .u64 softmax_cuda_device_0__kernel_param_6,
    .param .u64 softmax_cuda_device_0__kernel_param_7,
    .param .u64 softmax_cuda_device_0__kernel_param_8,
    .param .u64 softmax_cuda_device_0__kernel_param_9,
    .param .u64 softmax_cuda_device_0__kernel_param_10,
    .param .u64 softmax_cuda_device_0__kernel_param_11,
    .param .u64 softmax_cuda_device_0__kernel_param_12,
    .param .u64 softmax_cuda_device_0__kernel_param_13,
    .param .u64 softmax_cuda_device_0__kernel_param_14,
    .param .u64 softmax_cuda_device_0__kernel_param_15,
    .param .u64 softmax_cuda_device_0__kernel_param_16,
    .param .u64 softmax_cuda_device_0__kernel_param_17
)
{
    .reg .f32 %f<4>;
    .reg .b64 %rd<5>;

    ld.param.u64 %rd1, [softmax_cuda_device_0__kernel_param_1];
    ld.param.u64 %rd2, [softmax_cuda_device_0__kernel_param_10];
    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;
    ld.global.f32 %f1, [%rd3];
    mov.f32 %f2, 0f40000000;
    mul.f32 %f3, %f1, %f2;
    st.global.f32 [%rd4], %f3;
    ret;
}
)ptx";

}  // namespace

int main() {
  using namespace mlir::north_star;

  if (!isCUDADriverBackendAvailable()) {
    std::fprintf(stderr, "cuda driver backend unavailable\n");
    return 2;
  }

  int32_t registerStatus = north_star_cuda_register_descriptor(
      21, "softmax_cuda_device_0", 0, "nvidia_rtx_5060_cuda", 1, 1);
  if (registerStatus != static_cast<int32_t>(CUDARuntimeStatus::kSuccess)) {
    std::fprintf(stderr, "descriptor register failed: %d\n", registerStatus);
    return 1;
  }

  int32_t ptxStatus =
      north_star_cuda_register_ptx(21, kPTX, "softmax_cuda_device_0__kernel");
  if (ptxStatus != static_cast<int32_t>(CUDARuntimeStatus::kSuccess)) {
    std::fprintf(stderr, "ptx register failed: %d\n", ptxStatus);
    return 1;
  }

  float inputData[2] = {3.0f, 4.0f};
  float outputData[2] = {0.0f, 0.0f};
  int64_t sizes[3] = {2, 1, 1};
  int64_t strides[3] = {1, 1, 1};
  CUDATensorBindingF32 input{inputData, 3, sizes, strides};
  CUDATensorBindingF32 output{outputData, 3, sizes, strides};

  int32_t launchStatus =
      north_star_cuda_launch_descriptor_f32(21, &input, 1, &output, 1);
  if (launchStatus != static_cast<int32_t>(CUDARuntimeStatus::kSuccess)) {
    std::fprintf(stderr, "launch failed: %d\n", launchStatus);
    return 1;
  }

  std::printf("descriptor=%d output=%0.1f,%0.1f\n", 21, outputData[0],
              outputData[1]);
  return 0;
}
