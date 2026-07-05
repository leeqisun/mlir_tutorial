#include "Utils/CUDADriverBackend.h"

#include <cuda.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlir::north_star {
namespace {

struct LoadedKernel {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
};

struct PTXImage {
  std::string ptx;
  std::string kernelEntry;
  std::unordered_map<int64_t, LoadedKernel> loadedKernelsByDevice;
};

struct DeviceContextState {
  CUdevice device = 0;
  CUcontext context = nullptr;
};

struct DriverRuntimeState {
  std::mutex mutex;
  bool initialized = false;
  std::unordered_map<int64_t, PTXImage> images;
  std::unordered_map<int64_t, DeviceContextState> contexts;
};

DriverRuntimeState &getDriverRuntimeState() {
  static DriverRuntimeState state;
  return state;
}

static CUDARuntimeStatus mapDriverError(CUresult result) {
  if (result == CUDA_SUCCESS)
    return CUDARuntimeStatus::kSuccess;
  return CUDARuntimeStatus::kLaunchFailed;
}

static CUDARuntimeStatus ensureDriverInitializedLocked(DriverRuntimeState &state) {
  if (state.initialized)
    return CUDARuntimeStatus::kSuccess;
  if (cuInit(0) != CUDA_SUCCESS)
    return CUDARuntimeStatus::kBackendMissing;
  state.initialized = true;
  return CUDARuntimeStatus::kSuccess;
}

static CUDARuntimeStatus ensureDeviceContextLocked(DriverRuntimeState &state,
                                                   int64_t deviceId,
                                                   DeviceContextState *&ctxState) {
  auto initStatus = ensureDriverInitializedLocked(state);
  if (initStatus != CUDARuntimeStatus::kSuccess)
    return initStatus;

  auto it = state.contexts.find(deviceId);
  if (it != state.contexts.end()) {
    ctxState = &it->second;
    return CUDARuntimeStatus::kSuccess;
  }

  CUdevice device = 0;
  if (cuDeviceGet(&device, static_cast<int>(deviceId)) != CUDA_SUCCESS)
    return CUDARuntimeStatus::kInvalidArgument;

  CUcontext context = nullptr;
  if (cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS)
    return CUDARuntimeStatus::kBackendMissing;

  auto [insertedIt, _] =
      state.contexts.emplace(deviceId, DeviceContextState{device, context});
  ctxState = &insertedIt->second;
  return CUDARuntimeStatus::kSuccess;
}

static std::string inferKernelEntryName(const CUDAKernelDescriptor &descriptor,
                                        const PTXImage &image) {
  if (!image.kernelEntry.empty())
    return image.kernelEntry;
  return std::string(descriptor.kernelName) + "__kernel";
}

static CUDARuntimeStatus ensureKernelLoadedLocked(
    DriverRuntimeState &state, const CUDAKernelDescriptor &descriptor,
    LoadedKernel *&loadedKernel) {
  auto imageIt = state.images.find(descriptor.descriptorId);
  if (imageIt == state.images.end())
    return CUDARuntimeStatus::kKernelImageMissing;

  DeviceContextState *ctxState = nullptr;
  auto contextStatus =
      ensureDeviceContextLocked(state, descriptor.deviceId, ctxState);
  if (contextStatus != CUDARuntimeStatus::kSuccess)
    return contextStatus;

  auto &loadedByDevice = imageIt->second.loadedKernelsByDevice;
  auto loadedIt = loadedByDevice.find(descriptor.deviceId);
  if (loadedIt != loadedByDevice.end()) {
    loadedKernel = &loadedIt->second;
    return CUDARuntimeStatus::kSuccess;
  }

  if (cuCtxSetCurrent(ctxState->context) != CUDA_SUCCESS)
    return CUDARuntimeStatus::kBackendMissing;

  CUmodule module = nullptr;
  if (cuModuleLoadDataEx(&module, imageIt->second.ptx.c_str(), 0, nullptr,
                         nullptr) != CUDA_SUCCESS)
    return CUDARuntimeStatus::kModuleLoadFailed;

  CUfunction function = nullptr;
  std::string kernelEntry = inferKernelEntryName(descriptor, imageIt->second);
  if (cuModuleGetFunction(&function, module, kernelEntry.c_str()) !=
      CUDA_SUCCESS) {
    cuModuleUnload(module);
    return CUDARuntimeStatus::kModuleLoadFailed;
  }

  auto [insertedIt, _] = loadedByDevice.emplace(
      descriptor.deviceId, LoadedKernel{module, function});
  loadedKernel = &insertedIt->second;
  return CUDARuntimeStatus::kSuccess;
}

static int64_t getElementCount(const CUDATensorBindingF32 &binding) {
  int64_t elements = 1;
  for (int64_t i = 0; i < binding.rank; ++i)
    elements *= binding.sizes[i];
  return elements;
}

static CUDARuntimeStatus launchWithDriverBackend(
    const CUDAKernelDescriptor &descriptor, const CUDATensorBindingF32 *inputs,
    int64_t numInputs, CUDATensorBindingF32 *outputs, int64_t numOutputs) {
  if (numInputs != 1 || numOutputs != 1)
    return CUDARuntimeStatus::kUnsupported;
  if (!inputs || !outputs)
    return CUDARuntimeStatus::kInvalidArgument;

  const CUDATensorBindingF32 &input = inputs[0];
  CUDATensorBindingF32 &output = outputs[0];
  if (input.rank != 3 || output.rank != 3)
    return CUDARuntimeStatus::kUnsupported;
  if (!input.data || !output.data || !input.sizes || !input.strides ||
      !output.sizes || !output.strides)
    return CUDARuntimeStatus::kInvalidArgument;

  auto &state = getDriverRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);

  LoadedKernel *loadedKernel = nullptr;
  auto loadStatus = ensureKernelLoadedLocked(state, descriptor, loadedKernel);
  if (loadStatus != CUDARuntimeStatus::kSuccess)
    return loadStatus;

  auto ctxIt = state.contexts.find(descriptor.deviceId);
  if (ctxIt == state.contexts.end())
    return CUDARuntimeStatus::kBackendMissing;
  if (cuCtxSetCurrent(ctxIt->second.context) != CUDA_SUCCESS)
    return CUDARuntimeStatus::kBackendMissing;

  const int64_t inputElements = getElementCount(input);
  const int64_t outputElements = getElementCount(output);
  const size_t inputBytes =
      static_cast<size_t>(inputElements) * sizeof(float);
  const size_t outputBytes =
      static_cast<size_t>(outputElements) * sizeof(float);

  CUdeviceptr deviceInput = 0;
  CUdeviceptr deviceOutput = 0;
  CUDARuntimeStatus status = CUDARuntimeStatus::kSuccess;
  if (cuMemAlloc(&deviceInput, inputBytes) != CUDA_SUCCESS)
    return CUDARuntimeStatus::kLaunchFailed;
  if (cuMemAlloc(&deviceOutput, outputBytes) != CUDA_SUCCESS) {
    cuMemFree(deviceInput);
    return CUDARuntimeStatus::kLaunchFailed;
  }

  if (cuMemcpyHtoD(deviceInput, input.data, inputBytes) != CUDA_SUCCESS) {
    status = CUDARuntimeStatus::kLaunchFailed;
  } else {
    uint64_t inputAllocated = static_cast<uint64_t>(deviceInput);
    uint64_t inputAligned = static_cast<uint64_t>(deviceInput);
    int64_t inputOffset = 0;
    int64_t inputSizes[3] = {input.sizes[0], input.sizes[1], input.sizes[2]};
    int64_t inputStrides[3] = {input.strides[0], input.strides[1],
                               input.strides[2]};

    uint64_t outputAllocated = static_cast<uint64_t>(deviceOutput);
    uint64_t outputAligned = static_cast<uint64_t>(deviceOutput);
    int64_t outputOffset = 0;
    int64_t outputSizes[3] = {output.sizes[0], output.sizes[1], output.sizes[2]};
    int64_t outputStrides[3] = {output.strides[0], output.strides[1],
                                output.strides[2]};

    void *kernelParams[] = {
        &inputAllocated,   &inputAligned,   &inputOffset,     &inputSizes[0],
        &inputSizes[1],    &inputSizes[2],  &inputStrides[0], &inputStrides[1],
        &inputStrides[2],  &outputAllocated, &outputAligned,   &outputOffset,
        &outputSizes[0],   &outputSizes[1], &outputSizes[2],  &outputStrides[0],
        &outputStrides[1], &outputStrides[2],
    };

    CUresult launchResult =
        cuLaunchKernel(loadedKernel->function, 1, 1, 1, 256, 1, 1, 0, nullptr,
                       kernelParams, nullptr);
    if (launchResult != CUDA_SUCCESS) {
      status = mapDriverError(launchResult);
    } else if (cuCtxSynchronize() != CUDA_SUCCESS) {
      status = CUDARuntimeStatus::kLaunchFailed;
    } else if (cuMemcpyDtoH(output.data, deviceOutput, outputBytes) !=
               CUDA_SUCCESS) {
      status = CUDARuntimeStatus::kLaunchFailed;
    }
  }
  cuMemFree(deviceOutput);
  cuMemFree(deviceInput);
  return status;
}

}  // namespace

CUDARuntimeStatus registerCUDAPTXKernel(int64_t descriptorId, const char *ptx,
                                        const char *kernelEntry) {
  if (!ptx)
    return CUDARuntimeStatus::kInvalidArgument;

  auto &state = getDriverRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (auto it = state.images.find(descriptorId); it != state.images.end()) {
    std::string requestedEntry = kernelEntry ? kernelEntry : "";
    if (it->second.ptx == ptx && it->second.kernelEntry == requestedEntry)
      return CUDARuntimeStatus::kSuccess;
    return CUDARuntimeStatus::kModuleExists;
  }

  PTXImage image;
  image.ptx = ptx;
  if (kernelEntry)
    image.kernelEntry = kernelEntry;
  state.images.emplace(descriptorId, std::move(image));
  return CUDARuntimeStatus::kSuccess;
}

void clearCUDAPTXKernels() {
  auto &state = getDriverRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  for (auto &[_, image] : state.images) {
    for (auto &[__, loadedKernel] : image.loadedKernelsByDevice) {
      if (loadedKernel.module)
        cuModuleUnload(loadedKernel.module);
    }
  }
  state.images.clear();
}

bool isCUDADriverBackendAvailable() {
  auto &state = getDriverRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  return ensureDriverInitializedLocked(state) == CUDARuntimeStatus::kSuccess;
}

void enableCUDADriverBackend() {
  setCUDALaunchCallback(launchWithDriverBackend);
}

extern "C" int32_t north_star_cuda_register_ptx(int64_t descriptorId,
                                                 const char *ptx,
                                                 const char *kernelEntry) {
  auto status = registerCUDAPTXKernel(descriptorId, ptx, kernelEntry);
  if (status == CUDARuntimeStatus::kSuccess)
    enableCUDADriverBackend();
  return static_cast<int32_t>(status);
}

}  // namespace mlir::north_star
