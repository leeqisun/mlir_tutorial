#ifndef UTILS_MLIR_UTILS_KEY_H
#define UTILS_MLIR_UTILS_KEY_H

inline static const char* KEntryPointName = "main";
inline static const char* KDPAttrName = "dp_attr";
inline static const char* KHostFunc = "host_func";
inline static const char* KDeviceFunc = "device_kernel";
inline static const char* KNPUKernelAttr = "north_star.kernel";
inline static const char* KNPUTargetAttr = "north_star.target";
inline static const char* KH350NPUTarget = "amd_h350_npu";
inline static const char* KRTX5060CUDATarget = "nvidia_rtx_5060_cuda";
inline static const char* KSerializedKernelsAttr = "north_star.serialized_kernels";
inline static const char* KRuntimeJsonAttr = "north_star.runtime_json";
inline static const char* KNPURuntimeStubAttr = "north_star.runtime_stub";
inline static const char* KNPURuntimeKernelNameAttr = "north_star.kernel_name";
inline static const char* KNPURuntimeStubPrefix = "north_star_runtime_launch_";
inline static const char* KNPURuntimeBridgeAttr = "north_star.runtime_bridge";
inline static const char* KNPURuntimeBridgePrefix = "north_star_h350_launch_";
inline static const char* KNPURuntimeCUDABridgeAttr = "north_star.runtime_cuda_bridge";
inline static const char* KNPURuntimeCUDABridgePrefix = "north_star_cuda_launch_";
inline static const char* KNPURuntimeDescriptorIdAttr = "north_star.descriptor_id";
inline static const char* KNPURuntimeCApiAttr = "north_star.runtime_c_api";
inline static const char* KH350UnifiedLaunchAPI = "north_star_h350_launch_f32";
inline static const char* KNPURuntimeCUDACApiAttr = "north_star.runtime_cuda_c_api";
inline static const char* KCUDAUnifiedLaunchAPI = "north_star_cuda_launch_f32";
inline static const char* KGPUKernelEntryAttr = "north_star.gpu_kernel_entry";
inline static const char* KGPUKernelABIAttr = "north_star.gpu_kernel_abi";

#endif  // UTILS_MLIR_UTILS_KEY_H
