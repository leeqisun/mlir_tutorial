#ifndef UTILS_MLIR_UTILS_CUDADRIVERBACKEND_H
#define UTILS_MLIR_UTILS_CUDADRIVERBACKEND_H

#include <cstdint>

#include "Utils/CUDARuntimeAPI.h"

namespace mlir::north_star {

CUDARuntimeStatus registerCUDAPTXKernel(int64_t descriptorId, const char *ptx,
                                        const char *kernelEntry = nullptr);
void clearCUDAPTXKernels();
bool isCUDADriverBackendAvailable();
void enableCUDADriverBackend();

extern "C" int32_t north_star_cuda_register_ptx(int64_t descriptorId,
                                                 const char *ptx,
                                                 const char *kernelEntry);

}  // namespace mlir::north_star

#endif  // UTILS_MLIR_UTILS_CUDADRIVERBACKEND_H
