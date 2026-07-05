#ifndef UTILS_MLIR_UTILS_CUDARUNTIMEARTIFACTS_H
#define UTILS_MLIR_UTILS_CUDARUNTIMEARTIFACTS_H

#include <cstdint>

#include "Utils/CUDARuntimeAPI.h"

namespace mlir::north_star {

CUDARuntimeStatus loadCUDARuntimeArtifacts(const char *runtimeJson,
                                           const char *ptx,
                                           int64_t deviceOverride);
void resetCUDARuntimeArtifacts();

extern "C" int32_t north_star_cuda_load_runtime_artifacts(
    const char *runtimeJson, const char *ptx, int64_t deviceOverride);
extern "C" void north_star_cuda_reset_runtime_artifacts();

}  // namespace mlir::north_star

#endif  // UTILS_MLIR_UTILS_CUDARUNTIMEARTIFACTS_H
