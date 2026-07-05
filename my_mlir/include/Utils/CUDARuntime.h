#ifndef UTILS_MLIR_UTILS_CUDARUNTIME_H
#define UTILS_MLIR_UTILS_CUDARUNTIME_H

#include <string>

#include "llvm/ADT/StringRef.h"

namespace mlir::north_star {

std::string getCUDARuntimeBridgeName(llvm::StringRef kernelName);
llvm::StringRef getCUDAUnifiedLaunchAPIName();

}  // namespace mlir::north_star

#endif  // UTILS_MLIR_UTILS_CUDARUNTIME_H
