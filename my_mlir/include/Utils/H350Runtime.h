#ifndef UTILS_MLIR_UTILS_H350RUNTIME_H
#define UTILS_MLIR_UTILS_H350RUNTIME_H

#include <string>

#include "llvm/ADT/StringRef.h"

namespace mlir::north_star {

std::string getH350RuntimeBridgeName(llvm::StringRef kernelName);
llvm::StringRef getH350UnifiedLaunchAPIName();

}  // namespace mlir::north_star

#endif  // UTILS_MLIR_UTILS_H350RUNTIME_H
