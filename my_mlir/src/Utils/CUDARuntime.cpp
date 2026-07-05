#include "Utils/CUDARuntime.h"

#include "Utils/Key.h"
#include "llvm/ADT/Twine.h"

namespace mlir::north_star {

std::string getCUDARuntimeBridgeName(llvm::StringRef kernelName) {
  return (llvm::Twine(KNPURuntimeCUDABridgePrefix) + kernelName).str();
}

llvm::StringRef getCUDAUnifiedLaunchAPIName() { return KCUDAUnifiedLaunchAPI; }

}  // namespace mlir::north_star
