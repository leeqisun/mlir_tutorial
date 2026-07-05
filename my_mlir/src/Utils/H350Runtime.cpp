#include "Utils/H350Runtime.h"

#include "Utils/Key.h"
#include "llvm/ADT/Twine.h"

namespace mlir::north_star {

std::string getH350RuntimeBridgeName(llvm::StringRef kernelName) {
  return (llvm::Twine(KNPURuntimeBridgePrefix) + kernelName).str();
}

llvm::StringRef getH350UnifiedLaunchAPIName() { return KH350UnifiedLaunchAPI; }

}  // namespace mlir::north_star
