#ifndef CONVERSION_PASSES_H
#define CONVERSION_PASSES_H
#include "mlir/Pass/Pass.h"
namespace mlir::north_star {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"
}  // namespace mlir::north_star

#endif  // CONVERSION_PASSES_H
