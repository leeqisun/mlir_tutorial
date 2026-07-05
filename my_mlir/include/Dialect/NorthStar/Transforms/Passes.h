#ifndef DIALECT_NORTH_STAR_TRANSFORMS_PASSES_H
#define DIALECT_NORTH_STAR_TRANSFORMS_PASSES_H
#include "mlir/Pass/Pass.h"
namespace mlir::north_star {
void populateBufferCastOpCanonicalizationPatterns(RewritePatternSet &patterns);

void populateDeviceRegionFusionPatterns(RewritePatternSet &patterns);

std::unique_ptr<::mlir::Pass> createApplyDistributeTransformPass();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Dialect/NorthStar/Transforms/Passes.h.inc"
}  // namespace mlir::north_star

#endif  // DIALECT_NORTH_STAR_TRANSFORMS_PASSES_H