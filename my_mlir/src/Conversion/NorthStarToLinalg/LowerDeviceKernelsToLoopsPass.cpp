#include "Conversion/NorthStarToLinalg/NorthStarToLinalg.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::north_star {

#define GEN_PASS_DEF_LOWERNORTHSTARDEVICEKERNELSTOLOOPSPASS
#include "Conversion/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kKernelAttr = "north_star.kernel";

struct LowerNorthStarDeviceKernelsToLoopsPass
    : public impl::LowerNorthStarDeviceKernelsToLoopsPassBase<
          LowerNorthStarDeviceKernelsToLoopsPass> {
  void runOnOperation() override;
};

void LowerNorthStarDeviceKernelsToLoopsPass::runOnOperation() {
  SmallVector<func::FuncOp> kernelFuncs;
  getOperation().walk([&](func::FuncOp func) {
    if (func->hasAttr(kKernelAttr))
      kernelFuncs.push_back(func);
  });

  for (func::FuncOp func : kernelFuncs) {
    SmallVector<UnrealizedConversionCastOp> identityCasts;
    func.walk([&](UnrealizedConversionCastOp castOp) {
      if (castOp->getNumOperands() != 1 || castOp->getNumResults() != 1)
        return;
      if (castOp.getOperands().front().getType() != castOp.getResult(0).getType())
        return;
      identityCasts.push_back(castOp);
    });
    for (UnrealizedConversionCastOp castOp : identityCasts) {
      castOp.getResult(0).replaceAllUsesWith(castOp.getOperands().front());
      castOp.erase();
    }

    bufferization::OneShotBufferizationOptions options;
    options.bufferizeFunctionBoundaries = false;
    bufferization::BufferizationState state;
    if (failed(bufferization::runOneShotBufferize(func, options, state))) {
      func.emitError("failed to bufferize outlined device kernel");
      signalPassFailure();
      return;
    }

    OpPassManager pipeline(func::FuncOp::getOperationName());
    pipeline.addPass(createConvertLinalgToLoopsPass());
    if (failed(runPipeline(pipeline, func.getOperation()))) {
      signalPassFailure();
      return;
    }
  }
}

}  // namespace
}  // namespace mlir::north_star
