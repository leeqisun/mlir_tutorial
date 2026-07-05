

#include <memory>

#include "Conversion/NorthStarToLinalg/NorthStarToLinalg.h"
#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "convert-north-satr-to-linalg"

namespace mlir::north_star {

#define GEN_PASS_DEF_CONVERTNORTHSTARTOLINALGPASS
#include "Conversion/Passes.h.inc"

}  // namespace mlir::north_star

using namespace ::mlir;
using namespace ::mlir::north_star;

struct NorthStarToLinalgPassPass
    : public mlir::north_star::impl::ConvertNorthStarToLinalgPassBase<
          NorthStarToLinalgPassPass> {
  void runOnOperation() override;
};

void configNorthStarToLinalgTarget(ConversionTarget& target) {
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<linalg::LinalgDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<math::MathDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalOp<BufferCastOp>();
  target.addDynamicallyLegalOp<ReturnOp>([](ReturnOp op) {
    for (auto type : op->getOperandTypes()) {
      if (isa<::mlir::north_star::NSTensorType>(type)) return false;
    }
    return true;
  });
  target.addDynamicallyLegalOp<DeviceKernelOp>([](DeviceKernelOp op) {
  auto hasNSTensor = [](Type t){ return isa<north_star::NSTensorType>(t); };
  return llvm::none_of(op.getOperandTypes(), hasNSTensor) &&
         llvm::none_of(op.getResultTypes(), hasNSTensor);
});

  target.addLegalOp<UnrealizedConversionCastOp, BufferCastOp>();
  target.addLegalOp<DeviceKernelOp, ReturnOp>();
  target.addDynamicallyLegalOp<SoftmaxOp>([](Operation* op) {
    return !llvm::isa<DeviceKernelOp>(op->getParentOp());
  });
}
void NorthStarToLinalgPassPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in {0}\n", getPassName()));
  auto module = getOperation();
  TypeConverter type_convert;
  initNorthStarToLinalgTypeConvert(type_convert);
  RewritePatternSet patterns(&getContext());
  populateNorthStarToLinalgPatterns(type_convert, patterns);
  ConversionTarget target(getContext());
  configNorthStarToLinalgTarget(target);
  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyPartialConversion(module, target, std::move(patterns), config)))
    signalPassFailure();
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
}
