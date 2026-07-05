#include "Conversion/NorthStarToLinalg/NorthStarToLinalg.h"

#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "Utils/Key.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::north_star {

#define GEN_PASS_DEF_LOWERNORTHSTARHOSTTONPUCALLSPASS
#include "Conversion/Passes.h.inc"

namespace {

struct LowerNorthStarHostToNPUCallsPass
    : public impl::LowerNorthStarHostToNPUCallsPassBase<
          LowerNorthStarHostToNPUCallsPass> {
  void runOnOperation() override;
};

void LowerNorthStarHostToNPUCallsPass::runOnOperation() {
  auto module = getOperation();
  SymbolTable symbolTable(module);
  SmallVector<func::CallOp> hostCalls;
  module.walk([&](func::CallOp call) { hostCalls.push_back(call); });

  for (func::CallOp call : hostCalls) {
    auto callee = symbolTable.lookup<func::FuncOp>(call.getCallee());
    if (!callee || !callee->hasAttr(KNPUKernelAttr))
      continue;

    auto targetAttr =
        callee->getAttrOfType<StringAttr>(KNPUTargetAttr);
    auto deviceAttr = callee->getAttrOfType<IntegerAttr>("device_id");
    if (!targetAttr || !deviceAttr) {
      call.emitError("outlined kernel call is missing target/device metadata");
      signalPassFailure();
      return;
    }

    OpBuilder builder(call);
    auto npuCall = builder.create<north_star::NPUCallOp>(
        call.getLoc(), call.getResultTypes(),
        FlatSymbolRefAttr::get(callee.getSymNameAttr()),
        targetAttr, deviceAttr, call.getOperands());
    call.replaceAllUsesWith(npuCall.getResults());
    call.erase();
  }
}

}  // namespace
}  // namespace mlir::north_star
