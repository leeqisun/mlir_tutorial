#include "Conversion/NorthStarToLinalg/NorthStarToLinalg.h"

#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "Utils/Key.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::north_star {

#define GEN_PASS_DEF_LOWERNORTHSTARNPUCALLSTORUNTIMEPASS
#include "Conversion/Passes.h.inc"

namespace {

struct LowerNorthStarNPUCallsToRuntimePass
    : public impl::LowerNorthStarNPUCallsToRuntimePassBase<
          LowerNorthStarNPUCallsToRuntimePass> {
  void runOnOperation() override;
};

static func::FuncOp getOrCreateRuntimeStub(ModuleOp module,
                                           SymbolTable &symbolTable,
                                           north_star::NPUCallOp npuCall) {
  auto calleeName = npuCall.getCalleeAttr().getValue();
  std::string stubName =
      (llvm::Twine(KNPURuntimeStubPrefix) + calleeName).str();
  if (auto stub = symbolTable.lookup<func::FuncOp>(stubName))
    return stub;

  OpBuilder builder(module.getBodyRegion());
  auto stub = builder.create<func::FuncOp>(
      npuCall.getLoc(), stubName,
      builder.getFunctionType(npuCall.getOperandTypes(), npuCall.getResultTypes()));
  stub.setPrivate();
  stub->setAttr(KNPURuntimeStubAttr, builder.getUnitAttr());
  stub->setAttr(KNPURuntimeKernelNameAttr,
                builder.getStringAttr(calleeName));
  stub->setAttr(KNPUTargetAttr, npuCall.getTargetAttr());
  stub->setAttr("device_id", npuCall.getDeviceIdAttr());
  symbolTable.insert(stub);
  return stub;
}

void LowerNorthStarNPUCallsToRuntimePass::runOnOperation() {
  auto module = getOperation();
  SymbolTable symbolTable(module);
  SmallVector<north_star::NPUCallOp> npuCalls;
  module.walk([&](north_star::NPUCallOp npuCall) { npuCalls.push_back(npuCall); });

  for (north_star::NPUCallOp npuCall : npuCalls) {
    auto stub = getOrCreateRuntimeStub(module, symbolTable, npuCall);
    OpBuilder builder(npuCall);
    auto runtimeCall = builder.create<func::CallOp>(
        npuCall.getLoc(), npuCall.getResultTypes(), stub.getSymName(),
        npuCall.getInputs());
    npuCall.replaceAllUsesWith(runtimeCall.getResults());
    npuCall.erase();
  }
}

}  // namespace
}  // namespace mlir::north_star
