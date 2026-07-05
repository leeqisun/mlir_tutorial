#include "Conversion/NorthStarToLinalg/NorthStarToLinalg.h"

#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Utils/Key.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::north_star {

#define GEN_PASS_DEF_MATERIALIZENORTHSTARRUNTIMESTUBSPASS
#include "Conversion/Passes.h.inc"

namespace {

struct MaterializeNorthStarRuntimeStubsPass
    : public impl::MaterializeNorthStarRuntimeStubsPassBase<
          MaterializeNorthStarRuntimeStubsPass> {
  void runOnOperation() override;
};

void MaterializeNorthStarRuntimeStubsPass::runOnOperation() {
  auto module = getOperation();
  SymbolTable symbolTable(module);
  SmallVector<func::FuncOp> runtimeStubs;
  module.walk([&](func::FuncOp func) {
    if (func->hasAttr(KNPURuntimeStubAttr))
      runtimeStubs.push_back(func);
  });

  for (func::FuncOp stub : runtimeStubs) {
    if (!stub.empty())
      continue;

    auto kernelName =
        stub->getAttrOfType<StringAttr>(KNPURuntimeKernelNameAttr);
    if (!kernelName) {
      stub.emitError("runtime stub is missing kernel name metadata");
      signalPassFailure();
      return;
    }

    auto kernel = symbolTable.lookup<func::FuncOp>(kernelName.getValue());
    if (!kernel) {
      stub.emitError("cannot find kernel referenced by runtime stub: ")
          << kernelName.getValue();
      signalPassFailure();
      return;
    }

    Block *entryBlock = stub.addEntryBlock();
    OpBuilder builder(stub.getContext());
    builder.setInsertionPointToStart(entryBlock);
    auto call = builder.create<func::CallOp>(stub.getLoc(), kernel,
                                             entryBlock->getArguments());
    builder.create<func::ReturnOp>(stub.getLoc(), call.getResults());
  }
}

}  // namespace
}  // namespace mlir::north_star
