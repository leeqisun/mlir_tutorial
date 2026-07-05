#include "Conversion/NorthStarToLinalg/NorthStarToLinalg.h"

#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Utils/H350Runtime.h"
#include "Utils/Key.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::north_star {

#define GEN_PASS_DEF_LOWERNORTHSTARRUNTIMESTUBSTOH350CALLSPASS
#include "Conversion/Passes.h.inc"

namespace {

struct LowerNorthStarRuntimeStubsToH350CallsPass
    : public impl::LowerNorthStarRuntimeStubsToH350CallsPassBase<
          LowerNorthStarRuntimeStubsToH350CallsPass> {
  void runOnOperation() override;
};

static func::FuncOp getOrCreateBridgeDecl(func::FuncOp runtimeStub,
                                          SymbolTable &symbolTable,
                                          int64_t descriptorId) {
  auto kernelName =
      runtimeStub->getAttrOfType<StringAttr>(KNPURuntimeKernelNameAttr);
  std::string bridgeName = getH350RuntimeBridgeName(kernelName.getValue());
  if (auto bridge = symbolTable.lookup<func::FuncOp>(bridgeName))
    return bridge;

  OpBuilder builder(runtimeStub->getParentOfType<ModuleOp>().getBodyRegion());
  auto bridge = builder.create<func::FuncOp>(runtimeStub.getLoc(), bridgeName,
                                             runtimeStub.getFunctionType());
  bridge.setPrivate();
  bridge->setAttr(KNPURuntimeBridgeAttr, builder.getUnitAttr());
  bridge->setAttr(KNPURuntimeKernelNameAttr, kernelName);
  bridge->setAttr(KNPURuntimeDescriptorIdAttr,
                  builder.getI64IntegerAttr(descriptorId));
  if (auto target = runtimeStub->getAttr(KNPUTargetAttr))
    bridge->setAttr(KNPUTargetAttr, target);
  if (auto deviceId = runtimeStub->getAttr("device_id"))
    bridge->setAttr("device_id", deviceId);
  symbolTable.insert(bridge);
  return bridge;
}

static func::FuncOp getOrCreateUnifiedLaunchApi(ModuleOp module,
                                                SymbolTable &symbolTable) {
  auto apiName = getH350UnifiedLaunchAPIName();
  if (auto api = symbolTable.lookup<func::FuncOp>(apiName))
    return api;

  OpBuilder builder(module.getBodyRegion());
  auto unrankedF32Tensor = UnrankedTensorType::get(builder.getF32Type());
  auto apiType = builder.getFunctionType(
      TypeRange{builder.getI64Type(), unrankedF32Tensor},
      TypeRange{unrankedF32Tensor});
  auto api = builder.create<func::FuncOp>(module.getLoc(), apiName, apiType);
  api.setPrivate();
  api->setAttr(KNPURuntimeCApiAttr, builder.getUnitAttr());
  api->setAttr(KNPUTargetAttr, builder.getStringAttr(KH350NPUTarget));
  symbolTable.insert(api);
  return api;
}

static LogicalResult materializeBridgeBody(func::FuncOp bridge,
                                           func::FuncOp unifiedApi,
                                           int64_t descriptorId) {
  if (!bridge.empty())
    bridge.eraseBody();

  if (bridge.getFunctionType().getNumInputs() != 1 ||
      bridge.getFunctionType().getNumResults() != 1)
    return bridge.emitError("only 1-input/1-output kernels are supported");

  auto inputType = dyn_cast<TensorType>(bridge.getFunctionType().getInput(0));
  auto resultType = dyn_cast<TensorType>(bridge.getFunctionType().getResult(0));
  if (!inputType || !resultType)
    return bridge.emitError("bridge expects tensor input/output types");
  if (!inputType.getElementType().isF32() || !resultType.getElementType().isF32())
    return bridge.emitError("bridge currently supports f32 tensors only");

  Block *entryBlock = bridge.addEntryBlock();
  OpBuilder builder(bridge.getContext());
  builder.setInsertionPointToStart(entryBlock);
  auto descriptor = builder.create<arith::ConstantOp>(
      bridge.getLoc(), builder.getI64IntegerAttr(descriptorId));
  auto erasedInput = builder.create<tensor::CastOp>(
      bridge.getLoc(), UnrankedTensorType::get(builder.getF32Type()),
      entryBlock->getArgument(0));
  auto apiCall = builder.create<func::CallOp>(
      bridge.getLoc(), unifiedApi,
      ValueRange{descriptor.getResult(), erasedInput.getResult()});
  auto rankedResult = builder.create<tensor::CastOp>(bridge.getLoc(), resultType,
                                                     apiCall.getResult(0));
  builder.create<func::ReturnOp>(bridge.getLoc(), rankedResult.getResult());
  return success();
}

void LowerNorthStarRuntimeStubsToH350CallsPass::runOnOperation() {
  auto module = getOperation();
  SymbolTable symbolTable(module);
  SmallVector<func::FuncOp> runtimeStubs;
  module.walk([&](func::FuncOp func) {
    if (func->hasAttr(KNPURuntimeStubAttr))
      runtimeStubs.push_back(func);
  });
  llvm::sort(runtimeStubs, [](func::FuncOp lhs, func::FuncOp rhs) {
    return lhs.getSymName() < rhs.getSymName();
  });
  auto unifiedApi = getOrCreateUnifiedLaunchApi(module, symbolTable);

  for (auto [descriptorId, runtimeStub] : llvm::enumerate(runtimeStubs)) {
    auto kernelName =
        runtimeStub->getAttrOfType<StringAttr>(KNPURuntimeKernelNameAttr);
    if (!kernelName) {
      runtimeStub.emitError("runtime stub is missing kernel name metadata");
      signalPassFailure();
      return;
    }

    runtimeStub->setAttr(KNPURuntimeDescriptorIdAttr,
                         IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                          descriptorId));
    auto bridge =
        getOrCreateBridgeDecl(runtimeStub, symbolTable, descriptorId);
    if (failed(materializeBridgeBody(bridge, unifiedApi, descriptorId))) {
      signalPassFailure();
      return;
    }
    if (!runtimeStub.empty())
      runtimeStub.eraseBody();

    Block *entryBlock = runtimeStub.addEntryBlock();
    OpBuilder builder(runtimeStub.getContext());
    builder.setInsertionPointToStart(entryBlock);
    auto bridgeCall = builder.create<func::CallOp>(
        runtimeStub.getLoc(), bridge, entryBlock->getArguments());
    builder.create<func::ReturnOp>(runtimeStub.getLoc(), bridgeCall.getResults());
  }
}

}  // namespace
}  // namespace mlir::north_star
