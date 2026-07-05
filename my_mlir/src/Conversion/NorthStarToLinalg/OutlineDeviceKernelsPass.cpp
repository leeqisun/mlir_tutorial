

#include "Conversion/NorthStarToLinalg/NorthStarToLinalg.h"

#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "Utils/Key.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::north_star {

#define GEN_PASS_DEF_OUTLINENORTHSTARDEVICEKERNELSPASS
#include "Conversion/Passes.h.inc"

namespace {

struct OutlineNorthStarDeviceKernelsPass
    : public impl::OutlineNorthStarDeviceKernelsPassBase<
          OutlineNorthStarDeviceKernelsPass> {
  void runOnOperation() override;
};

static std::string buildKernelName(DeviceKernelOp op, SymbolTable &symbolTable) {
  std::string base = (op.getSymName() + "_device_" + llvm::utostr(op.getDeviceId())).str();
  std::string candidate = base;
  unsigned suffix = 0;
  while (symbolTable.lookup(candidate))
    candidate = base + "_" + llvm::utostr(++suffix);
  return candidate;
}

static UnrealizedConversionCastOp getBoundaryArgCast(BlockArgument arg) {
  if (!arg.hasOneUse())
    return {};
  auto castOp =
      dyn_cast<UnrealizedConversionCastOp>(*arg.getUsers().begin());
  if (!castOp || castOp->getNumOperands() != 1 || castOp->getNumResults() != 1)
    return {};
  if (castOp.getOperands().front() != arg)
    return {};
  if (!isa<TensorType>(castOp.getResult(0).getType()))
    return {};
  return castOp;
}

static UnrealizedConversionCastOp getBoundaryResultCast(Value value) {
  auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>();
  if (!castOp || castOp->getNumOperands() != 1 || castOp->getNumResults() != 1)
    return {};
  if (!isa<TensorType>(castOp.getOperands().front().getType()))
    return {};
  return castOp;
}

static LogicalResult outlineKernel(DeviceKernelOp op, SymbolTable &symbolTable) {
  auto *block = op.getRegion().empty() ? nullptr : &op.getRegion().front();
  if (!block)
    return op.emitOpError("expected a single non-empty region");
  auto ret = dyn_cast<north_star::ReturnOp>(block->getTerminator());
  if (!ret)
    return op.emitOpError("expected north_star.return terminator");

  SmallVector<Type> kernelArgTypes;
  SmallVector<UnrealizedConversionCastOp> argCasts;
  for (BlockArgument arg : block->getArguments()) {
    auto castOp = getBoundaryArgCast(arg);
    argCasts.push_back(castOp);
    kernelArgTypes.push_back(castOp ? castOp.getResult(0).getType() : arg.getType());
  }

  SmallVector<Type> kernelResultTypes;
  SmallVector<Value> kernelReturnValues;
  SmallVector<Operation *> skipOps;
  for (Value operand : ret.getOperands()) {
    auto castOp = getBoundaryResultCast(operand);
    if (castOp) {
      kernelResultTypes.push_back(castOp.getOperands().front().getType());
      kernelReturnValues.push_back(castOp.getOperands().front());
      skipOps.push_back(castOp.getOperation());
      continue;
    }
    kernelResultTypes.push_back(operand.getType());
    kernelReturnValues.push_back(operand);
  }

  OpBuilder moduleBuilder(op->getParentOfType<ModuleOp>().getBodyRegion());
  auto funcType = moduleBuilder.getFunctionType(kernelArgTypes, kernelResultTypes);
  auto funcName = buildKernelName(op, symbolTable);
  auto kernelFunc = func::FuncOp::create(op.getLoc(), funcName, funcType);
  kernelFunc.setPrivate();
  kernelFunc->setAttr(KNPUKernelAttr, moduleBuilder.getUnitAttr());
  kernelFunc->setAttr(KNPUTargetAttr, moduleBuilder.getStringAttr(KH350NPUTarget));
  kernelFunc->setAttr("device_id", moduleBuilder.getI64IntegerAttr(op.getDeviceId()));
  symbolTable.insert(kernelFunc);

  Block *entry = kernelFunc.addEntryBlock();
  OpBuilder funcBuilder = OpBuilder::atBlockBegin(entry);
  IRMapping mapping;
  for (auto [index, arg] : llvm::enumerate(block->getArguments())) {
    auto outlinedArg = entry->getArgument(index);
    mapping.map(arg, outlinedArg);
    if (argCasts[index])
      mapping.map(argCasts[index].getResult(0), outlinedArg);
  }

  llvm::SmallDenseSet<Operation *> skipSet(skipOps.begin(), skipOps.end());
  for (Operation &nestedOp : llvm::make_early_inc_range(*block)) {
    if (&nestedOp == ret.getOperation())
      continue;
    if (skipSet.contains(&nestedOp))
      continue;
    funcBuilder.clone(nestedOp, mapping);
  }

  SmallVector<Value> returnOperands;
  returnOperands.reserve(kernelReturnValues.size());
  for (Value value : kernelReturnValues)
    returnOperands.push_back(mapping.lookupOrDefault(value));
  funcBuilder.create<func::ReturnOp>(op.getLoc(), returnOperands);

  OpBuilder callBuilder(op);
  SmallVector<Value> callOperands;
  callOperands.reserve(op.getArgs().size());
  for (auto [operand, targetType] : llvm::zip(op.getArgs(), kernelArgTypes)) {
    if (operand.getType() == targetType) {
      callOperands.push_back(operand);
      continue;
    }
    callOperands.push_back(callBuilder
                               .create<UnrealizedConversionCastOp>(op.getLoc(),
                                                                   targetType,
                                                                   operand)
                               .getResult(0));
  }

  auto call = callBuilder.create<func::CallOp>(op.getLoc(), kernelFunc,
                                               callOperands);
  SmallVector<Value> replacements;
  replacements.reserve(op.getNumResults());
  for (auto [idx, result] : llvm::enumerate(call.getResults())) {
    if (result.getType() == op->getResult(idx).getType()) {
      replacements.push_back(result);
      continue;
    }
    replacements.push_back(callBuilder
                               .create<UnrealizedConversionCastOp>(
                                   op.getLoc(), op->getResult(idx).getType(),
                                   result)
                               .getResult(0));
  }

  op->replaceAllUsesWith(replacements);
  op.erase();
  return success();
}

void OutlineNorthStarDeviceKernelsPass::runOnOperation() {
  auto module = getOperation();
  SymbolTable symbolTable(module);
  SmallVector<DeviceKernelOp> kernels;
  module.walk([&](DeviceKernelOp op) { kernels.push_back(op); });
  for (DeviceKernelOp op : kernels) {
    if (failed(outlineKernel(op, symbolTable))) {
      signalPassFailure();
      return;
    }
  }
}

}  // namespace
}  // namespace mlir::north_star
