#include "Conversion/NorthStarToLinalg/NorthStarToLinalg.h"

#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Utils/Key.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::north_star {

#define GEN_PASS_DEF_LOWERNORTHSTARDEVICEKERNELSTOGPUPASS
#include "Conversion/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kKernelAttr = "north_star.kernel";
constexpr llvm::StringLiteral kGPUModuleName = "northstar_cuda_kernels";
constexpr llvm::StringLiteral kKernelEntrySuffix = "__kernel";
constexpr llvm::StringLiteral kLaunchWrapperSuffix = "__gpu_launch";

struct LowerNorthStarDeviceKernelsToGPUPass
    : public impl::LowerNorthStarDeviceKernelsToGPUPassBase<
          LowerNorthStarDeviceKernelsToGPUPass> {
  void runOnOperation() override;
};

static bool hasBufferedBoundary(func::FuncOp func) {
  bool buffered = false;
  func.walk([&](bufferization::ToBufferOp) { buffered = true; });
  return buffered;
}

static gpu::GPUModuleOp getOrCreateGPUModule(ModuleOp module,
                                             SymbolTable &symbolTable) {
  if (!module->hasAttr(gpu::GPUDialect::getContainerModuleAttrName()))
    module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                    UnitAttr::get(module.getContext()));

  if (auto gpuModule = symbolTable.lookup<gpu::GPUModuleOp>(kGPUModuleName))
    return gpuModule;

  OpBuilder builder(module.getBodyRegion());
  auto gpuModule = gpu::GPUModuleOp::create(builder, module.getLoc(),
                                            kGPUModuleName, ArrayRef<Attribute>{},
                                            Attribute{});
  gpuModule->setAttr(KNPUTargetAttr,
                     builder.getStringAttr(KRTX5060CUDATarget));
  symbolTable.insert(gpuModule);
  return gpuModule;
}

static void cloneKernelBody(func::FuncOp src, gpu::GPUFuncOp dst) {
  if (!dst.getBody().empty())
    dst.getBody().front().erase();

  Block *entry = dst.addEntryBlock();
  IRMapping mapping;
  for (auto [srcArg, dstArg] :
       llvm::zip_equal(src.getArguments(), entry->getArguments()))
    mapping.map(srcArg, dstArg);

  OpBuilder builder(dst.getBody());
  builder.setInsertionPointToStart(entry);
  for (Operation &op : src.getBody().front().without_terminator())
    builder.clone(op, mapping);

  auto returnOp = cast<func::ReturnOp>(src.getBody().front().getTerminator());
  SmallVector<Value> returnValues;
  returnValues.reserve(returnOp.getNumOperands());
  for (Value operand : returnOp.getOperands())
    returnValues.push_back(mapping.lookup(operand));
  builder.create<gpu::ReturnOp>(src.getLoc(), returnValues);
}

static void materializeKernelEntryBody(func::FuncOp src, gpu::GPUFuncOp dst) {
  if (!dst.getBody().empty())
    dst.getBody().front().erase();

  Block *entry = dst.addEntryBlock();
  OpBuilder builder(dst.getBody());
  builder.setInsertionPointToStart(entry);
  auto srcType = src.getFunctionType();
  size_t numInputs = srcType.getNumInputs();
  size_t numResults = srcType.getNumResults();

  bool bufferedBoundary = hasBufferedBoundary(src);

  if (bufferedBoundary) {
    IRMapping mapping;
    for (Operation &op : src.getBody().front().without_terminator()) {
      if (auto toBuffer = dyn_cast<bufferization::ToBufferOp>(op)) {
        if (auto blockArg = dyn_cast<BlockArgument>(toBuffer.getTensor())) {
          if (blockArg.getOwner() == &src.getBody().front() &&
              static_cast<size_t>(blockArg.getArgNumber()) < numInputs) {
            mapping.map(toBuffer.getResult(),
                        entry->getArgument(blockArg.getArgNumber()));
            continue;
          }
        }
      }
      if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op)) {
        mapping.map(toTensor.getResult(),
                    mapping.lookupOrDefault(toTensor.getBuffer()));
        continue;
      }
      builder.clone(op, mapping);
    }

    auto returnOp = cast<func::ReturnOp>(src.getBody().front().getTerminator());
    for (auto [result, outputBuffer] :
         llvm::zip_equal(returnOp.getOperands(),
                         entry->getArguments().drop_front(numInputs).take_front(
                             numResults))) {
      Value mapped = mapping.lookupOrDefault(result);
      if (!isa<BaseMemRefType>(mapped.getType())) {
        if (auto toTensor =
                result.getDefiningOp<bufferization::ToTensorOp>()) {
          mapped = mapping.lookupOrDefault(toTensor.getBuffer());
        }
      }
      if (isa<BaseMemRefType>(mapped.getType())) {
        builder.create<memref::CopyOp>(src.getLoc(), mapped, outputBuffer);
        continue;
      }
      builder.create<bufferization::MaterializeInDestinationOp>(
          src.getLoc(), TypeRange{}, mapped, outputBuffer, /*restrict=*/true,
          /*writable=*/true);
    }
    builder.create<gpu::ReturnOp>(src.getLoc());
    return;
  }

  IRMapping mapping;
  for (auto [srcArg, dstArg, type] :
       llvm::zip_equal(src.getArguments(),
                       entry->getArguments().take_front(numInputs),
                       srcType.getInputs())) {
    auto tensorArg =
        builder.create<bufferization::ToTensorOp>(src.getLoc(), type, dstArg);
    mapping.map(srcArg, tensorArg.getResult());
  }

  for (Operation &op : src.getBody().front().without_terminator())
    builder.clone(op, mapping);

  auto returnOp = cast<func::ReturnOp>(src.getBody().front().getTerminator());
  for (auto [result, outputBuffer] :
       llvm::zip_equal(returnOp.getOperands(),
                       entry->getArguments().drop_front(numInputs).take_front(
                           numResults))) {
    builder.create<bufferization::MaterializeInDestinationOp>(
        src.getLoc(), TypeRange{}, mapping.lookup(result), outputBuffer,
        /*restrict=*/true, /*writable=*/true);
  }
  builder.create<gpu::ReturnOp>(src.getLoc());
}

static FailureOr<MemRefType> getMemRefType(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType)
    return failure();
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
}

static FailureOr<SmallVector<Type>> getMemRefTypes(TypeRange types) {
  SmallVector<Type> memrefTypes;
  memrefTypes.reserve(types.size());
  for (Type type : types) {
    FailureOr<MemRefType> memrefType = getMemRefType(type);
    if (failed(memrefType))
      return failure();
    memrefTypes.push_back(*memrefType);
  }
  return memrefTypes;
}

static FailureOr<SmallVector<Value>> getDynamicDimsForTensorLikeResult(
    OpBuilder &builder, Location loc, ValueRange tensorInputs,
    RankedTensorType resultType) {
  SmallVector<Value> dynamicDims;
  if (!resultType.hasStaticShape()) {
    if (tensorInputs.empty())
      return failure();
    for (int64_t dim = 0, e = resultType.getRank(); dim < e; ++dim) {
      if (resultType.isDynamicDim(dim))
        dynamicDims.push_back(
            builder.create<tensor::DimOp>(loc, tensorInputs.front(), dim));
    }
  }
  return dynamicDims;
}

static FailureOr<func::FuncOp> getOrCreateLaunchWrapper(ModuleOp module,
                                                        SymbolTable &symbolTable,
                                                        func::FuncOp func) {
  std::string wrapperName = (func.getSymName() + kLaunchWrapperSuffix).str();
  if (auto wrapper = symbolTable.lookup<func::FuncOp>(wrapperName))
    return wrapper;

  auto context = module.getContext();
  FailureOr<SmallVector<Type>> inputMemRefTypes =
      getMemRefTypes(func.getFunctionType().getInputs());
  FailureOr<SmallVector<Type>> resultMemRefTypes =
      getMemRefTypes(func.getFunctionType().getResults());
  if (failed(inputMemRefTypes) || failed(resultMemRefTypes)) {
    func.emitError("expected ranked tensor inputs/results for GPU launch wrapper");
    return failure();
  }

  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPoint(func);
  auto wrapper = builder.create<func::FuncOp>(func.getLoc(), wrapperName,
                                              func.getFunctionType());
  wrapper.setPrivate();
  wrapper->setAttr(KNPUTargetAttr,
                   builder.getStringAttr(KRTX5060CUDATarget));

  Block *entry = wrapper.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  SmallVector<Value> launchArgs;
  launchArgs.reserve(inputMemRefTypes->size() + resultMemRefTypes->size());
  for (auto [arg, memrefType] :
       llvm::zip_equal(entry->getArguments(), *inputMemRefTypes)) {
    launchArgs.push_back(
        builder.create<bufferization::ToBufferOp>(func.getLoc(), memrefType, arg));
  }

  SmallVector<Value> resultBuffers;
  resultBuffers.reserve(resultMemRefTypes->size());
  for (auto [resultType, memrefType] :
       llvm::zip_equal(func.getFunctionType().getResults(), *resultMemRefTypes)) {
    auto rankedResultType = cast<RankedTensorType>(resultType);
    FailureOr<SmallVector<Value>> dynamicDims = getDynamicDimsForTensorLikeResult(
        builder, func.getLoc(), entry->getArguments(), rankedResultType);
    if (failed(dynamicDims)) {
      func.emitError("cannot infer dynamic dimensions for GPU launch result");
      wrapper.erase();
      return failure();
    }
    auto alloc = memref::AllocOp::create(builder, func.getLoc(),
                                         cast<MemRefType>(memrefType),
                                         *dynamicDims, ValueRange{});
    resultBuffers.push_back(alloc);
    launchArgs.push_back(alloc);
  }

  Value c1 = builder.create<arith::ConstantIndexOp>(func.getLoc(), 1);
  Value c256 = builder.create<arith::ConstantIndexOp>(func.getLoc(), 256);
  Value c0 = builder.create<arith::ConstantIntOp>(func.getLoc(), 0, 32);
  auto kernelRef = SymbolRefAttr::get(
      context, kGPUModuleName,
      {FlatSymbolRefAttr::get(context,
                              (func.getSymName() + kKernelEntrySuffix).str())});
  gpu::LaunchFuncOp::create(builder, func.getLoc(), kernelRef,
                            gpu::KernelDim3{c1, c1, c1},
                            gpu::KernelDim3{c256, c1, c1}, c0, launchArgs);

  SmallVector<Value> tensorResults;
  tensorResults.reserve(resultBuffers.size());
  for (auto [buffer, resultType] :
       llvm::zip_equal(resultBuffers, func.getFunctionType().getResults())) {
    tensorResults.push_back(builder.create<bufferization::ToTensorOp>(
        func.getLoc(), resultType, buffer));
  }
  builder.create<func::ReturnOp>(func.getLoc(), tensorResults);
  return wrapper;
}

static LogicalResult rewriteHostCalls(ModuleOp module, func::FuncOp kernelFunc,
                                      func::FuncOp launchWrapper) {
  SmallVector<func::CallOp> callsToRewrite;
  module.walk([&](func::CallOp call) {
    if (call.getCallee() != kernelFunc.getSymName())
      return;
    if (call->getParentOfType<gpu::GPUModuleOp>())
      return;
    callsToRewrite.push_back(call);
  });

  for (func::CallOp call : callsToRewrite) {
    OpBuilder builder(call);
    auto newCall =
        builder.create<func::CallOp>(call.getLoc(), launchWrapper, call.getOperands());
    call.replaceAllUsesWith(newCall.getResults());
    call.erase();
  }
  return success();
}

void LowerNorthStarDeviceKernelsToGPUPass::runOnOperation() {
  auto module = getOperation();
  SymbolTable symbolTable(module);
  SmallVector<func::FuncOp> kernelFuncs;
  module.walk([&](func::FuncOp func) {
    if (func->hasAttr(kKernelAttr))
      kernelFuncs.push_back(func);
  });

  if (kernelFuncs.empty())
    return;

  auto gpuModule = getOrCreateGPUModule(module, symbolTable);
  SymbolTable gpuSymbolTable(gpuModule);
  OpBuilder builder(gpuModule.getBodyRegion());
  builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

  for (func::FuncOp func : kernelFuncs) {
    SmallVector<int32_t> knownBlockSize = {256, 1, 1};
    SmallVector<int32_t> knownGridSize = {1, 1, 1};
    bool bufferedKernel = hasBufferedBoundary(func);
    if (!bufferedKernel &&
        !gpuSymbolTable.lookup<gpu::GPUFuncOp>(func.getSymName())) {
      auto gpuFunc = gpu::GPUFuncOp::create(
          builder, func.getLoc(), func.getSymName(), func.getFunctionType(),
          TypeRange{}, TypeRange{}, ArrayRef<NamedAttribute>{});
      gpuFunc.setKnownBlockSize(knownBlockSize);
      gpuFunc.setKnownGridSize(knownGridSize);
      gpuFunc->setAttr(KNPURuntimeKernelNameAttr,
                       builder.getStringAttr(func.getSymName()));
      gpuFunc->setAttr(KNPUTargetAttr,
                       builder.getStringAttr(KRTX5060CUDATarget));
      gpuFunc->setAttr(KGPUKernelABIAttr,
                       builder.getStringAttr("tensor_mirror"));
      if (auto deviceId = func->getAttr("device_id"))
        gpuFunc->setAttr("device_id", deviceId);
      cloneKernelBody(func, gpuFunc);
    }

    std::string kernelEntryName =
        (func.getSymName() + kKernelEntrySuffix).str();
    if (!gpuSymbolTable.lookup<gpu::GPUFuncOp>(kernelEntryName)) {
      FailureOr<SmallVector<Type>> inputMemRefTypes =
          getMemRefTypes(func.getFunctionType().getInputs());
      FailureOr<SmallVector<Type>> resultMemRefTypes =
          getMemRefTypes(func.getFunctionType().getResults());
      if (failed(inputMemRefTypes) || failed(resultMemRefTypes)) {
        func.emitError("expected ranked tensor inputs/results for GPU kernel entry");
        signalPassFailure();
        return;
      }

      SmallVector<Type> kernelArgTypes = *inputMemRefTypes;
      llvm::append_range(kernelArgTypes, *resultMemRefTypes);
      auto kernelEntryType =
          builder.getFunctionType(kernelArgTypes, TypeRange{});
      auto kernelEntry = gpu::GPUFuncOp::create(
          builder, func.getLoc(), kernelEntryName, kernelEntryType, TypeRange{},
          TypeRange{}, ArrayRef<NamedAttribute>{});
      kernelEntry.setKnownBlockSize(knownBlockSize);
      kernelEntry.setKnownGridSize(knownGridSize);
      kernelEntry->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                           builder.getUnitAttr());
      kernelEntry->setAttr(KNPURuntimeKernelNameAttr,
                           builder.getStringAttr(func.getSymName()));
      kernelEntry->setAttr(KNPUTargetAttr,
                           builder.getStringAttr(KRTX5060CUDATarget));
      kernelEntry->setAttr(KGPUKernelEntryAttr, builder.getUnitAttr());
      kernelEntry->setAttr(KGPUKernelABIAttr,
                           builder.getStringAttr("void_out_buffer"));
      if (auto deviceId = func->getAttr("device_id"))
        kernelEntry->setAttr("device_id", deviceId);
      materializeKernelEntryBody(func, kernelEntry);
    }

    FailureOr<func::FuncOp> launchWrapper =
        getOrCreateLaunchWrapper(module, symbolTable, func);
    if (failed(launchWrapper) ||
        failed(rewriteHostCalls(module, func, *launchWrapper))) {
      signalPassFailure();
      return;
    }
  }
}

}  // namespace
}  // namespace mlir::north_star
