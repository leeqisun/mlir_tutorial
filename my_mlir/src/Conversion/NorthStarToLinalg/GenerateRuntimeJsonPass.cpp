#include "Conversion/NorthStarToLinalg/NorthStarToLinalg.h"

#include <string>

#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Utils/Key.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::north_star {

#define GEN_PASS_DEF_GENERATENORTHSTARRUNTIMEJSONPASS
#include "Conversion/Passes.h.inc"

namespace {

struct GenerateNorthStarRuntimeJsonPass
    : public impl::GenerateNorthStarRuntimeJsonPassBase<
          GenerateNorthStarRuntimeJsonPass> {
  void runOnOperation() override;
};

static std::string stringifyType(Type type) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << type;
  return storage;
}

static llvm::json::Array buildShape(Type type) {
  llvm::json::Array shape;
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType)
    return shape;
  for (int64_t dim : shapedType.getShape())
    shape.push_back(ShapedType::isDynamic(dim) ? -1 : dim);
  return shape;
}

static llvm::json::Object buildTensorDesc(Type type) {
  llvm::json::Object obj;
  obj["shape"] = buildShape(type);
  obj["type"] = stringifyType(type);
  if (auto shapedType = dyn_cast<ShapedType>(type))
    obj["element_type"] = stringifyType(shapedType.getElementType());
  else
    obj["element_type"] = stringifyType(type);
  return obj;
}

static llvm::json::Array buildTensorDescArray(TypeRange types) {
  llvm::json::Array descs;
  for (Type type : types)
    descs.push_back(buildTensorDesc(type));
  return descs;
}

static bool isRuntimeBridge(func::FuncOp func) {
  return func->hasAttr(KNPURuntimeBridgeAttr) ||
         func->hasAttr(KNPURuntimeCUDABridgeAttr);
}

static llvm::StringRef inferCApiName(func::FuncOp bridge) {
  if (bridge->hasAttr(KNPURuntimeCUDABridgeAttr))
    return KCUDAUnifiedLaunchAPI;
  return KH350UnifiedLaunchAPI;
}

void GenerateNorthStarRuntimeJsonPass::runOnOperation() {
  auto module = getOperation();
  SmallVector<func::FuncOp> bridges;
  module.walk([&](func::FuncOp func) {
    if (isRuntimeBridge(func))
      bridges.push_back(func);
  });

  if (bridges.empty())
    return;

  llvm::sort(bridges, [](func::FuncOp lhs, func::FuncOp rhs) {
    auto lhsId = lhs->getAttrOfType<IntegerAttr>(KNPURuntimeDescriptorIdAttr);
    auto rhsId = rhs->getAttrOfType<IntegerAttr>(KNPURuntimeDescriptorIdAttr);
    return lhsId.getInt() < rhsId.getInt();
  });

  auto moduleTarget = bridges.front()->getAttrOfType<StringAttr>(KNPUTargetAttr);
  if (!moduleTarget) {
    bridges.front().emitError("runtime bridge is missing target metadata");
    signalPassFailure();
    return;
  }

  llvm::json::Array descriptors;
  for (func::FuncOp bridge : bridges) {
    auto descriptorId =
        bridge->getAttrOfType<IntegerAttr>(KNPURuntimeDescriptorIdAttr);
    auto kernelName =
        bridge->getAttrOfType<StringAttr>(KNPURuntimeKernelNameAttr);
    auto target = bridge->getAttrOfType<StringAttr>(KNPUTargetAttr);
    auto deviceId = bridge->getAttrOfType<IntegerAttr>("device_id");
    if (!descriptorId || !kernelName || !target || !deviceId) {
      bridge.emitError("runtime bridge is missing descriptor metadata");
      signalPassFailure();
      return;
    }

    llvm::json::Object desc;
    desc["descriptor_id"] = descriptorId.getInt();
    desc["bridge"] = bridge.getSymName().str();
    desc["kernel_name"] = kernelName.getValue().str();
    desc["target"] = target.getValue().str();
    desc["device_id"] = deviceId.getInt();
    desc["num_inputs"] = static_cast<int64_t>(bridge.getNumArguments());
    desc["num_outputs"] = static_cast<int64_t>(bridge.getNumResults());
    desc["inputs"] = buildTensorDescArray(bridge.getFunctionType().getInputs());
    desc["outputs"] = buildTensorDescArray(bridge.getFunctionType().getResults());
    descriptors.push_back(std::move(desc));
  }

  llvm::json::Object root;
  root["target"] = moduleTarget.getValue().str();
  root["c_api"] = inferCApiName(bridges.front()).str();
  root["descriptors"] = std::move(descriptors);

  std::string json;
  llvm::raw_string_ostream os(json);
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(root)));
  os.flush();
  module->setAttr(KRuntimeJsonAttr, StringAttr::get(module.getContext(), json));
}

}  // namespace
}  // namespace mlir::north_star
