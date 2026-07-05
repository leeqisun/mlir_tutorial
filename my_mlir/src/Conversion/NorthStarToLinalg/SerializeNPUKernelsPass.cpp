#include "Conversion/NorthStarToLinalg/NorthStarToLinalg.h"

#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "Utils/Key.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::north_star {

#define GEN_PASS_DEF_SERIALIZENORTHSTARNPUKERNELSPASS
#include "Conversion/Passes.h.inc"

namespace {

struct SerializeNorthStarNPUKernelsPass
    : public impl::SerializeNorthStarNPUKernelsPassBase<
          SerializeNorthStarNPUKernelsPass> {
  void runOnOperation() override;
};

static DenseI64ArrayAttr buildShapeAttr(Builder &builder, Type type) {
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType)
    return builder.getDenseI64ArrayAttr({});
  SmallVector<int64_t> shape(shapedType.getShape().begin(),
                             shapedType.getShape().end());
  return builder.getDenseI64ArrayAttr(shape);
}

static DictionaryAttr buildTensorDesc(Builder &builder, Type type) {
  NamedAttrList attrs;
  attrs.append("shape", buildShapeAttr(builder, type));
  if (auto shapedType = dyn_cast<ShapedType>(type))
    attrs.append("element_type", TypeAttr::get(shapedType.getElementType()));
  else
    attrs.append("element_type", TypeAttr::get(type));
  attrs.append("type", TypeAttr::get(type));
  return builder.getDictionaryAttr(attrs);
}

static ArrayAttr buildTensorDescArray(Builder &builder, TypeRange types) {
  SmallVector<Attribute> descs;
  descs.reserve(types.size());
  for (Type type : types)
    descs.push_back(buildTensorDesc(builder, type));
  return builder.getArrayAttr(descs);
}

void SerializeNorthStarNPUKernelsPass::runOnOperation() {
  auto module = getOperation();
  Builder builder(module.getContext());
  SmallVector<Attribute> kernels;

  module.walk([&](func::FuncOp func) {
    if (!func->hasAttr(KNPUKernelAttr))
      return;

    NamedAttrList attrs;
    attrs.append("name", func.getSymNameAttr());
    attrs.append("target", func->getAttr(KNPUTargetAttr));
    attrs.append("device_id", func->getAttr("device_id"));
    attrs.append("inputs",
                 buildTensorDescArray(builder, func.getFunctionType().getInputs()));
    attrs.append("outputs",
                 buildTensorDescArray(builder, func.getFunctionType().getResults()));
    kernels.push_back(builder.getDictionaryAttr(attrs));
  });

  module->setAttr(KSerializedKernelsAttr, builder.getArrayAttr(kernels));
}

}  // namespace
}  // namespace mlir::north_star
