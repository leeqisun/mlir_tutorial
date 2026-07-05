#include "Conversion/NorthStarToLinalg/NorthStarToLinalg.h"

#include <memory>

#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace mlir;
namespace {
struct SoftmaxOpToLinalgPattern final
    : public OpConversionPattern<mlir::north_star::SoftmaxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(north_star::SoftmaxOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  auto loc = op->getLoc();
  auto *convert = getTypeConverter();

  auto resType =
      llvm::dyn_cast_or_null<ShapedType>(convert->convertType(op.getType()));
  if (!resType) return failure();

  llvm::SmallVector<Value> outDySizes;
  auto input = adaptor.getInput();
  auto rank = resType.getRank();
  for (auto i : llvm::index_range(0, rank)) {
    if (!resType.isDynamicDim(i)) continue;
    auto dim = rewriter.create<tensor::DimOp>(loc, input, i);
    outDySizes.push_back(dim.getResult());
  }

  auto output = rewriter.create<tensor::EmptyOp>(
      loc, resType.getShape(), resType.getElementType(), outDySizes);
  auto newSoftmax = rewriter.create<linalg::SoftmaxOp>(
      loc, resType, adaptor.getInput(), output, adaptor.getAxis());
  auto decomposedResults = newSoftmax.decomposeOperation(rewriter);
  if (failed(decomposedResults) || decomposedResults->size() != 1)
    return failure();

  rewriter.replaceOp(op, decomposedResults->front());
  rewriter.eraseOp(newSoftmax);
  return success();
}

};

struct DeviceKernelOpConvertPattern final
    : public OpConversionPattern<mlir::north_star::DeviceKernelOp> {
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(north_star::DeviceKernelOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  auto loc = op.getLoc();

  SmallVector<Type> newResultTypes;
  if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), newResultTypes)))
    return failure();

  // 用转换后的 operands/results 建新 op
  auto newOp = rewriter.create<north_star::DeviceKernelOp>(
      loc, newResultTypes, adaptor.getSymName(), adaptor.getDeviceId(), adaptor.getArgs());

  // 关键：move region，不要 clone（避免“new illegal ops”）
  rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(), newOp.getRegion().end());

  // old result -> new result 的桥接 cast
  for (auto [oldRes, newRes] : llvm::zip(op.getResults(), newOp.getResults())) {
    auto cast = rewriter.create<UnrealizedConversionCastOp>(loc, oldRes.getType(), newRes);
    rewriter.replaceAllUsesWith(oldRes, cast.getResult(0));
  }

  rewriter.eraseOp(op);
  return success();
}

};
}  // namespace

namespace mlir::north_star {
namespace {

static Value materializeToNSTensor(OpBuilder &builder, NSTensorType type,
                                   ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(isa<RankedTensorType>(inputs[0].getType()));
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

static Value materializeToTensor(OpBuilder &builder, TensorType type,
                                 ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(isa<NSTensorType>(inputs[0].getType()));
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

}  // namespace
void initNorthStarToLinalgTypeConvert(TypeConverter &typeConverter) {
  typeConverter.addConversion([](NSTensorType type) {
    return RankedTensorType::get(type.getShape(), type.getElementType());
  });
  typeConverter.addSourceMaterialization(
      [&](OpBuilder &builder, Type resultType, ValueRange inputs,
          Location loc) -> Value {
        if (inputs.size() != 1) return Value();

        return builder
            .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
      });
  typeConverter.addTargetMaterialization(
      [&](OpBuilder &builder, Type resultType, ValueRange inputs,
          Location loc) -> Value {
        if (inputs.size() != 1) return Value();

        return builder
            .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
      });
}
void populateNorthStarToLinalgPatterns(TypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {

  patterns.add<SoftmaxOpToLinalgPattern, DeviceKernelOpConvertPattern>(typeConverter, patterns.getContext());

};
}  // namespace mlir::north_star
