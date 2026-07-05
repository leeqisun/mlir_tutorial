#include <cstdint>
#include <memory>

#include "Dialect/NorthStar/IR/NorthStarAttrs.h"
#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "Dialect/NorthStar/Transforms/Passes.h"
#include "Interfaces/DistributeParallelismInterfaces.h"
#include "Utils/Key.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"

// 1) include 区新增
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"


#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir::north_star {
#define GEN_PASS_DEF_DEVICEREGIONFUSIONPASS
#include "Dialect/NorthStar/Transforms/Passes.h.inc"

}  // namespace mlir::north_star
using namespace ::mlir;
using namespace ::mlir::north_star;

namespace {

namespace {
static inline llvm::SmallString<4> getFusionName(
    mlir::ArrayRef<::mlir::Operation*> ops) {
  llvm::SmallString<4> name;
  for (auto op : ops) {
    name.append(op->getName().stripDialect());
    name.append("_");
    for (auto type : op->getOperandTypes()) {
      if (auto shaped = llvm::dyn_cast_or_null<ShapedType>(type)) {
        for (auto index : llvm::index_range(0, shaped.getRank())) {
          if (shaped.isDynamicDim(index)) {
            name.append("d_");
          } else {
            name.append(llvm::to_string(shaped.getDimSize(index)));
            name.append("_");
          }
        }
      }
    }
  }
  return name;
}

static inline int getDeviceid(mlir::ArrayRef<::mlir::Operation*> ops) {
  if (auto tensor = llvm::cast_or_null<north_star::NSTensorType>(
          ops.back()->getResultTypes().front())) {
    return tensor.getDeviceId();
  }
  llvm_unreachable("");
  return -1;
}

static inline llvm::MapVector<Value, std::pair<Operation*, int>>
getFusionInputs(mlir::ArrayRef<::mlir::Operation*> ops) {
  mlir::SetVector<Operation*> op_set(ops.begin(), ops.end());
  llvm::MapVector<Value, std::pair<Operation*, int>> res;
  for (auto op : ops) {
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (isa<BlockArgument>(operand))
        res[operand] = std::make_pair(nullptr, 0);
      if (op_set.contains(operand.getDefiningOp())) continue;
      res[operand] = std::make_pair(op, index);
    }
  }
  return res;
}

static inline llvm::MapVector<Value, std::pair<Operation*, int>>
getFusionOutputs(mlir::ArrayRef<::mlir::Operation*> ops) {
  mlir::SetVector<Operation*> op_set(ops.begin(), ops.end());
  llvm::MapVector<Value, std::pair<Operation*, int>> outs;
  for (auto op : ops) {
    for (auto [index, res] : llvm::enumerate(op->getResults())) {
      for (auto user : res.getUsers()) {
        if (op_set.contains(user)) continue;
        outs[res] = std::make_pair(op, index);
        break;
      }
    }
  }
  return outs;
}
}  // namespace
void FusionOps(::mlir::RewriterBase& rewriter,
               mlir::ArrayRef<::mlir::Operation*> ops, ::mlir::Location loc) {
  if (ops.empty()) return;

  auto context = rewriter.getContext();
  auto insertPoint = rewriter.saveInsertionPoint();

  auto name = getFusionName(ops);
  auto deviceId = getDeviceid(ops);
  name.append(llvm::to_string(deviceId));

  auto inputsMap = getFusionInputs(ops);
  auto outputsMap = getFusionOutputs(ops);
  if (outputsMap.empty()) return;

  llvm::SmallVector<Value> inputsVal;
  llvm::SmallVector<Value> outputsVal;
  llvm::SmallVector<Type> inputsType;
  llvm::SmallVector<Type> outputsType;

  for (auto [v, _] : inputsMap) {
    inputsVal.push_back(v);
    inputsType.push_back(v.getType());
  }
  for (auto [v, _] : outputsMap) {
    outputsType.push_back(v.getType());
  }

  rewriter.setInsertionPoint((*ops.begin())->getParentOp());
  auto kernel = rewriter.create<func::FuncOp>(
      loc, name, FunctionType::get(context, inputsType, outputsType));
  kernel->setAttr(KDeviceFunc, UnitAttr::get(context));
  Block *block = kernel.addEntryBlock();

  std::map<Operation *, Operation *> opMap;
  for (Operation *op : ops) {
    Operation *cloneOp = op->clone();
    block->push_back(cloneOp);
    opMap[op] = cloneOp;

    for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
      if (isa<BlockArgument>(operand)) continue;
      Operation *def = operand.getDefiningOp();
      if (!def || !opMap.count(def)) continue;
      opMap[op]->setOperand(
          idx, opMap[def]->getResult(cast<OpResult>(operand).getResultNumber()));
    }
  }

  for (auto [_, p] : outputsMap) {
    outputsVal.push_back(opMap[p.first]->getResult(p.second));
  }

  for (auto [idx, kv] : llvm::enumerate(inputsMap)) {
    Operation *consumer = kv.second.first;
    int operandIndex = kv.second.second;
    if (!consumer) continue;
    opMap[consumer]->setOperand(operandIndex, block->getArgument(idx));
  }

  rewriter.setInsertionPointToEnd(block);
  func::ReturnOp::create(rewriter, loc, ValueRange(outputsVal));

  rewriter.setInsertionPoint(insertPoint.getBlock(), insertPoint.getPoint());
  auto call = func::CallOp::create(rewriter, loc, kernel, ValueRange(inputsVal));

  for (auto [idx, kv] : llvm::enumerate(outputsMap)) {
    rewriter.replaceAllUsesWith(kv.first, call->getResult(idx));
  }

  for (Operation *oldOp : llvm::reverse(ops)) {
    if (oldOp->getBlock()) rewriter.eraseOp(oldOp);
  }
}


struct BufferCastOpDeviceRegionFusion
    : public OpRewritePattern<::mlir::north_star::BufferCastOp> {
  using OpRewritePattern::OpRewritePattern;

  virtual LogicalResult matchAndRewrite(::mlir::north_star::BufferCastOp op,
                                        PatternRewriter& rewriter) const  override {
    llvm::outs() << "match:" << getDebugName() << "\n";
    auto loc = op->getLoc();
    llvm::SmallVector<llvm::SetVector<Operation*>> op_list;
    for (auto res : op->getResults()) {
      rewriter.setInsertionPointAfterValue(res);
      llvm::SetVector<Operation*> ops;
      for (auto use : res.getUsers()) {
        addops(ops, use);
      }
      if (ops.size() != 0) op_list.push_back(ops);
    }
    if (op_list.size() == 0) return llvm::failure();
    for (auto ops : op_list) {
      FusionOps(rewriter, ops.takeVector(), loc);
    }
    return llvm::success();
  }

  void addops(llvm::SetVector<Operation*>& ops, Operation* op) const {
    if (!isa<DistributeParallelOp>(op)) return;
    ops.insert(op);
    for (auto user : op->getUsers()) {
      addops(ops, user);
    }
  }
};

struct BufferCastOpFold
    : public OpRewritePattern<::mlir::north_star::BufferCastOp> {
  using OpRewritePattern::OpRewritePattern;


  LogicalResult matchAndRewrite(::mlir::north_star::BufferCastOp op,
                              PatternRewriter &rewriter) const override {
  llvm::outs() << "match:" << getDebugName() << "\n";
  Operation *above_cast = nullptr;
  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    if (isa<BlockArgument>(operand)) return failure();
    if (!above_cast) {
      above_cast = operand.getDefiningOp();
    } else if (operand.getDefiningOp() != above_cast) {
      return failure();
    }
    if (!above_cast) return failure();
    if (operand.getType() != above_cast->getResult(index).getType())
      return failure();
    if (!above_cast->getResult(index).hasOneUse()) return failure();
  }

  for (auto [index, res] : llvm::enumerate(op->getResults())) {
    rewriter.replaceAllUsesWith(res, above_cast->getOperand(index));
  }
  rewriter.eraseOp(op);
  rewriter.eraseOp(above_cast);
  return success();
}


  // virtual LogicalResult match(::mlir::north_star::BufferCastOp op) const {
  //   llvm::outs() << "match:" << getDebugName() << "\n";
  //   Operation* above_cast = nullptr;
  //   for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
  //     if (isa<BlockArgument>(operand)) return llvm::failure();
  //     if (!above_cast) {
  //       above_cast = operand.getDefiningOp();
  //     } else {
  //       if (operand.getDefiningOp() != above_cast) return llvm::failure();
  //     }
  //     if (operand.getType() != above_cast->getResult(index).getType())
  //       return llvm::failure();
  //     if (!above_cast->getResult(index).hasOneUse()) return llvm::failure();
  //   }
  //   return llvm::success();
  // }

  // virtual void rewrite(::mlir::north_star::BufferCastOp op,
  //                      PatternRewriter& rewriter) const {
  //   Operation* above_cast = op->getOperand(0).getDefiningOp();
  //   for (auto [index, res] : llvm::enumerate(op->getResults())) {
  //     rewriter.replaceAllUsesWith(res, above_cast->getOperand(index));
  //   }
  //   rewriter.eraseOp(op);
  //   rewriter.eraseOp(above_cast);
  //   llvm::outs() << "match:" << getDebugName() << "\n";
  // }
};
}  // namespace

void ::mlir::north_star::populateDeviceRegionFusionPatterns(
    RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.addWithLabel<BufferCastOpDeviceRegionFusion>(
      StringRef("BufferCastOpDeviceRegionFusion"), context, 100);
};

void ::mlir::north_star::populateBufferCastOpCanonicalizationPatterns(
    RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.addWithLabel<BufferCastOpFold>(StringRef("BufferCastOpFold"),
                                          context, 2);
}

struct DeviceRegionFusionPass
    : ::mlir::north_star::impl::DeviceRegionFusionPassBase<
          DeviceRegionFusionPass> {
  using DeviceRegionFusionPassBase<
      DeviceRegionFusionPass>::DeviceRegionFusionPassBase;
  void runOnOperation() override;
};

void DeviceRegionFusionPass::runOnOperation() {
  llvm::outs() << "run in: " << getPassName() << "\n";
  auto module = getOperation();
  llvm::outs() << "root op: " << module->getName() << "\n";

  RewritePatternSet patterns(&getContext());

  // benefit 越大越先尝试
  patterns.addWithLabel<BufferCastOpDeviceRegionFusion>(
      StringRef("BufferCastOpDeviceRegionFusion"), &getContext(), 1);
  patterns.addWithLabel<BufferCastOpFold>(
      StringRef("BufferCastOpFold"), &getContext(), 10);

  GreedyRewriteConfig config;
  config.setMaxIterations(10);
// config.setUseTopDownTraversal(true);


  bool changed = false;
  if (failed(applyPatternsGreedily(
          getOperation(), FrozenRewritePatternSet(std::move(patterns)), config,
          &changed)))
    signalPassFailure();

  llvm::outs() << "region has changed: " << changed << "\n";
  llvm::outs() << "run out: " << getPassName() << "\n\n";
}
